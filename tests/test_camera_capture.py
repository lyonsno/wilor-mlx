"""Camera output survives the real pipeline serialization boundary; no model load."""
from dataclasses import asdict
from unittest.mock import patch
import json
import numpy as np
import mlx.core as mx
import pytest
from wilor_mlx.pipeline import HandPosePipeline

@pytest.mark.parametrize('side',['left','right'])
def test_pipeline_retains_camera_and_model_projection(side):
    class Detector:
        def __call__(self, image):
            raw=np.zeros((1,69,1),dtype=np.float32)
            raw[0,:4,0]=[208,216,80,64]
            raw[0,4+(side=='right'),0]=.9
            raw[0,6:,0]=np.tile([208,216,1],21)
            return mx.array(raw)
    class Pose:
        FOCAL_LENGTH=5000.
        def __call__(self,image):
            return {'pred_cam':mx.array([[2.,.04,-.02]]),
                    'pred_keypoints_3d':mx.array(np.tile([.02,.01,.03],(1,21,1))),
                    'pred_vertices':mx.zeros((1,778,3))}
    with patch('wilor_mlx.pipeline._crop_and_resize_mlx',return_value=mx.zeros((2,2,3),dtype=mx.uint8)):
        hands=HandPosePipeline(Detector(),Pose())(np.zeros((480,640,3),dtype=np.uint8),include_vertices=True)
    saved=json.loads(json.dumps(asdict(hands[0]),allow_nan=False))
    assert saved['camera']['status']=='observed'
    assert saved['camera']['rawPredCam']==pytest.approx([2,.04,-.02])
    assert saved['camera']['imageSize']==[640,480]
    assert saved['camera']['intrinsics']==[12500.,12500.,320.,240.]
    # This rectangle's detector center maps to(260,190), size100*2.5=250.
    tx=(-.04 if side=='left' else .04)+2*(260-320)/(250*2+1e-9)
    expected=[tx,-.02+2*(190-240)/(250*2+1e-9),25000/(250*2+1e-9)]
    assert saved['camera']['translation']==pytest.approx(expected)
    j=np.array(saved['keypoints_3d'])+expected
    projected=j[:,:2]/j[:,2:]*12500+[320,240]
    assert np.allclose(saved['model_keypoints_2d'],projected)
    assert saved['model_keypoints_2d']!=saved['keypoints_2d']
    assert len(saved['vertices'])==778


def test_detector_only_does_not_invent_camera():
    from wilor_mlx.pipeline import HandPose
    saved=asdict(HandPose('right',.9,[0,0,1,1],[[0,0]]*21))
    assert saved['camera'] is None
    assert saved['model_keypoints_2d'] is None
