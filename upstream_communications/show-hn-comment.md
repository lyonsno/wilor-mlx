# Show HN Comment — wilor-mlx

Status: final draft, pending operator go
Surface: Hacker News (Show HN)
Link: https://github.com/lyonsno/wilor-mlx
Title: Show HN: WiLoR hand pose rebuilt in MLX for Apple Silicon
Target: Wednesday 2026-06-10 morning ET (or operator discretion)
Gate: operator posts manually

---

I've been doing a lot of local inference work recently, and also a lot of dictation, which has had the funny side effect of making keyboards feel more and more like legacy hardware.

So, like everybody who is occasionally feeling both imaginative and lazy, I started daydreaming about holographic gesture interfaces.

I went looking for hand tracking on Mac that could run at something close to real-time, and the best thing I found was WiLoR-mini. Since I'd already been working with MLX, I used it to rebuild the WiLoR-mini reconstruction model end-to-end — ViT-H/16 backbone, MANO hand model, and RefineNet — so the pose/reconstruction stage can run natively on Apple Silicon without PyTorch at inference time.

It was not magic. But it did help. On my M4 Max, a clean same-harness smoke over recent Perceptasia saved frames puts the MLX pose/reconstruction model stage at about 37ms median versus 49ms for PyTorch MPS, and the full saved-frame route at about 49ms versus 60ms. That is the number I care about here: not a one-off batch benchmark, but the route that turns camera frames into hand-pose events for an actual interaction loop.

I originally reached for MLX because older app-level telemetry around the PyTorch MPS route made the hand tracker feel less reliable than I wanted. Clean reruns changed that comparison denominator enough that I do not want to sell this as a universal PyTorch-vs-MLX tail-collapse number. The stronger and simpler claim is that WiLoR-mini now has a native MLX runtime on Apple Silicon, with live sidecar latency low enough to build interaction on.

That flatness is the difference between a hand tracker that feels impressive in bursts and one that can plausibly act as an input device.

The reason appears to be mostly dispatch and synchronization, not memory copies: both routes sit on Apple Silicon unified memory. In this workload, PyTorch's eager MPS path exposes many per-op Metal submissions and sync boundaries that can block behind other GPU work. MLX builds a lazy graph and evaluates it in fewer, fused submissions. At least, that's where our traces pointed; the benchmark does not depend on that being the whole story.

So some problem spaces are suddenly more viable.

Setup is one line:

    model = WiLoR.from_pretrained()

First run needs torch for a one-time MANO conversion from the upstream WiLoR-mini checkpoint, or you can pass your own MANO data via `mano_path`. The weights we publish do not bundle or rehost MANO. After conversion/cache, inference is pure MLX.

I couldn't find another public MLX or Core ML port of WiLoR-mini when I looked, but if I missed one, let me know.

Float32 and int4 weights are up on Hugging Face. Int4 is mostly a download-size win: about 490MB instead of 2.4GB. It does not really run faster here, because at the sequence length WiLoR-mini actually uses, the model appears to be compute-bound rather than bandwidth-bound.

WiLoR is by Rolandos Alexandros Potamias, Jinglei Zhang, Jiankang Deng, and Stefanos Zafeiriou:

    https://github.com/rolpotamias/WiLoR

WiLoR-mini is warmshao's lighter-weight package around it:

    https://github.com/warmshao/WiLoR-mini

All modeling credit is theirs; this is a runtime rebuild. I've opened an issue on the upstream repo to let them know it exists. Big thank you to both teams for releasing the code and weights.

One caveat: the M4 Max numbers are the cleanest story. On my M2 Pro validation box, current Tahoe reruns over the same frozen frame corpus are much faster than my earlier numbers for both MLX and PyTorch/MPS; MLX still wins p50/p90/p95, but the tail still has outliers. I'm treating those as rebaseline numbers, not as a Metal 4/TensorOps claim.

The thing I'm most curious about next is M5. I don't have access to one yet, and the numbers above are M4 Max only. If someone with an M5 Air or M5 Pro wants to run the benchmark script from the repo, I would genuinely love to see the results.

Repo: https://github.com/lyonsno/wilor-mlx
