# Show HN Comment — wilor-mlx

Status: final draft, pending operator go
Surface: Hacker News (Show HN)
Link: https://github.com/lyonsno/wilor-mlx
Title: Show HN: WiLoR hand pose in MLX - p99 427ms to 66ms vs PyTorch MPS
Target: Wednesday 2026-06-10 morning ET (or operator discretion)
Gate: operator posts manually

---

I've been doing a lot of local inference work recently, and also a lot of dictation, which has had the funny side effect of making keyboards feel more and more like legacy hardware.

So, like everybody who is occasionally feeling both imaginative and lazy, I started daydreaming about holographic gesture interfaces.

I went looking for hand tracking on Mac that could run at something close to real-time, and the best thing I found was WiLoR-mini. Since I'd already been working with MLX, I used it to rebuild the WiLoR-mini reconstruction model end-to-end — ViT-H/16 backbone, MANO hand model, and RefineNet — so the pose/reconstruction stage can run natively on Apple Silicon without PyTorch at inference time.

It was not magic. But it did help. The win I trust most is the tail behavior in a live capture loop:

            PyTorch MPS    MLX
    p50     85ms           61ms
    p90     144ms          62ms
    p95     238ms          63ms
    p99     427ms          66ms

(benchmarked on M4 Max, 40-core GPU, 128GB unified memory; MLX 0.31.2; PyTorch MPS row from 2.5.0 telemetry)

The MLX p99 is faster than the PyTorch median. That flatness is
the difference between a hand tracker that feels impressive in bursts and one that can plausibly act as an input device — MPS would mostly be fine, then randomly hitch and drop out from under me.

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
