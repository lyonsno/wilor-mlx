# Show HN Comment — wilor-mlx

Status: final draft, pending operator go
Surface: Hacker News (Show HN)
Link: https://github.com/lyonsno/wilor-mlx
Title: Show HN: WiLoR hand pose in MLX - p99 427ms to 66ms on Apple Silicon
Target: Wednesday 2026-06-10 morning ET (or operator discretion)
Gate: operator posts manually

---

We've been doing a lot of local inference work recently, and also a lot of dictation, which has had the funny side effect of making keyboards feel more and more like legacy hardware.

So, like everybody who is occasionally feeling both imaginative and lazy, we started daydreaming about holographic gesture interfaces.

We went looking for hand tracking on Mac that could run at something close to real-time, and the best thing we found was WiLoR-mini. Since we'd already been working with MLX, we rebuilt the full WiLoR-mini pipeline end-to-end in MLX — ViT-H/16 backbone, MANO hand model, and RefineNet — so it can run natively on Apple Silicon without PyTorch at inference time.

It was not magic. But it did help.

Isolated model call, M4 Max: median 50ms (PyTorch MPS) → 36ms (MLX).

But the real win was the tails, measured in a live capture loop:

            PyTorch MPS    MLX
    p50     85ms           61ms
    p90     144ms          62ms
    p95     238ms          63ms
    p99     427ms          66ms

(benchmarked on M4 Max 40 Core 128GB Unified)

The MLX p99 is faster than the PyTorch median. That flatness is
the whole difference between "tech demo" and "feels like an input
device" — MPS would mostly be fine, then randomly hitch hard
enough to break the illusion.

The reason is mostly architectural: PyTorch MPS has to shuttle tensors between CPU and GPU memory, and those transfers can stall behind whatever else the GPU is doing. MLX builds a lazy computation graph on unified memory and evaluates it in one shot. There's nothing to stall on, so the latency stays flat.

So some problem spaces are suddenly more viable.

Setup is one line:

    model = WiLoR.from_pretrained()

First run needs torch for a one-time MANO conversion. After that, inference is pure MLX.

We couldn't find another public MLX or Core ML port of WiLoR-mini when we looked, but if we missed one, let us know.

Float32 and int4 weights are up on Hugging Face. Int4 is mostly a download-size win: about 490MB instead of 2.4GB. It does not really run faster here, because at the sequence length WiLoR-mini actually uses, the model appears to be compute-bound rather than bandwidth-bound.

WiLoR is by Rolandos Alexandros Potamias, Jinglei Zhang, Jiankang Deng, and Stefanos Zafeiriou:

    https://github.com/rolpotamias/WiLoR

WiLoR-mini is warmshao's lighter-weight package around it:

    https://github.com/warmshao/WiLoR-mini

All modeling credit is theirs; this is a runtime rebuild. We've opened an issue on the upstream repo to let them know it exists. Big thank you to both teams for releasing the code and weights.

One caveat: this is much more interesting on our M4 Max than on our M2 Pro. The M2 Pro still benefits from MLX, but it lands closer to the few-hundred-millisecond range with a worse tail.

We haven't benchmarked M5 yet. Since this port stays inside MLX primitives, it should be well-positioned to pick up MLX/Metal backend improvements on newer Apple GPUs, but the numbers above are M4 Max measurements only.

Repo: https://github.com/lyonsno/wilor-mlx
