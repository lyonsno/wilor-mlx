# Show HN Comment — wilor-mlx

Status: final draft, pending operator go
Surface: Hacker News (Show HN)
Link: https://github.com/lyonsno/wilor-mlx
Title: Show HN: WiLoR hand pose estimation rebuilt in MLX for Apple Silicon
Target: Wednesday 2026-06-10 morning ET (or operator discretion)
Gate: operator posts manually

---

I've been doing a lot of local inference work recently, and also a lot of dictation, which has had the funny side effect of making my keyboard act more and more like legacy hardware.

So, like everybody who is occasionally feeling both imaginative and lazy, I started daydreaming about holographic gesture interfaces.

I went looking for hand tracking on Mac that could run at something close to real-time, and the best thing I found was WiLoR-mini. Since I'd already been working with MLX, I rebuilt the full WiLoR-mini pipeline end-to-end in MLX — ViT-H/16 backbone, MANO hand model, and RefineNet — so it can run natively on Apple Silicon without PyTorch at inference time.

It was not magic. But it did help.

On isolated benchmarks, median latency dropped from 50ms to 36ms.

The bigger win was tail latency. That was the thing actually killing the interactive feel. PyTorch MPS would mostly be okay, then randomly hitch badly enough that the illusion fell apart. On my M4 Max, p95 went from 238ms to about 61ms, and p99 went from 427ms to about 61ms.

That's the difference between something that lags once or twice a second and something that basically feels stable.

Setup is one line:

  `model = WiLoR.from_pretrained()`

First run needs torch for a one-time MANO conversion. After that, inference is pure MLX.

I couldn't find another public MLX or Core ML port of WiLoR-mini when I looked, but if I missed one, let me know.

Float32 and int4 weights are up on Hugging Face. Int4 is mostly a download-size win: about 490MB instead of 2.4GB. It does not really run faster here, because at the sequence length WiLoR-mini actually uses, the model appears to be compute-bound rather than bandwidth-bound.

One caveat: this is much more interesting on my M4 Max than on my M2 Pro. The M2 Pro still benefits from MLX, but it lands closer to the few-hundred-millisecond range with a worse tail. I'm not sure yet how base M4 or M5 configurations will fare.
