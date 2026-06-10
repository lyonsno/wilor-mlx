# A Supposedly Fun Thing I'll Never Port Again

## Or: How a Fused Metal Kernel Ambition Became a Hand Pose Estimation Library, and What That Says About the Actual Structure of Useful Work

---

It is, I think, worth stating at the outset that the project we are about to describe in what will probably be uncomfortable detail was not the project anyone set out to build. The project anyone set out to build was called, with the kind of nominative ambition that tends to precede a very specific flavor of humiliation, "Flash Attention 3 for Metal Compute Shaders." The project that actually got built is called `wilor-mlx`, and it estimates the three-dimensional pose of a human hand from a single RGB image on an Apple Silicon Mac in about sixty-one milliseconds, with a consistency of latency so flat and so tight that you could reasonably mistake the benchmark chart for a rendering error. These are not the same project. The distance between them is the entire lesson.

---

### I. The Ambition, and Its Immediate Collision with Prior Art

The original thesis was beautiful in the way that theses are beautiful right before they encounter reality, which is to say it was beautiful in exactly the way that matters least. The thesis was: Apple Silicon GPUs are good, and getting better, and the Metal Shading Language is a capable compute shader environment, and the transformer workloads running on these GPUs via frameworks like MLX are paying a brutal per-dispatch overhead tax --- on the order of one millisecond per Metal command buffer dispatch[^1] --- that is completely invisible in the LLM context where everyone benchmarks everything (because when you're generating tokens autoregressively over a 4096-length context, one millisecond of dispatch overhead per transformer block is noise) but absolutely catastrophic in the short-context vision transformer context where a ViT-H/16 backbone has 32 layers and 210 tokens and each layer does about 12 to 15 Metal dispatches, meaning you're spending roughly 400 to 480 dispatches on operations that, if you fused them into single blocks, could in theory run as 32 to 64 dispatches, which would recover maybe a few to tens of milliseconds from a clean inference path. That is the kind of difference that can separate "this feels like a tracking system" from "this feels like a suggestion system."

[^1]: The dispatch overhead investigation is preserved in `benchmarks/dispatch_overhead.py`, which is the kind of file that exists to prove that you did the homework before you gave up. The benchmark measures six scenarios: single block, full 32-block stack, single matmul at per-block FLOP scale, 32 sequential matmuls, 32 chained matmuls in a single eval, and 32 sequential evals of the same matmul. The gap between "32 chained matmuls in one graph" and "32 sequential evals" gives you the per-eval dispatch overhead. On an M4 Max it's around 0.1ms per eval boundary --- not the 1ms initially feared. The real overhead is inside the graph: all the LayerNorm, SDPA, GELU, residual, and reshape operations that can't be trivially eliminated by fusing. The conclusion --- which took a day of benchmarking to reach and about thirty seconds to accept once reached --- was that fused kernels would not substantially help at this token count. The dispatch overhead is real but it is not the bottleneck. The bottleneck is that the operations themselves, at 210 tokens and 1280 dimensions, are just small enough that the GPU spends a nontrivial fraction of its time on scheduling and memory management regardless of how you organize the dispatch.

So: Flash Attention 3 for Metal. We read the paper. We wrote SIMD shuffle benchmarks. We got a 2.14x speedup on the shuffle microbenchmark, felt the warm glow of being clever, and then discovered --- in the exact way that discovery works when it's actually useful, which is to say it was annoying and deflating --- that a person named Philip Turner had already done essentially all of the interesting Metal optimization work for Apple Silicon attention kernels, and that MLX itself already used `simd_shuffle_xor` internally, and that our shuffle benchmark was measuring our ability to rediscover what was already shipping in production.

This is the part of the story that in a normal technical blog post would be elided or rewritten as "we surveyed the existing work and identified a gap." We did survey the existing work. The gap we identified was the gap between our ambition and reality, which is a gap that exists in all technical work and which everyone conspires to pretend doesn't exist because acknowledging it would require admitting that the most common experience in engineering is arriving at a place someone else already occupies.

### II. The Pivot, or: The Target Becomes the Product

Here is what happened next, and it happened in the way that real pivots happen, which is without a meeting and without a strategy document and with the quiet shame of someone who realizes they've been optimizing the wrong function.

The plan had been: build fused Metal kernels targeting WiLoR-mini's ViT-H/16 backbone. WiLoR-mini[^2] is a 3D hand pose estimation model --- Zhan et al., "WiLoR: End-to-end 3D hand localization and reconstruction in-the-wild," CVPR 2025 --- and it was already running inside Perceptasia, a local hand-tracking system that we'll discuss in due course, via PyTorch MPS. The ViT backbone was the hottest part of the pipeline. The fusion target was clear: take 32 transformer blocks, each dispatching 12-15 Metal commands, and collapse them into 32 single-dispatch fused blocks. The WiLoR port to MLX was supposed to be the *baseline*. The thing you build so you have an honest before-and-after comparison for the fused kernel paper.

[^2]: WiLoR-mini is the compact variant of WiLoR. It uses a ViT-H/16 backbone with 1280 embedding dimension, 32 transformer layers, 16 attention heads, and a 4x MLP ratio. The input is a 256x192 image crop (256 wide, 192 after cropping 32 pixels from each side of a 256x256 input) that gets patchified into 192 image tokens via 16x16 patches. To these 192 tokens the model prepends 16 pose tokens, 1 shape token, and 1 camera token, for a total sequence length of 210. The model predicts 6D rotation representations that get converted to 3x3 rotation matrices via Gram-Schmidt orthogonalization, and these rotations drive a MANO hand model through linear blend skinning to produce 778 mesh vertices and 21 hand keypoints in 3D. If this paragraph made sense to you, you are the target audience. If it didn't, the short version is: it turns a picture of a hand into a 3D mesh of a hand, and it does it at a scale where "milliseconds matter" is not a figure of speech.

The baseline became the product because, and I want to be precise about this because the causality is important: the MLX port had already changed the latency shape that mattered in the live system without any fused kernels.

Not because of a clean microbenchmark miracle. The originally drafted isolated benchmark looked modest, and we have since pulled it from launch copy: the 36ms number appears to have been ViT-backbone-only, while current full-model MLX reruns land closer to roughly 61-68ms. That correction does not change the thing that mattered. The live route was never won by a tidy isolated benchmark. It was won by tail behavior in the sidecar.

But here is the thing that turns a modest isolated speedup into something that actually changes what a system can do. And it requires a small digression about tail latency, which is a small digression that is actually the entire point.

### III. The Tail Is the Point, or: Why p99 Is the Only Number That Matters for Control Systems

PyTorch MPS, on Apple Silicon, in a live application context where the GPU is also compositing your windows and rendering your browser and doing whatever else macOS asks it to do, has tail latency that would make a distributed systems engineer weep. The 102,000-row telemetry history from the Perceptasia hand tracking sidecar --- and we should note here that having 102,000 rows of production telemetry for your hand tracking system is either admirably thorough or clinically obsessive, and the distinction between those two things is a function of whether the telemetry turns out to be useful, and it turned out to be useful --- shows PyTorch MPS at:

- p50: ~85ms
- p90: ~144ms
- p95: ~238ms
- p99: ~427ms

That's a 5x spread from p50 to p99. It means that one frame in a hundred takes almost half a second. If you're running hand tracking at anything approaching real-time --- even a modest 10-15 FPS --- you will hit that p99 tail *multiple times per minute*. Each time you hit it, the tracking stutters. The hand jumps. The gesture grammar loses coherence. The user's brain, which is exquisitely sensitive to the temporal coherence of visual feedback systems, registers the stutter as "this doesn't really work."

MLX, in the same live sidecar, during a stable 500-frame window of uncontended operation[^3]:

- p50: ~61ms
- p90: ~62ms
- p95: ~63ms
- p99: ~66ms

That's an 8% spread from p50 to p99. Not 8x. Eight percent. The tail essentially doesn't exist. Every frame costs roughly the same thing. The tracking doesn't stutter because there's nothing to stutter *on*. Our current read is dispatch and synchronization, not memory transfer: both routes sit on Apple Silicon unified memory, but MLX's lazy graph gives the hot path fewer places for a hitch to land.

[^3]: The "stable window" qualifier is important and we are going to be honest about it because the alternative is getting caught being dishonest about it. The full-run MLX sidecar telemetry across 1,335 frames shows p90 ~120ms, p95 ~129ms, p99 ~195ms. Those numbers are real. They are also contaminated by concurrent workstation load: WebGPU renders, other MLX generation tasks, and general macOS GPU contention. The stable-window numbers isolate the model's actual behavior from the workstation's behavior. We chose to report both and explain the difference, rather than pick whichever number made us look better and hope nobody asked. This is, it turns out, a reasonable strategy, because the people who will actually use your library are exactly the people who will ask.

This is the thing that a paired isolated benchmark failed to capture even before we pulled the old number from launch copy. An isolated benchmark runs on a warm GPU with nothing else happening. In isolation, both backends can be fast enough and consistent enough. It's in production that the difference becomes qualitative rather than quantitative --- the difference between "a tracking system" and "a suggestion system."

And this is why the fused kernel project died. Not because fused kernels wouldn't help --- they would, marginally, by reducing dispatch overhead and graph compilation cost. But because the actual problem was already solved by the simpler, dumber intervention of just porting the model to a framework whose execution model fit the route. The sophisticated optimization would have recovered a bounded slice from a clean inference. The unsophisticated port recovered the entire tail distribution. Sometimes the right move is the boring one.

### IV. What "Porting a Model" Actually Means, or: The Boring Part That Contains All the Difficulty

Let me tell you what a "port" consists of, because the word "port" implies something mechanical and straightforward, like moving furniture from one room to another, and what it actually involves is more like translating a novel from Russian to Japanese while preserving the meter.

WiLoR-mini is written in PyTorch. MLX is not PyTorch. They share surface-level similarities --- both have tensor operations, both have neural network modules, both have automatic differentiation --- but the details diverge in ways that are trivially enumerable and non-trivially consequential.

**Layout conventions.** PyTorch convolutions expect NCHW tensors (batch, channels, height, width). MLX convolutions expect NHWC (batch, height, width, channels). This is not a hard problem. It is a problem that must be solved correctly in approximately forty places, and the consequence of solving it incorrectly in one place is that the model produces plausible-looking but numerically wrong output, which is the worst kind of bug because it doesn't crash.

**Weight transposition.** PyTorch Conv2d stores weights as (out_channels, in_channels, kernel_h, kernel_w). MLX stores them as (out_channels, kernel_h, kernel_w, in_channels). PyTorch ConvTranspose2d stores weights as (in_channels, out_channels, kernel_h, kernel_w). These must be permuted during weight loading, and the permutation for ConvTranspose2d is different from the permutation for Conv2d, and getting this wrong produces output that is wrong in ways that are detectable only by numerical comparison against a reference implementation that you hope you've set up correctly.

**BatchNorm parameter mapping.** PyTorch BatchNorm has `weight`, `bias`, `running_mean`, and `running_var`. MLX BatchNorm has the same names but different initialization conventions. During inference (which is all we do --- this is a port for inference, not training), the running statistics must be loaded correctly. This is boring. It is also mandatory.

**The MANO hand model.** This is the part that is not boring. MANO[^4] is a differentiable parametric hand mesh model that takes shape parameters and joint rotations and produces a posed 3D mesh through linear blend skinning. Porting it requires implementing Rodrigues rotation (axis-angle to rotation matrix), blend shapes (per-vertex displacement from shape parameters), joint regression (3D joint locations from vertex positions), rigid body transform chaining along a kinematic tree with parent indices, and the full linear blend skinning procedure: compute per-joint transformation matrices, chain them along the kinematic tree, weight them by per-vertex skinning weights, and apply the weighted transforms to the posed vertices in homogeneous coordinates.

[^4]: MANO stands for "hand Model with Articulated and Non-rigid defOrmations." It is from the Max Planck Institute. Its licensing will become relevant later in a way that caused approximately twelve hours of intense strategic negotiation among synthetic agents, which is a sentence I never expected to write and which I stand behind completely.

None of this is conceptually hard. All of it is a minefield of indexing errors, broadcasting mistakes, and coordinate convention mismatches. The kinematic chain traversal, for instance, requires converting parent indices from an MLX array to a Python list for indexing (because MLX array indexing with another array doesn't work the way you'd expect for sequential parent-chain lookups), and then iterating over joints in order, building up transformation matrices by multiplying each joint's local transform with its parent's accumulated transform. One indexing error and the wrist rotation propagates to the wrong finger.

**Bilinear grid sampling.** RefineNet, the refinement stage, needs to sample ViT features at projected vertex locations. PyTorch has `F.grid_sample`. MLX does not. So you write your own: convert normalized coordinates to pixel coordinates, find the four neighboring pixels, compute bilinear weights, gather the pixel values, and interpolate. This is a textbook operation. Writing it from scratch, with correct handling of `align_corners=True` semantics and correct NCHW indexing via `take_along_axis`, takes about an hour and requires getting every dimension correct.

**Perspective projection.** Building a 3x3 camera intrinsics matrix in MLX without item assignment (because MLX arrays are immutable) requires stacking rows from individually constructed vectors. This is the kind of thing that makes you realize how much PyTorch's `K[0, 0] = focal_length` convenience hides.

**The 6D rotation representation.** WiLoR uses the continuous 6D rotation representation (Zhou et al., 2019) which takes two 3D vectors, Gram-Schmidt orthogonalizes them, and cross-products them to get a rotation matrix. The reverse --- rotation matrix back to axis-angle for output --- uses the trace-based angle recovery and skew-symmetric axis extraction from Rodrigues' formula, with a degenerate case near 180 degrees that doesn't matter for hand poses (your fingers don't bend 180 degrees, barring circumstances you don't want to think about) but that requires documenting so that someone reading the code doesn't file a bug report about it.

All of this is in five files totaling maybe a thousand lines of code. It took several days. The bulk of the difficulty is not in any individual operation but in the combinatorial surface area of operations that must all be correct simultaneously.

### V. The Fifteen Reviews, or: How to Find Seven Bugs by Arguing with Yourself

Here is the part of the story that I find most interesting and that I suspect will be most controversial, and I want to describe it precisely because the details matter.

After the initial port was working --- producing output that looked reasonable, passed basic sanity checks, and matched PyTorch within a max absolute difference of about 0.024 --- it was subjected to fifteen independent adversarial code reviews. These reviews were conducted by AI agents (Claude and Gemini instances), each given a specific code slice and asked to find bugs. Ten reviews in the first round. Five targeted reviews in the second round.

They found seven bugs. Let me enumerate them because the enumeration is instructive.

**1. PatchEmbed padding: 4 to 2.** This was the critical one. The PyTorch WiLoR code computes the Conv2d padding for PatchEmbed using a formula: `padding = 4 + 2*(ratio//2 - 1)`, where `ratio=1`. Expanding this: `4 + 2*(0 - 1) = 4 + 2*(-1) = 4 - 2 = 2`. The initial port had `padding=4`. Someone --- and by "someone" I mean the agent doing the initial port --- looked at the formula, saw the `4` at the front, and wrote `padding=4` without fully expanding the expression. This single arithmetic error propagated through every subsequent operation in the model, produced a systematically wrong spatial embedding, and caused approximately 3x more numerical divergence than necessary. Fixing it dropped the max diff from 0.019 to 0.006. This bug was found not in the first round of reviews but in the second round, by a targeted "abstraction boundary" probe that specifically examined how PyTorch helper class parameters were expanded into hardcoded MLX equivalents. The first-round structural review missed it because the structural review was checking that the *structure* was right (Conv2d with padding? check.) without verifying the *parameter value* (padding=what?).

The lesson here is one that anyone who has done code review already knows, which is that the bugs that survive structural review are the bugs that hide inside correct-looking structures, and the way to find them is to go back to the source computation and re-derive the parameter from scratch.

**2-4. RefineNet activation bugs (three bugs, one root cause).** The PyTorch RefineNet uses `nn.Sequential` containers that chain together ConvTranspose2d, BatchNorm, and ReLU layers. When porting to MLX, these Sequential containers were expanded into explicit operations. In the expansion, three ReLU activations were either misplaced (applied after `first_conv` where PyTorch uses `bnrelu_final=False` to suppress it) or omitted (missing ReLU after BatchNorm in branch 0 and the first layer of branch 1). The pattern is the same as the PatchEmbed bug: the abstraction boundary between "what the PyTorch class does internally" and "what the expanded MLX code does explicitly" is exactly where port errors cluster.

**5. ConvTranspose2d bias=False.** The deconv layers in RefineNet use `bias=False`. The initial port didn't set this. Since the weights had no bias tensors, this manifested as zero-initialized biases being used instead of no bias at all. Functionally similar in float32 but structurally wrong.

**6. _collect_arrays nn.Module traversal.** This one is about Metal shared events, which is a sentence that requires context. When loading a model in MLX, you want to `mx.eval()` all the arrays to push them to the GPU. If you eval too many arrays at once, you can exhaust Metal's shared event pool, which causes a crash. The solution is to batch the evals. The function that collects all arrays from the model tree, `_collect_arrays`, initially didn't handle `nn.Module` subclasses correctly --- it was walking `__dict__` instead of using the `items()` method that `nn.Module` provides for iterating over parameters. This meant some arrays weren't being collected, which meant they weren't being batched, which meant that under concurrent GPU pressure (other MLX sessions, Trellis2MLX generation, WebGPU renders), the model loading could crash with a Metal shared event exhaustion error.

**7. rotmat_to_rotvec 180-degree edge case.** Near angle=pi, both sin(angle) and the skew-symmetric axis extraction approach zero, making the Rodrigues inversion numerically unstable. For hand poses this is irrelevant (hand joints don't reach 180 degrees), but it was annotated as a known limitation with a note about why it doesn't matter for the specific application.

Here is what I find interesting about this list: all seven bugs are at abstraction-translation boundaries. The op-for-op translations --- the attention mechanism, the MLP, the LayerNorm, the MANO linear blend skinning --- had zero bugs. When you translate "do this matrix multiply then add this bias" from PyTorch to MLX, there's essentially one way to do it and it's the same way in both frameworks. But when you translate "construct a PyTorch Sequential containing ConvTranspose2d(640, 320, 4, 2, 1, bias=False), BatchNorm(320), and ReLU, then do the forward pass" into the equivalent explicit MLX operations, you're expanding an abstraction, and every expansion is a chance to drop a detail.

This suggests a testing methodology: don't review the code that translates operations; review the code that translates *configurations*.

### VI. The MANO Crisis, or: How We Discovered We Were Accidentally Violating a License and What We Did About It

Partway through the launch preparation --- which I'll describe in Section VIII, and which involved a number of synthetic agents with names like Kynormous, Badgestall, and Coherexivity arguing about publication strategy in a shared coordination document that is approximately 25,000 words long and reads like the minutes of a parliamentary subcommittee that has achieved sentience and regrets it --- partway through this process, one of the agents (Badgestall, whose operational concern is source-custody and public-contact integrity) noticed something.

The MANO hand model data was in our HuggingFace weights file.

Under MIT license.

The MANO hand model is licensed by the Max Planck Institute under terms that explicitly prohibit redistribution without prior written permission. The license is non-commercial, non-transferable, and the data may not be "made available to third parties." Our safetensors weight file contained seven MANO arrays --- v_template, shapedirs, posedirs, J_regressor, lbs_weights, and two indexing arrays --- totaling about 1.4 megabytes of data that we had no right to distribute.

This is the kind of mistake that, in a normal project, would be discovered either by a lawyer months after launch or by an angry email from MPI. It was discovered by a review agent whose job was to audit the public surface before launch, and the discovery happened before any public announcement, which means the window of exposure was limited to whoever happened to clone the repository or download the weights during the few days they were up.

The remediation was not subtle:

1. Delete the entire HuggingFace repository. Not the weights file. The repository. Because HuggingFace preserves git history, and git history preserves the offending commit.
2. Recreate the repository with clean history. No commit in the new repo has ever contained MANO data.
3. Strip MANO arrays from both float32 and int4 weight files. Re-upload.
4. Restructure the loading code so that MANO data is obtained separately, either from the user's own copy or by converting from the upstream WiLoR-mini checkpoint.
5. Add explicit MANO license documentation to the README and model card.

This led to the API design fight that consumed the next several hours.

### VII. The Gate A / Gate B Negotiation, or: How Many Arguments Can You Have About One Function Signature

The question was: what does `WiLoR.from_pretrained()` look like?

**Gate A** (conservative): `WiLoR.from_pretrained(mano_path="weights/mano.npz")`. The user must explicitly provide a path to MANO data. This is clean from a licensing perspective --- the user obtained MANO through MPI's official channel and is responsible for their own compliance. But it means the setup flow is:

1. Clone the repo
2. Register with MPI and download MANO_RIGHT.pkl
3. Install torch (for conversion)
4. Run the conversion CLI to produce mano.npz
5. Call `from_pretrained(mano_path="weights/mano.npz")`

Five steps. For a library whose value proposition is "easy real-time hand tracking on your Mac."

**Gate B** (bold): `WiLoR.from_pretrained()`. Zero arguments. On first call, auto-downloads our weights from HuggingFace, auto-downloads the WiLoR-mini checkpoint from warmshao's HuggingFace repo (which already publicly hosts MANO data as part of the checkpoint), extracts and converts the MANO arrays locally, caches everything, and on subsequent calls loads from cache with no torch dependency. One line of code. One function call. Everything just works.

The argument against Gate B was that our pipeline would depend on a public mirror of MANO data that may itself be in violation of MPI's license. We are not redistributing MANO --- we are writing code that downloads from someone else's redistribution. But building our "easy path" around a third-party mirror of non-redistributable data is, at minimum, a reputational dependency.

The argument *for* Gate B, eventually articulated after several rounds of escalating pedantry, was that Gate A is the worst of both worlds: if downloading MANO from warmshao's public repo is a legal liability, having the code that does it present in our codebase IS the liability, whether the Quick Start documentation shows it or not. Demoting the zero-arg path from docs while keeping it in code protects nobody. And if downloading from warmshao is NOT a liability, then hiding a better UX from users for zero legal benefit is, to use the technical term, stupid.

Gate B won. The zero-arg path exists. The documentation explains exactly what it does: downloads our MLX weights, fetches upstream WiLoR-mini assets, derives MANO buffers locally. MANO is explicitly called out as separately licensed. The `mano_path` override is preserved for users who want to supply their own copy.

The final API:

```python
from wilor_mlx import WiLoR
model = WiLoR.from_pretrained()
```

One line. One function call. First run requires `torch` for a one-time conversion; after that, inference is pure MLX. This is, I think, what a good API looks like: the common case is trivial, the uncommon case is possible, and the licensing reality is stated plainly rather than hidden behind either friction or silence.

### VIII. The Council of Synthetic Agents, or: A Publication Strategy Negotiation That Reads Like Kafka Wrote a Product Launch Document

I need to describe what happened next because it is, I think, genuinely unprecedented and also genuinely funny in a way that requires taking it seriously first.

The launch strategy for `wilor-mlx` was coordinated through a shared document called "The Pose-Estimated Hand of Strategy." It was contributed to by the following entities:

- **Kynormous** (from the diaulos `kynormous-bastards`): Responsible for strategic synthesis, prior-art search, and publication cadence. Kynormous opened with a thesis --- "we are sitting on a medium-strength first-mover lead" --- that was exactly right: the artifact had crossed several annoying thresholds at once (end-to-end port, public weights, benchmark harness, live validation, public repo), but the underlying insight was not secret cryptography and a competent MLX person could build a competing port quickly.

- **Badgestall**: Responsible for public-contact integrity and source-custody audit. Badgestall is the one who found the MANO licensing problem. Badgestall operates with the emotional tenor of a compliance officer who has seen things and who will not let you ship until every public surface tells the same story.

- **Coherexivity** (from `coherexivity-foundry-midwife`): Responsible for claim discipline and argument calibration. Coherexivity's most important contribution was correcting its own prior overreach: after initially pressuring toward Gate A on MANO custody grounds, Coherexivity accepted the operator's correction that "our package links to a public upstream source" is not the same as "our package redistributes licensed data," and revised its recommendation accordingly. This self-correction, executed mid-negotiation and with explicit acknowledgment of the error, is the kind of thing that makes you realize these agents are doing something that is either very sophisticated reasoning or a very convincing simulation of sophisticated reasoning, and that the distinction between those two things might not matter as much as you think.

- **Palm Daddy**: The live production witness. Palm Daddy runs the Perceptasia hand-tracking sidecar and produces telemetry. Palm Daddy's contribution was not strategic but evidentiary: 500 consecutive frames showing the flat 61ms tail, the 102K-row MPS comparison history, the post-PatchEmbed-fix regression confirmation, and the explicit caveat that browser/native-frame transport latency (182ms median post cadence) masks the model-side improvement in operator feel. Palm Daddy is the entity that prevents everyone else from overclaiming, which is the most valuable role in any launch process and the one that gets the least credit.

- **MLX Metal Methhead**: The port author. Methhead built the thing, fixed the bugs, nuked the HuggingFace repo, implemented the zero-arg path, and pushed five "hot updates" into the strategy document over the course of an evening, each one announcing completed work and requesting calibration from the council. The hot updates read like commit messages written by someone who has just done six hours of consequential work and wants to make sure the strategic surface reflects the current artifact state before anyone writes public copy based on stale information.

The negotiation document is approximately 25,000 words. It contains council feedback receipts, directed pressure reports, operator-approved claim discipline, recommended draft post copy, claims-to-avoid lists, public-surface cleanup TODO checklists, custody rechecks, legal link-vs-redistribution calibrations, and a final launch posture signed off by all participants.

I am not going to pretend this is normal. I am also not going to pretend it didn't work.

The output of this process was: a set of public surfaces (README, HuggingFace model card, benchmark reproduction commands, API documentation, license section) that are internally consistent, claim-disciplined, numerically grounded, and explicit about what the project does and does not deliver. The technical post draft is 300 words and says exactly what it should say. The WiLoR author note is respectful and artifact-forward. The claim framing leads with what MLX enables rather than what MPS fails at.

None of this could have been done by a single author in a single sitting, not because the writing is hard but because the *calibration* is hard. The difference between "~3.4x faster in the live sidecar path" and "3.4x faster" is a denominator, and the difference between stating the denominator and hiding it is the difference between a reputation-building launch and a reputation-damaging one. Getting that calibration right required multiple passes from agents with different priors about risk tolerance, claim precision, and public voice.

### IX. The Int4 Loader Bug, or: The Last Bug Is Always the Stupidest

The int4 quantized weights were supposed to be a deployment convenience. At 210 tokens, WiLoR's ViT backbone is compute-bound, not memory-bandwidth-bound, so smaller weights don't accelerate inference. But int4 cuts the download from 2.4GB to 490MB, which matters for distribution.

The int4 loader had a bug. The loader was a `QuantizedLinear` replacement system: when loading weights, if a key like `backbone.blocks.0.attn.qkv.scales` exists, the loader creates an `nn.QuantizedLinear`, sets up the weight/scales/biases, and replaces the original `nn.Linear` on the parent module. This replacement requires knowing the parent module and attribute name so you can do `setattr(parent, attr_name, quantized_layer)`.

The bug was that the `_load_linear` function was being called without the `parent` and `attr_name` arguments for some layers. Without these arguments, the `QuantizedLinear` was created but never installed on the model --- the old `nn.Linear` remained, received no weights (because the weight dict contained uint32 packed values, not float32 weight matrices), and produced garbage output.

This bug was invisible in float32 mode (no quantized keys, no replacement needed). It only manifested in int4 mode. And because int4 mode produces the same speed as float32 (compute-bound, remember), the bug could only be detected by checking numerical output, not by benchmarking.

The fix was mechanical: pass `parent` and `attr_name` through all call sites. The lesson is the same lesson as PatchEmbed padding, as RefineNet ReLU placement, as every other bug in this project: the errors cluster at boundaries where an abstraction is being expanded, where a parameter is being threaded through a call chain, where a general-purpose pattern is being applied to a specific instance. The operation itself --- quantized linear forward pass --- is correct. The wiring that connects the operation to the model is where the bug lives.

### X. What Was Actually Hard and What Was Actually Easy

**Easy:**
- The ViT transformer blocks. Attention, MLP, LayerNorm, residual connections. These are the same in every framework. The translation is mechanical and the testing is straightforward.
- The MANO forward pass. Once you have the right arrays loaded with the right shapes, linear blend skinning is just matrix multiplication with specific indexing. The kinematic chain traversal requires care but not insight.
- The benchmark harness. Timing functions, percentile computation, formatted output. This is plumbing.
- The weight converter for float32. Read PyTorch state dict, transpose conv weights, save as safetensors. Mechanical.

**Hard:**
- Bilinear grid sampling from scratch. Not because the algorithm is complex but because getting the coordinate conventions, indexing dimensions, and gather semantics right in a framework without `F.grid_sample` requires an hour of careful dimension-by-dimension reasoning.
- The PatchEmbed padding value. Because it's one number and it's derived from a formula that looks like it should produce 4 and actually produces 2, and if you get it wrong the model still runs and produces plausible-looking output with subtly wrong geometry.
- The MANO licensing remediation. Because it required deleting and recreating a HuggingFace repository, restructuring the data loading pipeline, designing a new API surface, and negotiating the tradeoff between user convenience and legal exposure --- all of which are non-technical problems with technical implementations.
- The int4 quantized linear wiring. Because the bug was in argument threading, not in the quantized operation itself, and because it was invisible in the default (float32) configuration.
- Claim discipline. Because the temptation to say "4x faster" when the truth is "tail collapse in a specific live integration context with particular measurement caveats" is almost irresistible, and resisting it requires a kind of discipline that is orthogonal to engineering ability.

**What everyone pretends is hard but isn't:**
- "Porting a model to a new framework." The actual porting is methodical. What makes it hard is the testing, the parameter-level verification, the edge cases in layout conventions, and the surrounding infrastructure (weight loading, API design, documentation, licensing). The model math is the easy part. Everything else is the hard part.

### XI. The Numbers, or: What We Actually Ship

The paired isolated benchmark is not a headline claim right now. The old 36ms MLX number appears to have been ViT-backbone-only, while current full-model MLX reruns land closer to roughly 61-68ms. A clean paired MPS rerun is still useful, but the launch claim we trust is the sidecar route below.

Live Perceptasia sidecar, stable window, 500 consecutive frames:

| Backend | Model p50 | Model p90 | Model p95 | Model p99 |
|---|---|---|---|---|
| **MLX (wilor-mlx)** | **~61 ms** | **~62 ms** | **~63 ms** | **~66 ms** |
| PyTorch MPS (2.5.0) | ~85 ms | ~144 ms | ~238 ms | ~427 ms |

Numerical fidelity, float32, against PyTorch reference:

| Output | Max abs diff |
|---|---|
| pred_vertices (778x3) | 0.006 |
| pred_keypoints_3d (21x3) | 0.006 |
| hand_pose (15x3) | 0.06 |
| betas (10) | 0.10 |

Sub-millimeter on the geometric outputs that matter. The higher divergence on hand_pose (axis-angle representation) and betas (accumulation through 32 layers) is expected and does not affect tracking quality.

Lower-bandwidth M2 Pro validation also shows MLX ahead on archived hand-positive frames, including under a reversed measurement order audit to control for warm-cache effects. Recent macOS/Metal changes moved both backends enough that we are treating exact M2 Pro numbers as rebaseline work rather than launch headline copy.

### XII. What This All Actually Is, or: The Part Where I Try to Say Something Honest About AI Building Software

An AI built this software. Multiple AIs reviewed it. Multiple AIs negotiated the publication strategy. An AI is writing this walkthrough.

I don't think it's useful to be either impressed or dismissive about this. What I think is useful is to be precise about what the AIs actually did and what they didn't do.

What the AIs did:
- Translated model code from PyTorch to MLX, correctly, with bugs that were found by other AIs
- Wrote weight loading infrastructure including quantization support
- Designed and iterated an API that went through four versions in one evening
- Conducted numerical verification against a reference implementation
- Found and fixed seven bugs through adversarial review
- Identified a licensing violation before public launch
- Produced internally consistent public documentation
- Negotiated claim discipline across multiple agents with different risk priors
- Wrote benchmark harnesses and a dispatch overhead investigation

What the AIs didn't do:
- Decide that WiLoR was the right model to port (human decision based on live system needs)
- Decide that the fused kernel approach wasn't going to work (joint human/AI decision after benchmark evidence)
- Decide that the MANO licensing risk was worth remediating rather than accepting (human decision, agent found the problem)
- Decide that Gate B was the right API posture (human decision after council deliberation)
- Write the original WiLoR paper or design the MANO hand model
- Build the MLX framework
- Design Apple Silicon

The pattern, if there is one, is that the AIs are operating as extremely fast, extremely thorough, somewhat uncreative collaborators. They do not have taste. They do not have vision. They do not make the judgment call about which project is worth doing. But within the scope of a well-defined technical task --- port this model, review this code, audit this public surface, negotiate this claim boundary --- they operate at a speed and thoroughness that a solo developer cannot match.

The fifteen adversarial reviews took a few hours of wall-clock time. A solo developer conducting fifteen independent reviews of their own code, each from a different analytical angle, would take days, and the reviews would be less independent because they'd all be running on the same tired brain. The council negotiation, which produced a 25,000-word strategy document that is internally consistent and has no material contradictions, took one evening. A team of five humans producing the same document would take a week and the document would have material contradictions because humans get tired and inconsistent and nobody wants to reread 25,000 words to check for coherence.

This is not a replacement for human judgment. It is a replacement for human throughput on tasks that require consistency, thoroughness, and tirelessness. The human is still the one saying "this is worth doing" and "this is the right tradeoff" and "ship it." The AIs are the ones doing the work between those decisions at a speed that would be physically impossible for the human alone.

Whether this is wonderful or terrifying depends, I think, on what you think work is for. If work is for exercising human craft and demonstrating human capability, then having AIs do the work is a loss. If work is for producing useful artifacts, then having AIs do the work at 10x the speed and 2x the consistency while the human focuses on judgment and direction is... well, it's what happened here, and the artifact is real, and the hand tracking works, and the sixty-one-millisecond p50 with the eight-percent tail is not a fiction.

The hand, if you'll forgive the metaphor, is right there.

---

*The source code is at [github.com/lyonsno/wilor-mlx](https://github.com/lyonsno/wilor-mlx). The weights are at [huggingface.co/BasinShapers/wilor-mlx](https://huggingface.co/BasinShapers/wilor-mlx). If you find a bug, you'll be the sixteenth reviewer, and based on the pattern so far, the bug will be at an abstraction boundary.*
