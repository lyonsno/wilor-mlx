# Probolē: Public Surface Contract

Target: `README.md`, `pyproject.toml`, `benchmarks/bench_wilor.py`, `weights/README.md` (HF model card)
Scope: Review that all public-facing claims, code examples, and API documentation are internally consistent and match the actual code.

## Review targets

- **README Quick Start**: Does `WiLoR.from_pretrained()` with zero args actually work? Does the example code run as written?
- **Performance claims**: Are the two-tier numbers (1.4x isolated, 3.4x live) stated with proper context? Is there any compressed/unqualified speed claim?
- **Input format**: README says RGB uint8 NHWC. Does `model.py __call__` actually expect RGB? (Check for any channel flip.)
- **Output format table**: Do all 6 output keys exist with the documented shapes?
- **Install instructions**: Does `pip install -e .` + `pip install torch` actually get a working install? Are the dependencies in pyproject.toml correct and sufficient?
- **Benchmark reproduction**: Does the documented command `python benchmarks/bench_wilor.py --backend mlx --weights ... --mano-npz ...` actually run? Do the argparse flags match the docstring?
- **License section**: Does it accurately describe what is and isn't distributed? Is the MANO upstream-fetch description precise?
- **HF model card**: Does it tell the same story as the GitHub README? Same Quick Start, same license language, same performance numbers?
- **pyproject.toml**: Does the description match the artifact-first framing? Are project.urls correct?
