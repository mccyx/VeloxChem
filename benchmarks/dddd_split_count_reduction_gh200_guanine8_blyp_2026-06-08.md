# DDDD Split-Count Reduction Benchmark on GH200 (guanine-8, BLYP/def2-SVP)

## Goal
- Repeat the [`split30 -> split26`](dddd_split_count_reduction_gh200_guanine8_2026-04-17.md) DDDD split-count comparison, but for a DFT functional (`BLYP/def2-SVP`) instead of HF/def2-SVP.
- Check whether the ~1.12x host-timing speedup and the accuracy parity reported for HF also hold once the SCF goes through the BLYP XC path (different density matrices each iteration, more SCF iterations to convergence).

## Variants
- `split30 mixed`: original mixed-precision DDDD path in [`FockDriverGPU.cu`](/cfs/klemming/home/y/yuch4126/repo/VeloxChem.mixed-precision-2/src/gpu/FockDriverGPU.cu), using the original `3` and `4` kernels separately (host-side result buffer `J2_2kernels`).
- `split26 v2 mixed`: new mixed-precision DDDD path using the generated `v2` kernels included from [`dddd_split26_v2_fp32.inc`](/cfs/klemming/home/y/yuch4126/repo/VeloxChem.mixed-precision-2/src/gpu/generated/dddd_split26_v2_fp32.inc) and [`dddd_split26_v2_fp64.inc`](/cfs/klemming/home/y/yuch4126/repo/VeloxChem.mixed-precision-2/src/gpu/generated/dddd_split26_v2_fp64.inc) (host-side result buffer `Jv2`).

## Input
- [`guanine-8-blyp.inp`](/cfs/klemming/home/y/yuch4126/repo/VeloxChem.mixed-precision-2/guanine-8-blyp.inp), copied from `VeloxChem.mp-coulomb/guanine-8-blyp.inp` and adjusted for this build:
  - `xcfun: blyp`, `basis: def2-svp` (the original `guanine-8.inp` used here had no `xcfun`, i.e. it ran HF, not BLYP)
  - `acc_type: diis`, `timing: yes`, `timing_gpu: yes`
  - dropped `mixed_prec_thresh: 1e-6` from the copied input: this keyword is defined in `VeloxChem.mp-coulomb`'s `scfdriver.py` but **not** in this build's `scfdriver.py`/`inputparser.py` (unknown keys are silently ignored, see `inputparser.py:401-402`), so it would have had no effect here.
- Molecule: `guanine-8.xyz` (same geometry file as the HF benchmark; verified identical via `diff` against `VeloxChem.mp-coulomb/guanine-8.xyz`).

## Run
- GH200 node `nid002892` (4x GH200 120GB), via `srun --jobid=<salloc allocation>`.
- Run 1 (cold start): `VLX_ABLATION_LOG=ablation_results_gh200_guanine8_blyp_split2630_20260608.log vlx guanine-8-blyp.inp`
  - SCF converged in **21 iterations**, total execution time **126.81 sec**, Total Energy = `-4650.2829788196 a.u.`
  - Wrote checkpoint `guanine-8-blyp.scf.h5`.
- Run 2 (checkpoint restart, same input/log dir): re-ran `vlx guanine-8-blyp.inp` with the checkpoint from run 1 present.
  - SCF converged in **1 iteration**, total execution time **10.20 sec**, same Total Energy `-4650.2829788196 a.u.` — confirms run 1 had already converged and the checkpoint restart starts from essentially the same density.
  - This run isolates a *single* SCF iteration's worth of DDDD timing samples (4 per variant — one per GPU), giving a clean, minimal-noise confirmation of the speedup and of the per-GPU pattern described below.

## Data sources
- [`ablation_results_gh200_guanine8_blyp_split2630_20260608.log`](/cfs/klemming/home/y/yuch4126/repo/VeloxChem.mixed-precision-2/ablation_results_gh200_guanine8_blyp_split2630_20260608.log) — run 1 (21 iterations, 84 samples per variant).
- [`ablation_results_gh200_guanine8_blyp_split2630_rerun_20260608.log`](/cfs/klemming/home/y/yuch4126/repo/VeloxChem.mixed-precision-2/ablation_results_gh200_guanine8_blyp_split2630_rerun_20260608.log) — run 2 (checkpoint restart, 1 iteration, 4 samples per variant).

## Host Timing Summary
As in the HF benchmark, host timing (measured with `std::chrono::steady_clock` wrapped tightly around each path's kernel launches plus a `gpuStreamSynchronize`, written via `append_kernel_timing`) is the primary comparison metric here — see [`FockDriverGPU.cu:11933-12256`](/cfs/klemming/home/y/yuch4126/repo/VeloxChem.mixed-precision-2/src/gpu/FockDriverGPU.cu). Each measured interval contains *only* that path's kernels (60 launches for `split30`, 2 launch-groups for `split26 v2`), with no extra/duplicate launches inside the timed window — the duplicate-launch issue described below only affects raw `nsys` per-kernel-name sums, not these host-timing numbers.

### Run 1 — full SCF (21 iterations, 84 samples per variant)
Because BLYP runs through 21 SCF iterations (vs. the small fixed sample count used in the HF run), this run produced many more timing samples — all of them are used here (not just "the latest 4") for a more robust average:

| Variant | Samples (n) | Avg (ms) | Min / Max (ms) | Speedup vs `split30` |
|---|---:|---:|---|---:|
| `split30 mixed` | 84 | 46.581 | 42.369 / 58.256 | 1.0000x |
| `split26 v2 mixed` | 84 | 41.391 | 37.694 / 50.549 | 1.1254x |

`split26 v2 mixed` is about **11.14% faster** than `split30 mixed` here — essentially the same speedup as the HF/def2-SVP result (**1.1215x**, ~10.83% faster, see the [original benchmark](dddd_split_count_reduction_gh200_guanine8_2026-04-17.md)).

Both variants show a recurring "low cluster + occasional high outlier" pattern (`split30`: ~42-44 ms vs. occasional ~55-58 ms; `split26 v2`: ~37-39 ms vs. occasional ~48-50 ms). This is **not** run-to-run noise — `84 = 21 SCF iterations x 4 GPUs` (the run used all 4 GPUs of the GH200 node, confirmed by the "GPU Timer: rank 0 gpu 0/1/2/3" lines in the run log and the per-GPU `mat_Fock_omp[gpu_id]` loop structure in `FockDriverGPU.cu` — the DDDD ablation timing block fires once per `mat_Fock_omp[gpu_id]` call, i.e. once per GPU per SCF iteration). The hypothesis from this run alone was that one specific GPU rank is consistently ~12 ms slower; run 2 below isolates a single iteration and confirms this directly.

(Aside: the original HF benchmark only compared its "latest 4" samples — cherry-picked from a long-lived shared `ablation_results.log` accumulating many unrelated experiments, not because that run itself produced only 4 samples.)

### Run 2 — checkpoint restart (1 iteration, 4 samples per variant)
Restarting from the converged checkpoint of run 1 isolates exactly one SCF iteration's worth of DDDD calls — one sample per GPU, with no cross-iteration variation to obscure the pattern:

| Variant | Samples (ms), in GPU order | Avg (ms) | Speedup vs `split30` |
|---|---|---:|---:|
| `split30 mixed` | 43.972, 43.906, 44.024, **56.321** | 47.056 | 1.0000x |
| `split26 v2 mixed` | 39.024, 39.047, 39.172, **51.481** | 42.181 | 1.1156x |

This single-iteration view makes the per-GPU effect unambiguous: in **both** variants, the 4th sample (one specific GPU rank) is ~12 ms slower than the other three, which are tightly clustered (sub-0.1 ms spread). This directly confirms — rather than merely hypothesizes — that the "low cluster + high outlier" shape in the 84-sample data is a deterministic per-GPU effect (likely an uneven DDDD shell-pair distribution across GPUs for this molecule/functional combination), not run-to-run noise or an SCF-iteration-dependent effect. The resulting speedup (**1.1156x**, ~10.4% faster) is consistent with both the 84-sample run-1 average and the original HF result — the same `split30 -> split26 v2` improvement holds whether measured over many iterations or isolated to a single one.

## Accuracy Summary
(Last sample from the run; representative of the run as a whole — `rel error` stays in the `~1e-8` to `~1e-10` range across all samples.)

### `split30 mixed` (`J2_2kernels`) vs reference
- max `|ΔJ|` = `8.261285e-11`
- rms `|ΔJ|` = `7.417770e-12`
- rel error = `9.088520e-09`

### `split26 v2 mixed` (`Jv2`) vs reference
- max `|ΔJ|` = `8.235206e-11`
- rms `|ΔJ|` = `7.418819e-12`
- rel error = `9.059830e-09`

### `split26 v2 mixed` vs `split30 mixed`
- max `|ΔJ|` = `2.208675e-12`
- rms `|ΔJ|` = `2.234123e-13`
- rel error = `2.429838e-10`

### `split26 v2 - split30` delta contribution vs zero
- max `|ΔJ|` = `2.208675e-12`
- rms `|ΔJ|` = `2.234123e-13`

Interpretation:
- Both `split26 v2` and `split30` track the full-double reference at the same `~1e-8` relative-error level under BLYP, exactly as they did under HF.
- The direct `split26 v2` vs `split30` difference (`~1e-10` relative error) is ~2 orders of magnitude smaller than either path's deviation from the reference — i.e. switching `split30 -> split26 v2` does not introduce any meaningfully new error under a DFT functional either.

## Conclusion
- The `split30 -> split26 v2` speedup and accuracy-parity results from the HF/def2-SVP benchmark **carry over to BLYP/def2-SVP**: ~1.13x host-timing speedup, with `split26 v2` staying within ~`2e-12` max `|ΔJ|` of `split30` (both ~`8e-11` max `|ΔJ|` vs. the full-double reference).
- This further supports using `split26` as the base DDDD implementation independent of whether the surrounding SCF uses HF or a GGA functional like BLYP.
