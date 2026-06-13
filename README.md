# Phase-Coherent Audio Watermarking Has a Near-Zero Synchronisation Tolerance

Reproducibility pipeline for the IEEE Signal Processing Letters submission.
Each script is self-contained, checkpoints its progress, and writes its
outputs to storage folder (`ipc-envelope/`) so the pipeline can be
resumed or re-run incrementally.

## Dataset

All audio is AI-generated; no real recordings are used.

- **Source prompts**: 200 text prompts sampled from MusicCaps via
  `df.sample(n=200, random_state=42)`.
- **Generators and corpus** (431 ten-second clips at 16 kHz, produced by
  `spl00_data_generation.py`):
  - **MusicGen-medium**: 265 clips, all 200 prompts at variation 0 (the
    `gen_musicgen_p000_v0.wav` ... `p199_v0.wav` set used as the
    N=200/N=50/N=30 MusicGen subsets throughout spl02-spl07), plus a
    stratified subset of prompts at variations 1-4.
  - **AudioLDM-2**: 83 clips (72 unique prompts).
  - **Stable Audio Open**: 83 clips (68 unique prompts; gated checkpoint,
    requires HuggingFace authentication).
- **249-clip stratified sample** (used for Table I, and for the AudioLDM-2 /
  Stable Audio Open subsets in the cross-generator check): 83 clips per
  generator, selected via
  `df_meta.groupby("generator").sample(n=83, random_state=42)` on the sorted
  clip-filename list. This selection is recomputed from first principles in
  both `spl01` and `spl06`, not stored as a separate manifest.
- **Storage**: all 431 clips are written to `ipc-envelope/`, named
  `gen_<generator>_p<prompt_idx>_v<variation>.wav`.

## Pipeline Overview

| Script | Produces | Paper reference | GPU |
|---|---|---|---|
| `spl00_data_generation.py` | Audio corpus: 431 clips across MusicGen-medium, AudioLDM-2, Stable Audio Open | Source corpus for all other scripts | **Yes** (T4 min, A100 recommended) |
| `spl01_baseline_comparison.py` | Imperceptibility & clean-channel detection: IPC vs. Spread-Spectrum vs. WavMark (249-clip stratified sample) | Table I | Yes (WavMark only; PEAQ and IPC run on CPU) |
| `spl02_sync_sensitivity.py` | IPC BER under leading-sample drift sweep (N=200 MusicGen clips) | Table II, Fig. 2 | No |
| `spl03_audioseal_drift.py` | AudioSeal 16-bit message BER under the same drift sweep (N=200) | Table III (superseded for publication by spl04's combined table) | Recommended |
| `spl04_audioseal_presence.py` | Companion to spl03: adds AudioSeal presence-detection score alongside message BER | Combined AudioSeal table (message BER + presence rate) | Recommended |
| `spl05_delta_sweep.py` | Two-panel figure: BER-vs-drift across $\Delta_{\max}\in\{\pi/8,\pi/4,\pi/2\}$ (Part A, N=200) and mean PEAQ ODG vs. $\Delta_{\max}$ (Part B, N=50 subset) | Fig. 3 | No (Part B uses the peaqb-fast CPU binary, same as spl01) |
| `spl06_cross_generator_cliff.py` | Drift-collapse check on AudioLDM-2 / Stable Audio Open (N=30 per generator) at drift = {0, 0.625, 6.25} ms | Cross-generator generalisation table | No |
| `spl07_robustness_battery.py` | IPC BER under MP3 (128/64 kbps), AWGN (30/20 dB SNR), and 16↔44.1 kHz resample round-trip (N=50) | Robustness battery table | No |

## Run Order

1. **`spl00_data_generation.py`** must run first. It samples 200 prompts from
   MusicCaps (`random_state=42`) and generates the 431-clip corpus (265
   MusicGen-medium, 83 AudioLDM-2, 83 Stable Audio Open), saved to
   `ipc-envelope/`. Requires an interactive HuggingFace token (login runs once
   up front, before any model download, both for rate-limit relief and to
   access the gated Stable Audio Open checkpoint).

2. **`spl01`–`spl07`** each depend only on the corpus produced by spl00 and
   can be run in any order. Each reconstructs whatever clip subset it needs
   deterministically:
   - `spl01` and `spl06` independently reconstruct the same 249-clip
     stratified sample via
     `df_meta.groupby("generator").sample(n=83, random_state=42)` on the
     sorted filename list, so no intermediate sample manifest is needed.
   - `spl02`, `spl03`, `spl04`, `spl05` (Part A), and `spl07` operate on the
     same N=200 (or N=50/N=30 subsets thereof) MusicGen clips
     (`gen_musicgen_p000_v0.wav` ... `gen_musicgen_p199_v0.wav`).

## Fixed Parameters

Unless explicitly swept, all scripts use:
- $\Delta_{\max} = \pi/4$ (perturbation budget)
- $H^* = 8$ (harmonics)
- $B^* = 32$ (IPC payload bits) / 16-bit payload for AudioSeal and WavMark
- Sample rate: 16 kHz
- Leading-sample drift sweep: {0, 10, 50, 100, 500, 1000} samples
  (0–62.5 ms at 16 kHz)

`spl05_delta_sweep.py` is the exception, sweeping
$\Delta_{\max}\in\{\pi/8,\pi/4,\pi/2\}$; its $\Delta_{\max}=\pi/4$ curve
should reproduce Table II as a consistency check.

## Notes and Caveats

- **`spl03`'s drift definition**: `apply_leading_drift()` (prepend $d$ zero
  samples, truncate to original length $L$) is this script's own definition
  and should be verified against the drift operation used in `spl02` before
  treating the AudioSeal/IPC comparison as apples-to-apples. If the
  conventions differ, only `apply_leading_drift()` needs to change.
- **`spl03` vs. `spl04`**: `spl04` is the version used for the manuscript's
  AudioSeal table (it reports both message BER and presence-detection rate
  in one table). `spl03` is kept for provenance but its BER-only table is not
  used directly in the final manuscript.
- **`spl05` Part B and `spl07`** both use the same N=50 MusicGen subset
  (`p000`–`p049`, `v0`) for PEAQ-based and robustness measurements
  respectively.
- **`spl06`**'s AudioLDM-2 / Stable Audio Open clips are the first 30 (by
  sorted filename) of each generator's 83-clip subset from the same
  stratified sample used in `spl01`, so all referenced clips already exist
  in `ipc-envelope/` from the spl00 corpus, no additional generation step
  is required.
- **`spl07`** length-matches every perturbed signal back to the original
  length before detection, since MP3 and resample round-trips can shift
  length by a few samples and `detect_watermark` indexes `ref_phase` /
  `harmonic_bins` by STFT frame, assuming the original frame count.

## Environment

- `spl00` installs: `torch`, `torchvision`, `torchaudio`, `torchsde`,
  `transformers`, `diffusers`, `accelerate`, `scipy`, `soundfile`, `pandas`,
  `openpyxl`.
- `spl02` installs: `numpy`, `librosa`, `soundfile`, `pandas`, `matplotlib`.
- `spl01` and `spl05` (Part B) compile the `peaqb-fast` PEAQ binary on first
  run.
- `spl07` requires the `lame` CLI (for MP3 encode/decode); installed
  automatically if not present.
  