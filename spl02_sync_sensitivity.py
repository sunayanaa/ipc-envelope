# ==============================================================================
# Program Name: spl02_sync_sensitivity.py
# Version: 1.0
# Description: SPL Letter reproducibility script - generates Table II and
#              Fig. 2 (Synchronisation Sensitivity). Measures IPC watermark
#              BER under a leading-sample drift sweep
#              {0, 10, 50, 100, 500, 1000} samples (0-62.5 ms at 16 kHz),
#              with no additional degradation, on N=200 MusicGen clips
#              (gen_musicgen_p000_v0.wav ... gen_musicgen_p199_v0.wav).
#              Generates fig_spl02_drift_sensitivity.png (symlog x-axis,
#              no title, mean +/- 1 std, STFT hop boundary reference line).
#
#              The leading-sample drift is applied to the watermarked signal
#              ONLY (the reference phase Phi_ref and f0 track are extracted
#              from the unshifted original, as a detector would in practice);
#              the shifted signal is zero-padded back to its original length
#              so the STFT frame count matches ref_phase, isolating the
#              effect of misalignment from any change in frame count.
#
# Change Log: 1.0 - Initial version.
# GPU Required: NO
# ==============================================================================

!pip install -q numpy librosa soundfile pandas matplotlib

import os
import sys
import json
import shutil
import numpy as np
import librosa
import soundfile as sf
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Configuration ---
PROJECT_DIR = "/content/drive/MyDrive/ipc-envelope/"  # Persistent storage
LOCAL_WORKSPACE = "/content/spl02_workspace"
os.makedirs(LOCAL_WORKSPACE, exist_ok=True)

CHECKPOINT_FILE = "spl02_checkpoint.json"
RESULTS_FILE = "spl02_table2_results.json"
TABLE_FILE = "spl02_table2.tex"
FIGURE_FILE = "fig_spl02_drift_sensitivity.png"

N_CLIPS = 200  # gen_musicgen_p000_v0.wav ... gen_musicgen_p199_v0.wav

# --- Audio & Watermark Params (locked, H*=8, B*=32) ---
SR = 16000
N_FFT = 2048
HOP_LENGTH = 512
H_STARS = 8
B_STARS = 32

DRIFT_SAMPLES = [0, 10, 50, 100, 500, 1000]
STFT_HOP_MS = 1000.0 * HOP_LENGTH / SR  # = 32.0 ms

# --- 2. Google Drive Helper Functions ---
def ensure_project_dir():
    """Create project directory in Google Drive if it doesn't exist."""
    os.makedirs(PROJECT_DIR, exist_ok=True)

def save_to_drive(local_filepath, remote_filename):
    """Copy a local file to Google Drive project folder."""
    ensure_project_dir()
    dest_path = os.path.join(PROJECT_DIR, remote_filename)
    try:
        shutil.copy2(local_filepath, dest_path)
        print(f"  [DRIVE OK] {local_filepath}  →  {dest_path}")
    except Exception as e:
        print(f"  [DRIVE FAIL] {local_filepath}: {e}")

def load_from_drive(remote_filename, local_filepath):
    """Copy a file from Google Drive project folder to local path."""
    ensure_project_dir()
    src_path = os.path.join(PROJECT_DIR, remote_filename)
    if os.path.exists(src_path):
        try:
            shutil.copy2(src_path, local_filepath)
            print(f"  [DRIVE OK] {src_path}  →  {local_filepath}")
            return True
        except Exception as e:
            print(f"  [DRIVE FAIL] copy from {src_path}: {e}")
            return False
    else:
        print(f"  [DRIVE MISSING] {src_path} not found")
        return False

def list_drive_files():
    """List files in the Google Drive project directory."""
    ensure_project_dir()
    try:
        return [f for f in os.listdir(PROJECT_DIR) if os.path.isfile(os.path.join(PROJECT_DIR, f))]
    except Exception as e:
        print(f"  [DRIVE] Could not list files: {e}")
        return []

def load_checkpoint():
    """Load checkpoint from Google Drive project folder."""
    local_cp = os.path.join(LOCAL_WORKSPACE, "temp_checkpoint.json")
    if load_from_drive(CHECKPOINT_FILE, local_cp):
        with open(local_cp, "r") as f:
            return json.load(f)
    return {"processed": [], "results": []}

def save_checkpoint(state):
    """Save checkpoint to Google Drive project folder."""
    local_cp = os.path.join(LOCAL_WORKSPACE, "temp_checkpoint.json")
    with open(local_cp, "w") as f:
        json.dump(state, f)
    save_to_drive(local_cp, CHECKPOINT_FILE)

# --- 3. IPC Core (verbatim from 02_exp1_imperceptibility.py / 03_exp2_exp4_detection.py) ---
def generate_watermark(identity_str, bits=B_STARS):
    """Generates a stable pseudo-random binary sequence from a generator ID."""
    seed = sum(ord(c) for c in identity_str)
    np.random.seed(seed)
    return np.random.choice([-1, 1], size=bits)

def get_harmonic_bins(f0_track, sr, n_fft, H):
    bins = []
    for f0 in f0_track:
        if np.isnan(f0) or f0 == 0:
            bins.append(None)
        else:
            h_bins = [int(np.floor((h * f0) / (sr / n_fft))) for h in range(1, H + 1)]
            bins.append(h_bins)
    return bins

def embed_watermark(y, sr, H, B, watermark_bits):
    D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    S_mag, S_phase = librosa.magphase(D)

    f0, _, _ = librosa.pyin(y, fmin=65, fmax=2000, sr=sr)
    harmonic_bins = get_harmonic_bins(f0, sr, N_FFT, H)

    modified_phase = np.angle(S_phase)
    delta_max = np.pi / 4

    for m in range(D.shape[1]):
        if harmonic_bins[m] is None:
            continue
        for idx, k_h in enumerate(harmonic_bins[m]):
            if k_h < D.shape[0]:
                bit_idx = (m + idx) % B
                w_b = watermark_bits[bit_idx]
                modified_phase[k_h, m] += delta_max * w_b

    D_watermarked = S_mag * np.exp(1.j * modified_phase)
    y_wm = librosa.istft(D_watermarked, hop_length=HOP_LENGTH, length=len(y))
    return y_wm

def extract_reference_phase(y_orig, sr, H):
    D = librosa.stft(y_orig, n_fft=N_FFT, hop_length=HOP_LENGTH)
    _, S_phase = librosa.magphase(D)
    f0, _, _ = librosa.pyin(y_orig, fmin=65, fmax=2000, sr=sr)

    harmonic_bins = get_harmonic_bins(f0, sr, N_FFT, H)
    ref_phase = np.zeros((H, D.shape[1]))

    for m in range(D.shape[1]):
        if harmonic_bins[m] is not None:
            for idx, k_h in enumerate(harmonic_bins[m]):
                if k_h < D.shape[0]:
                    ref_phase[idx, m] = np.angle(S_phase[k_h, m])
    return f0, ref_phase

def detect_watermark(y_deg, sr, H, B, f0, ref_phase):
    D = librosa.stft(y_deg, n_fft=N_FFT, hop_length=HOP_LENGTH)
    _, S_phase = librosa.magphase(D)
    deg_phase = np.angle(S_phase)

    harmonic_bins = get_harmonic_bins(f0, sr, N_FFT, H)
    bit_votes = {b: [] for b in range(B)}

    for m in range(D.shape[1]):
        if harmonic_bins[m] is None:
            continue
        for idx, k_h in enumerate(harmonic_bins[m]):
            if k_h < D.shape[0]:
                bit_idx = (m + idx) % B
                phase_diff = deg_phase[k_h, m] - ref_phase[idx, m]
                phase_diff = (phase_diff + np.pi) % (2 * np.pi) - np.pi
                decoded_bit = 1 if phase_diff > 0 else -1
                bit_votes[bit_idx].append(decoded_bit)

    recovered_bits = []
    for b in range(B):
        if len(bit_votes[b]) > 0:
            vote = 1 if sum(bit_votes[b]) >= 0 else -1
        else:
            vote = np.random.choice([-1, 1])
        recovered_bits.append(vote)

    return np.array(recovered_bits)

def compute_ber(ref_bits, rec_bits, n_bits=B_STARS):
    assert len(ref_bits) == len(rec_bits) == n_bits
    errors = np.sum(np.asarray(ref_bits) != np.asarray(rec_bits))
    return float(errors / n_bits)

# --- 4. Drift Injection ---
def apply_leading_drift(y, drift_samples):
    """
    Drops the leading `drift_samples` samples and zero-pads the tail back to
    the original length, so the STFT frame count matches ref_phase (which was
    computed on the unshifted, full-length original). drift_samples=0 returns
    y unchanged.
    """
    if drift_samples == 0:
        return y
    y_shifted = y[drift_samples:]
    pad = np.zeros(drift_samples, dtype=y.dtype)
    return np.concatenate([y_shifted, pad])

# --- 5. Execution Pipeline ---
print(f"\n--- Starting SPL Table II Evaluation: synchronisation sensitivity, "
      f"N={N_CLIPS} MusicGen clips, drift in samples = {DRIFT_SAMPLES} ---")

# Mount Google Drive first
from google.colab import drive
drive.mount('/content/drive')

state = load_checkpoint()
processed = set(state["processed"])

target_payload = generate_watermark("musicgen")

# Get list of available files on Drive once
drive_files = list_drive_files()

for i in range(N_CLIPS):
    filename = f"gen_musicgen_p{i:03d}_v0.wav"
    if filename in processed:
        continue

    local_path = os.path.join(LOCAL_WORKSPACE, filename)

    if filename not in drive_files:
        print(f"\n[FATAL] Required clip '{filename}' was not found on Drive "
              f"under '{PROJECT_DIR}'.")
        print("Table II / Fig. 2 are defined over a fixed N=200 MusicGen "
              "clips; a missing clip means the reported statistics would be "
              "computed over an incomplete sample.")
        print("Run spl00_data_generation.py first to generate the full "
              "required corpus for this paper, then re-run this script.")
        sys.exit(1)

    if not load_from_drive(filename, local_path):
        print(f"\n[FATAL] Could not download clip '{filename}' from Drive.")
        sys.exit(1)

    print(f"Evaluating: {filename}")
    y_orig, _ = librosa.load(local_path, sr=SR)
    os.remove(local_path)

    f0_track, ref_phase = extract_reference_phase(y_orig, SR, H_STARS)
    y_wm = embed_watermark(y_orig, SR, H_STARS, B_STARS, target_payload)

    for drift in DRIFT_SAMPLES:
        y_drifted = apply_leading_drift(y_wm, drift)
        recovered = detect_watermark(y_drifted, SR, H_STARS, B_STARS, f0_track, ref_phase)
        ber = compute_ber(target_payload, recovered, B_STARS)

        state["results"].append({
            "filename": filename,
            "drift_samples": drift,
            "drift_ms": 1000.0 * drift / SR,
            "ber": ber,
        })

    state["processed"].append(filename)
    save_checkpoint(state)

# --- 6. Aggregation: Table II ---
print("\n--- Aggregating Table II ---")
df = pd.DataFrame(state["results"])

local_results = os.path.join(LOCAL_WORKSPACE, RESULTS_FILE)
df.to_json(local_results, orient="records", indent=4)
save_to_drive(local_results, RESULTS_FILE)

agg = (df.groupby("drift_samples")
         .agg(drift_ms=("drift_ms", "first"),
              mean_ber=("ber", lambda x: 100.0 * x.mean()),
              std_ber=("ber", lambda x: 100.0 * x.std()))
         .reset_index()
         .sort_values("drift_samples"))

print(agg.to_string(index=False))

table2_tex = (
    "\\begin{table}[t]\n"
    "\\centering\n"
    "\\caption{BER Under Leading-Sample Drift ($N=200$)}\n"
    "\\label{tab:drift}\n"
    "\\begin{tabular}{ccc}\n"
    "\\toprule\n"
    "Drift (ms) & Mean BER (\\%) & Std (\\%) \\\\\n"
    "\\midrule\n"
)
for _, row in agg.iterrows():
    table2_tex += f"{row['drift_ms']:.3f} & {row['mean_ber']:.2f} & {row['std_ber']:.2f} \\\\\n"
table2_tex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"

print(table2_tex)
local_table = os.path.join(LOCAL_WORKSPACE, TABLE_FILE)
with open(local_table, "w") as f:
    f.write(table2_tex)
save_to_drive(local_table, TABLE_FILE)

# --- 7. Fig. 2: Synchronisation Sensitivity (symlog x-axis, no title) ---
print("\n--- Generating Fig. 2 ---")
drift_ms = agg["drift_ms"].values
mean_ber = agg["mean_ber"].values
std_ber = agg["std_ber"].values

fig, ax = plt.subplots(figsize=(3.45, 2.6), dpi=300)

ax.errorbar(
    drift_ms, mean_ber, yerr=std_ber,
    fmt='o-', color='#E69F00', ecolor='#E69F00',
    elinewidth=1.0, capsize=3, markersize=4,
    linewidth=1.5, label=r'Mean BER $\pm$ 1 std'
)
ax.axvline(
    STFT_HOP_MS, color='#56B4E9', linestyle='--', linewidth=1.2,
    label=f'STFT hop boundary ({STFT_HOP_MS:.0f} ms)'
)

ax.set_xscale('symlog', linthresh=1, linscale=1)
ax.set_xlim(-0.5, max(drift_ms) * 1.3)
ax.set_ylim(0, max(mean_ber + std_ber) * 1.1)

ax.set_xlabel('Leading Sample Drift (ms)', fontsize=9)
ax.set_ylabel('Mean BER (%)', fontsize=9)
ax.tick_params(axis='both', labelsize=8)
ax.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.6)
ax.legend(fontsize=7, loc='lower right', frameon=True)

fig.tight_layout(pad=0.3)
local_fig = os.path.join(LOCAL_WORKSPACE, FIGURE_FILE)
local_fig_pdf = local_fig.replace(".png", ".pdf")
fig.savefig(local_fig, dpi=300, bbox_inches='tight')
fig.savefig(local_fig_pdf, bbox_inches='tight')
plt.close(fig)

save_to_drive(local_fig, FIGURE_FILE)
save_to_drive(local_fig_pdf, FIGURE_FILE.replace(".png", ".pdf"))

# --- Sync to ensure all writes are flushed ---
print("\n[SYNC] Flushing file system buffers...")
os.sync()
print("[SYNC] Complete.")

print(f"\n[SUCCESS] Table II and Fig. 2 complete. Results: {RESULTS_FILE}, "
      f"LaTeX: {TABLE_FILE}, Figure: {FIGURE_FILE} (saved to Drive).")