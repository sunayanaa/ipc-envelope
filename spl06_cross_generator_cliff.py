# ==============================================================================
# Program Name: spl06_cross_generator_cliff.py
# Version: 1.0
# Description: Checks whether the leading-sample-drift BER collapse seen for
#              MusicGen (Table II, N=200) also occurs for AudioLDM-2 and
#              Stable Audio Open, on a smaller N=30-per-generator subset, at
#              three drift levels: 0, 10, and 100 samples (0, 0.625, 6.25 ms),
#              i.e. baseline, cliff-onset, and plateau.
#
#              The AudioLDM-2 / Stable Audio Open clips are the first 30 (by
#              sorted filename) of each generator's 83-clip stratified subset
#              from the same df_meta.groupby("generator").sample(n=83,
#              random_state=42) selection used to build Table I's 249-clip
#              sample, so all clips referenced here already exist on Drive. The
#              MusicGen subset is the first 30 of Table II's 200 clips
#              (p000-p029, v0). IPC core (embed/detect, Delta_max=pi/4,
#              H*=8, B*=32) and the drift definition match Table II.
#
#              This is a separate, smaller-N (30 vs 200) generalisation check;
#              it is not expected to reproduce Table II's exact MusicGen
#              numbers.
#
# Change Log: 1.0 - Initial version.
# GPU Required: NO.
# ==============================================================================

import os
import sys
import json
import shutil
import numpy as np
import pandas as pd
import librosa

# --- 1. Configuration ---
PROJECT_DIR = "/content/drive/MyDrive/ipc-envelope/"  # Persistent storage
LOCAL_WORKSPACE = "/content/spl06_workspace"
os.makedirs(LOCAL_WORKSPACE, exist_ok=True)

SR = 16000
N_FFT = 2048
HOP_LENGTH = 512
H_STARS = 8
B_STARS = 32
DELTA_MAX = np.pi / 4

DRIFT_SAMPLES = [0, 10, 100]
N_SUBSET = 30

GENERATORS = ["musicgen", "audioldm2", "stableaudio"]
TARGET_PROMPTS = 200
VARIATIONS_PER_PROMPT = 5

CHECKPOINT_FILE = "spl06_checkpoint.json"
RESULTS_FILE = "spl06_results.json"
TABLE_FILE = "spl06_table.tex"

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
    local = os.path.join(LOCAL_WORKSPACE, CHECKPOINT_FILE)
    if load_from_drive(CHECKPOINT_FILE, local):
        with open(local) as f:
            return json.load(f)
    return {"processed": [], "results": []}

def save_checkpoint(state):
    """Save checkpoint to Google Drive project folder."""
    local = os.path.join(LOCAL_WORKSPACE, CHECKPOINT_FILE)
    with open(local, "w") as f:
        json.dump(state, f)
    save_to_drive(local, CHECKPOINT_FILE)

# --- 3. IPC Core (Delta_max-parameterised, defaulting to pi/4 here) ---
def generate_watermark(identity_str, bits=B_STARS):
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
    return f0, harmonic_bins, ref_phase

def embed_watermark(y, harmonic_bins, H, B, watermark_bits, delta_max=DELTA_MAX):
    D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    S_mag, S_phase = librosa.magphase(D)
    modified_phase = np.angle(S_phase)

    for m in range(D.shape[1]):
        if harmonic_bins[m] is None:
            continue
        for idx, k_h in enumerate(harmonic_bins[m]):
            if k_h < D.shape[0]:
                bit_idx = (m + idx) % B
                w_b = watermark_bits[bit_idx]
                modified_phase[k_h, m] += delta_max * w_b

    D_watermarked = S_mag * np.exp(1.j * modified_phase)
    return librosa.istft(D_watermarked, hop_length=HOP_LENGTH, length=len(y))

def detect_watermark(y_deg, harmonic_bins, B, ref_phase):
    D = librosa.stft(y_deg, n_fft=N_FFT, hop_length=HOP_LENGTH)
    _, S_phase = librosa.magphase(D)
    deg_phase = np.angle(S_phase)

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
    errors = np.sum(np.asarray(ref_bits) != np.asarray(rec_bits))
    return float(errors / n_bits)

def apply_leading_drift(y, drift_samples):
    if drift_samples == 0:
        return y
    y_shifted = y[drift_samples:]
    pad = np.zeros(drift_samples, dtype=y.dtype)
    return np.concatenate([y_shifted, pad])

# --- 4. Reproduce the 249-clip stratified sample to get AudioLDM-2 /
#        Stable Audio Open filenames (same selection used for Table I) ---
records = []
for gen in GENERATORS:
    for i in range(TARGET_PROMPTS):
        for var in range(VARIATIONS_PER_PROMPT):
            records.append({"filename": f"gen_{gen}_p{i:03d}_v{var}.wav",
                             "generator": gen, "prompt_idx": i, "variation": var})

df_meta = pd.DataFrame(records).sort_values("filename").reset_index(drop=True)
df_sample = (df_meta.groupby("generator", group_keys=False)
                     .sample(n=83, random_state=42)
                     .reset_index(drop=True))

clip_lists = {"musicgen": [f"gen_musicgen_p{i:03d}_v0.wav" for i in range(N_SUBSET)]}
for gen in ["audioldm2", "stableaudio"]:
    filenames = sorted(df_sample[df_sample["generator"] == gen]["filename"].tolist())
    clip_lists[gen] = filenames[:N_SUBSET]
    print(f"  {gen}: {len(clip_lists[gen])} clips, first={clip_lists[gen][0]}, "
          f"last={clip_lists[gen][-1]}")

# --- 5. Main Loop ---
print(f"\n--- Cross-generator cliff check, N={N_SUBSET} per generator, "
      f"drift_samples={DRIFT_SAMPLES} ---")

# Mount Google Drive first
from google.colab import drive
drive.mount('/content/drive')

# Get list of available files on Drive once
drive_files = list_drive_files()

state = load_checkpoint()
processed = set(state["processed"])

for gen in GENERATORS:
    payload = generate_watermark(gen)
    for filename in clip_lists[gen]:
        if filename in processed:
            continue

        if filename not in drive_files:
            print(f"[FATAL] Required clip '{filename}' not found on Drive under '{PROJECT_DIR}'.")
            sys.exit(1)

        local_path = os.path.join(LOCAL_WORKSPACE, filename)
        if not load_from_drive(filename, local_path):
            print(f"[FATAL] Could not download clip '{filename}' from Drive.")
            sys.exit(1)

        y_orig, _ = librosa.load(local_path, sr=SR)
        os.remove(local_path)

        _, harmonic_bins, ref_phase = extract_reference_phase(y_orig, SR, H_STARS)
        y_wm = embed_watermark(y_orig, harmonic_bins, H_STARS, B_STARS, payload)

        for drift in DRIFT_SAMPLES:
            y_drift = apply_leading_drift(y_wm, drift)
            recovered = detect_watermark(y_drift, harmonic_bins, B_STARS, ref_phase)
            ber = compute_ber(payload, recovered, B_STARS)
            state["results"].append({
                "generator": gen,
                "filename": filename,
                "drift_samples": drift,
                "drift_ms": 1000.0 * drift / SR,
                "ber": ber,
            })

        state["processed"].append(filename)

    save_checkpoint(state)
    print(f"  {gen} done.")

# --- 6. Aggregation ---
print("\n--- Aggregating cross-generator cliff table ---")
df = pd.DataFrame(state["results"])
local_results = os.path.join(LOCAL_WORKSPACE, RESULTS_FILE)
df.to_json(local_results, orient="records", indent=2)
save_to_drive(local_results, RESULTS_FILE)

agg = (df.groupby(["generator", "drift_samples"])
         .agg(drift_ms=("drift_ms", "first"),
              mean_ber=("ber", lambda x: 100.0 * x.mean()),
              std_ber=("ber", lambda x: 100.0 * x.std()))
         .reset_index())
print(agg.to_string(index=False))

GEN_LABELS = {"musicgen": "MusicGen", "audioldm2": "AudioLDM-2", "stableaudio": "Stable Audio"}

tex_lines = []
tex_lines.append(r"\begin{table}[t]")
tex_lines.append(r"\centering")
tex_lines.append(r"\caption{BER Under Leading-Sample Drift Across Generators ($N=30$ per generator)}")
tex_lines.append(r"\label{tab:cross_generator}")
tex_lines.append(r"\begin{tabular}{cccc}")
tex_lines.append(r"\toprule")
tex_lines.append(r"Drift (ms) & MusicGen & AudioLDM-2 & Stable Audio \\")
tex_lines.append(r"\midrule")
for drift in DRIFT_SAMPLES:
    drift_ms = 1000.0 * drift / SR
    row_cells = [f"{drift_ms:.3f}"]
    for gen in GENERATORS:
        sub = agg[(agg["generator"] == gen) & (agg["drift_samples"] == drift)]
        mean_ber = sub["mean_ber"].iloc[0]
        std_ber = sub["std_ber"].iloc[0]
        row_cells.append(f"{mean_ber:.2f}$\\pm${std_ber:.2f}")
    tex_lines.append(" & ".join(row_cells) + r" \\")
tex_lines.append(r"\bottomrule")
tex_lines.append(r"\end{tabular}")
tex_lines.append(r"\end{table}")
tex_content = "\n".join(tex_lines)
print("\n" + tex_content)

local_table = os.path.join(LOCAL_WORKSPACE, TABLE_FILE)
with open(local_table, "w") as f:
    f.write(tex_content)
save_to_drive(local_table, TABLE_FILE)

# --- Sync to ensure all writes are flushed ---
print("\n[SYNC] Flushing file system buffers...")
os.sync()
print("[SYNC] Complete.")

print(f"\n[SUCCESS] Cross-generator cliff check complete. Results: {RESULTS_FILE}, "
      f"LaTeX: {TABLE_FILE} (saved to Drive).")