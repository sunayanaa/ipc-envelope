# ==============================================================================
# Program Name: spl07_robustness_battery.py
# Version: 1.0
# Description: Generates a robustness-battery table for IPC (Delta_max=pi/4,
#              H*=8, B*=32) on the same N=50 MusicGen subset as spl05 Part B
#              (gen_musicgen_p000_v0.wav ... p049_v0.wav). For each clip, the
#              watermark is embedded once; the SAME watermarked signal is then
#              tested under six conditions: clean (no perturbation), MP3 @
#              128 kbps, MP3 @ 64 kbps, AWGN @ 30 dB SNR, AWGN @ 20 dB SNR, and
#              a 16->44.1->16 kHz resample round-trip. Mean/std BER is reported
#              per condition. Intended as a contrast to Table~\ref{tab:drift}:
#              IPC's response to these (often more aggressive) common
#              signal-processing operations versus its collapse under just
#              10 samples of leading drift.
#
#              Each perturbed signal is length-matched to the original before
#              detection (MP3 and resample round-trips can shift length by a
#              few samples; detect_watermark indexes ref_phase/harmonic_bins
#              by STFT frame and assumes the original frame count).
#
# Change Log: 1.0 - Initial version.
# GPU Required: NO. Pure CPU: numpy/librosa/soxr plus the `lame` CLI for MP3.
# ==============================================================================

import os
import sys
import json
import hashlib
import subprocess
import shutil
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import soxr

# --- 0. Dynamic Environment Setup ---
if subprocess.run("which lame", shell=True, capture_output=True).returncode != 0:
    print("\n--- Installing lame ---")
    subprocess.run("apt update -y && apt install -y lame", shell=True, check=True)
try:
    import soxr  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "soxr", "librosa", "soundfile", "pandas", "numpy"], check=True)

# --- 1. Configuration ---
PROJECT_DIR = "/content/drive/MyDrive/ipc-envelope/"  # Persistent storage
LOCAL_WORKSPACE = "/content/spl07_workspace"
os.makedirs(LOCAL_WORKSPACE, exist_ok=True)

SR = 16000
N_FFT = 2048
HOP_LENGTH = 512
H_STARS = 8
B_STARS = 32
DELTA_MAX = np.pi / 4

N_SUBSET = 50  # p000-p049, v0

CONDITIONS = [
    ("clean", "Clean"),
    ("mp3_128", "MP3 @ 128 kbps"),
    ("mp3_64", "MP3 @ 64 kbps"),
    ("awgn_30", "AWGN, 30 dB SNR"),
    ("awgn_20", "AWGN, 20 dB SNR"),
    ("resample", "Resample 16$\\to$44.1$\\to$16 kHz"),
]

CHECKPOINT_FILE = "spl07_checkpoint.json"
RESULTS_FILE = "spl07_results.json"
TABLE_FILE = "spl07_table.tex"

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

# --- 3. IPC Core (Delta_max=pi/4 fixed, same as spl02/spl06) ---
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

def match_length(y, target_len):
    if len(y) > target_len:
        return y[:target_len]
    elif len(y) < target_len:
        return np.concatenate([y, np.zeros(target_len - len(y), dtype=y.dtype)])
    return y

# --- 4. Perturbations ---
def mp3_roundtrip(y, sr, bitrate_kbps):
    tmp_wav = os.path.join(LOCAL_WORKSPACE, "tmp_mp3in.wav")
    tmp_mp3 = os.path.join(LOCAL_WORKSPACE, "tmp.mp3")
    tmp_out = os.path.join(LOCAL_WORKSPACE, "tmp_mp3out.wav")
    sf.write(tmp_wav, y, sr, subtype='PCM_16')
    subprocess.run(["lame", "--quiet", "-b", str(bitrate_kbps), tmp_wav, tmp_mp3], check=True)
    subprocess.run(["lame", "--quiet", "--decode", tmp_mp3, tmp_out], check=True)
    y_out, _ = librosa.load(tmp_out, sr=sr)
    for p in [tmp_wav, tmp_mp3, tmp_out]:
        if os.path.exists(p):
            os.remove(p)
    return y_out

def add_awgn(y, snr_db, seed):
    rng = np.random.default_rng(seed)
    signal_power = np.mean(y.astype(np.float64) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = rng.normal(0, np.sqrt(noise_power), size=len(y))
    return (y + noise).astype(y.dtype)

def resample_roundtrip(y, sr=SR, intermediate_sr=44100):
    y_up = soxr.resample(y, sr, intermediate_sr)
    return soxr.resample(y_up, intermediate_sr, sr)

def apply_condition(y_wm, condition, filename):
    if condition == "clean":
        return y_wm
    if condition == "mp3_128":
        return mp3_roundtrip(y_wm, SR, 128)
    if condition == "mp3_64":
        return mp3_roundtrip(y_wm, SR, 64)
    if condition == "awgn_30":
        seed = int(hashlib.md5(f"{filename}_awgn_30".encode()).hexdigest(), 16) % (2**32)
        return add_awgn(y_wm, 30, seed)
    if condition == "awgn_20":
        seed = int(hashlib.md5(f"{filename}_awgn_20".encode()).hexdigest(), 16) % (2**32)
        return add_awgn(y_wm, 20, seed)
    if condition == "resample":
        return resample_roundtrip(y_wm)
    raise ValueError(f"Unknown condition: {condition}")

# --- 5. Main Loop ---
print(f"\n--- Robustness battery, N={N_SUBSET}, conditions={[c[0] for c in CONDITIONS]} ---")

# Mount Google Drive first
from google.colab import drive
drive.mount('/content/drive')

# Get list of available files on Drive once
drive_files = list_drive_files()

state = load_checkpoint()
processed = set(state["processed"])
payload = generate_watermark("musicgen")

for i in range(N_SUBSET):
    filename = f"gen_musicgen_p{i:03d}_v0.wav"
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

    for condition, _label in CONDITIONS:
        y_test = apply_condition(y_wm, condition, filename)
        y_test = match_length(y_test, len(y_orig))
        recovered = detect_watermark(y_test, harmonic_bins, B_STARS, ref_phase)
        ber = compute_ber(payload, recovered, B_STARS)
        state["results"].append({
            "filename": filename,
            "condition": condition,
            "ber": ber,
        })

    state["processed"].append(filename)
    if len(state["processed"]) % 10 == 0:
        print(f"  {len(state['processed'])}/{N_SUBSET}")
        save_checkpoint(state)

save_checkpoint(state)
print(f"--- Done: {len(state['processed'])}/{N_SUBSET} ---")

# --- 6. Aggregation ---
print("\n--- Aggregating robustness battery table ---")
df = pd.DataFrame(state["results"])
local_results = os.path.join(LOCAL_WORKSPACE, RESULTS_FILE)
df.to_json(local_results, orient="records", indent=2)
save_to_drive(local_results, RESULTS_FILE)

agg = (df.groupby("condition")
         .agg(mean_ber=("ber", lambda x: 100.0 * x.mean()),
              std_ber=("ber", lambda x: 100.0 * x.std()))
         .reset_index())
print(agg.to_string(index=False))

tex_lines = []
tex_lines.append(r"\begin{table}[t]")
tex_lines.append(r"\centering")
tex_lines.append(r"\caption{IPC BER Under Common Signal-Processing Operations ($N=50$, $\Delta_{\max}=\pi/4$)}")
tex_lines.append(r"\label{tab:robustness}")
tex_lines.append(r"\begin{tabular}{lcc}")
tex_lines.append(r"\toprule")
tex_lines.append(r"Condition & Mean BER (\%) & Std (\%) \\")
tex_lines.append(r"\midrule")
for cond, label in CONDITIONS:
    row = agg[agg["condition"] == cond].iloc[0]
    tex_lines.append(f"{label} & {row['mean_ber']:.2f} & {row['std_ber']:.2f} \\\\")
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

print(f"\n[SUCCESS] Robustness battery complete. Results: {RESULTS_FILE}, "
      f"LaTeX: {TABLE_FILE} (saved to Drive).")