# ==============================================================================
# Program Name: spl05_delta_sweep.py
# Version: 1.0
# Description: Generates the Delta_max sweep figure (two stacked panels).
#
#              Part A: for Delta_max in {pi/8, pi/4, pi/2}, repeats Table II's
#              leading-sample drift sweep {0,10,50,100,500,1000} samples on
#              the same N=200 MusicGen clips (gen_musicgen_p000_v0.wav ...
#              gen_musicgen_p199_v0.wav), recording mean BER per (Delta_max,
#              drift). The Delta_max=pi/4 curve should reproduce Table II's
#              numbers (same clips, same protocol) as a consistency check.
#
#              Part B: for the same three Delta_max values, embeds IPC on a
#              N=50 subset of those clips (p000-p049, v0) and computes mean
#              PEAQ ODG, using the converged-final-frame ODG extraction.
#
#              Together: Panel 1 = BER vs drift for each Delta_max (does the
#              sync cliff move?). Panel 2 = mean ODG vs Delta_max
#              (imperceptibility cost of a larger perturbation budget).
#
#              f0 and harmonic-bin computation (the expensive pYIN step) is
#              done ONCE per clip and reused across all three Delta_max
#              values, since neither depends on Delta_max.
#
# Change Log: 1.0 - Initial version.
# GPU Required: NO. Part A is pure numpy/librosa; Part B uses the peaqb-fast
#                CPU binary (same as spl01, without WavMark).
# ==============================================================================

import os
import subprocess
import sys

PEAQ_BINARY_PATH = "/content/peaqb-fast/src/peaqb"

# --- 0. Dynamic Environment Setup (PEAQ only needed for Part B) ---
if not os.path.exists(PEAQ_BINARY_PATH):
    print("\n--- Environment missing. Installing dependencies and compiling PEAQ ---")
    subprocess.run("apt-get update -y", shell=True, check=True)
    subprocess.run("apt-get install -y lame libsndfile1-dev", shell=True, check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "soxr", "librosa", "soundfile", "pandas", "numpy",
                    "matplotlib"], check=True)
    subprocess.run("rm -rf /content/peaqb-fast", shell=True)
    subprocess.run("git clone https://github.com/akinori-ito/peaqb-fast.git /content/peaqb-fast", shell=True, check=True)
    subprocess.run("cd /content/peaqb-fast && ./configure && make", shell=True, check=True)
    print("--- Environment setup complete ---\n")

import json
import shutil
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import soxr
import matplotlib.pyplot as plt

# --- 1. Configuration ---
PROJECT_DIR = "/content/drive/MyDrive/ipc-envelope/"  # Persistent storage
LOCAL_WORKSPACE = "/content/spl05_workspace"
os.makedirs(LOCAL_WORKSPACE, exist_ok=True)

SR = 16000
N_FFT = 2048
HOP_LENGTH = 512
H_STARS = 8
B_STARS = 32

N_CLIPS_A = 200   # gen_musicgen_p000_v0.wav ... p199_v0.wav
N_CLIPS_B = 50    # subset: p000_v0.wav ... p049_v0.wav
DRIFT_SAMPLES = [0, 10, 50, 100, 500, 1000]

DELTA_VALUES = [np.pi / 8, np.pi / 4, np.pi / 2]
DELTA_LABELS = [r"$\pi/8$", r"$\pi/4$", r"$\pi/2$"]
DELTA_COLORS = ['#56B4E9', '#E69F00', '#D55E00']  # blue, orange, vermillion

CKPT_A = "spl05a_checkpoint.json"
CKPT_B = "spl05b_checkpoint.json"
RESULTS_A = "spl05a_results.json"
RESULTS_B = "spl05b_results.json"
FIGURE_FILE = "fig_spl05_delta_sweep.png"

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

def load_checkpoint(name):
    """Load checkpoint from Google Drive project folder."""
    local = os.path.join(LOCAL_WORKSPACE, name)
    if load_from_drive(name, local):
        with open(local) as f:
            return json.load(f)
    return {"processed": [], "results": []}

def save_checkpoint(state, name):
    """Save checkpoint to Google Drive project folder."""
    local = os.path.join(LOCAL_WORKSPACE, name)
    with open(local, "w") as f:
        json.dump(state, f)
    save_to_drive(local, name)

# --- 3. IPC Core ---
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

def embed_watermark(y, harmonic_bins, H, B, watermark_bits, delta_max):
    """Delta_max-parameterised embedding. f0/harmonic_bins are computed once
    per clip (independent of Delta_max) and passed in."""
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

# --- 4. PEAQ Helpers (Part B only) ---
def write_peaq_wav(y_mono_16k, path):
    y_48k = soxr.resample(y_mono_16k, SR, 48000)
    y_stereo = np.stack([y_48k, y_48k], axis=1)
    sf.write(path, y_stereo, 48000, subtype='PCM_16')

def calculate_peaq_odg(ref_path, deg_path):
    """Returns the LAST 'ODG:' line (converged whole-clip value)."""
    try:
        result = subprocess.run(
            [PEAQ_BINARY_PATH, "-r", ref_path, "-t", deg_path],
            capture_output=True, text=True, timeout=120
        )
        odg_value = np.nan
        for line in result.stdout.splitlines():
            if "ODG:" in line:
                odg_value = float(line.strip().split(":")[-1].strip())
        return odg_value
    except Exception as e:
        print(f"PEAQ failed: {e}")
        return np.nan

# ==============================================================================
# PART A: BER vs drift for Delta_max in {pi/8, pi/4, pi/2}, N=200 MusicGen
# ==============================================================================
print(f"\n--- Part A: drift sweep x Delta_max, N={N_CLIPS_A} ---")

# Mount Google Drive first
from google.colab import drive
drive.mount('/content/drive')

# Get list of available files on Drive once
drive_files = list_drive_files()

state_a = load_checkpoint(CKPT_A)
processed_a = set(state_a["processed"])
target_payload = generate_watermark("musicgen")

for i in range(N_CLIPS_A):
    filename = f"gen_musicgen_p{i:03d}_v0.wav"
    if filename in processed_a:
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

    f0_track, harmonic_bins, ref_phase = extract_reference_phase(y_orig, SR, H_STARS)

    for delta_max, delta_label in zip(DELTA_VALUES, DELTA_LABELS):
        y_wm = embed_watermark(y_orig, harmonic_bins, H_STARS, B_STARS, target_payload, delta_max)
        for drift in DRIFT_SAMPLES:
            y_drift = apply_leading_drift(y_wm, drift)
            recovered = detect_watermark(y_drift, harmonic_bins, B_STARS, ref_phase)
            ber = compute_ber(target_payload, recovered, B_STARS)
            state_a["results"].append({
                "filename": filename,
                "delta_max": delta_max,
                "delta_label": delta_label,
                "drift_samples": drift,
                "drift_ms": 1000.0 * drift / SR,
                "ber": ber,
            })

    state_a["processed"].append(filename)
    if len(state_a["processed"]) % 20 == 0:
        print(f"  Part A: {len(state_a['processed'])}/{N_CLIPS_A}")
        save_checkpoint(state_a, CKPT_A)

save_checkpoint(state_a, CKPT_A)
print(f"--- Part A done: {len(state_a['processed'])}/{N_CLIPS_A} ---")

df_a = pd.DataFrame(state_a["results"])
local_a = os.path.join(LOCAL_WORKSPACE, RESULTS_A)
df_a.to_json(local_a, orient="records", indent=2)
save_to_drive(local_a, RESULTS_A)

agg_a = (df_a.groupby(["delta_label", "drift_samples"])
              .agg(drift_ms=("drift_ms", "first"),
                   mean_ber=("ber", lambda x: 100.0 * x.mean()),
                   std_ber=("ber", lambda x: 100.0 * x.std()))
              .reset_index())
print(agg_a.to_string(index=False))

# ==============================================================================
# PART B: ODG vs Delta_max, N=50 MusicGen subset
# ==============================================================================
print(f"\n--- Part B: ODG vs Delta_max, N={N_CLIPS_B} ---")
state_b = load_checkpoint(CKPT_B)
processed_b = set(state_b["processed"])

for i in range(N_CLIPS_B):
    filename = f"gen_musicgen_p{i:03d}_v0.wav"
    if filename in processed_b:
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

    f0_track, harmonic_bins, _ = extract_reference_phase(y_orig, SR, H_STARS)

    tmp_orig = os.path.join(LOCAL_WORKSPACE, "tmp_orig.wav")
    write_peaq_wav(y_orig, tmp_orig)

    for delta_max, delta_label in zip(DELTA_VALUES, DELTA_LABELS):
        y_wm = embed_watermark(y_orig, harmonic_bins, H_STARS, B_STARS, target_payload, delta_max)
        tmp_wm = os.path.join(LOCAL_WORKSPACE, "tmp_wm.wav")
        write_peaq_wav(y_wm, tmp_wm)
        odg = calculate_peaq_odg(tmp_orig, tmp_wm)
        os.remove(tmp_wm)

        state_b["results"].append({
            "filename": filename,
            "delta_max": delta_max,
            "delta_label": delta_label,
            "odg": odg,
        })

    os.remove(tmp_orig)
    state_b["processed"].append(filename)
    if len(state_b["processed"]) % 10 == 0:
        print(f"  Part B: {len(state_b['processed'])}/{N_CLIPS_B}")
        save_checkpoint(state_b, CKPT_B)

save_checkpoint(state_b, CKPT_B)
print(f"--- Part B done: {len(state_b['processed'])}/{N_CLIPS_B} ---")

df_b = pd.DataFrame(state_b["results"])
local_b = os.path.join(LOCAL_WORKSPACE, RESULTS_B)
df_b.to_json(local_b, orient="records", indent=2)
save_to_drive(local_b, RESULTS_B)

agg_b = (df_b.groupby("delta_label")
              .agg(delta_max=("delta_max", "first"),
                   mean_odg=("odg", "mean"),
                   std_odg=("odg", "std"))
              .reset_index()
              .sort_values("delta_max"))
print(agg_b.to_string(index=False))

# ==============================================================================
# Figure: two stacked panels
# ==============================================================================
print("\n--- Generating Fig. fig_spl05_delta_sweep ---")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.45, 4.6), dpi=300)

def bare(label):
    """delta_label values are already e.g. '$\\pi/8$'; strip the '$' before
    embedding in a larger mathtext expression to avoid nested math mode."""
    return label.strip("$")

for delta_label, color in zip(DELTA_LABELS, DELTA_COLORS):
    sub = agg_a[agg_a["delta_label"] == delta_label].sort_values("drift_ms")
    ax1.plot(sub["drift_ms"], sub["mean_ber"], 'o-', color=color,
             markersize=4, linewidth=1.5, label=fr"$\Delta_{{\max}}={bare(delta_label)}$")

ax1.set_xscale('symlog', linthresh=1, linscale=1)
ax1.set_xlabel('Leading Sample Drift (ms)', fontsize=9)
ax1.set_ylabel('Mean BER (%)', fontsize=9)
ax1.tick_params(axis='both', labelsize=8)
ax1.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.6)
ax1.legend(fontsize=7, loc='lower right', frameon=True)

x_pos = np.arange(len(agg_b))
ax2.errorbar(x_pos, agg_b["mean_odg"], yerr=agg_b["std_odg"],
              fmt='o', color='#009E73', ecolor='#009E73',
              elinewidth=1.0, capsize=4, markersize=6)
ax2.set_xticks(x_pos)
ax2.set_xticklabels([fr"$\Delta_{{\max}}={bare(l)}$" for l in agg_b["delta_label"]], fontsize=8)
ax2.set_ylabel('Mean PEAQ ODG', fontsize=9)
ax2.tick_params(axis='y', labelsize=8)
ax2.grid(True, axis='y', linestyle=':', linewidth=0.5, alpha=0.6)
ax2.set_xlim(-0.5, len(agg_b) - 0.5)

fig.tight_layout(pad=0.6)
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

print(f"\n[SUCCESS] Delta_max sweep complete. Results: {RESULTS_A}, {RESULTS_B}, "
      f"Figure: {FIGURE_FILE} (saved to Drive).")