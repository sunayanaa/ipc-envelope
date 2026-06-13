# ==============================================================================
# Program Name: spl03_audioseal_drift.py
# Version: 1.0
# Description: Generates Table III: AudioSeal (16-bit) BER under the same
#              leading-sample drift sweep as Table II, on the same N=200
#              musicgen clips (p000-p199, v0). Each clip is embedded once
#              with AudioSeal's 16-bit watermark; the SAME watermarked
#              signal is then tested at drift levels {0,10,50,100,500,1000}
#              samples (0 - 62.5 ms at 16 kHz), and BER (fraction of the
#              16-bit message decoded incorrectly) is recorded per drift
#              level. Aggregated as mean/std BER per drift level, mirroring
#              Table II's format, for direct comparison against IPC's
#              sync-sensitivity curve.
#
#              NOTE: BER is reported as a fraction (bit-error RATE), so the
#              16-bit AudioSeal message and the 32-bit IPC payload (B*=32)
#              remain directly comparable on this axis.
#
#              IMPORTANT: apply_leading_drift() below is THIS script's
#              definition of "leading sample drift" (prepend d zero samples,
#              truncate to original length L). VERIFY this matches the drift
#              operation used for Table II before trusting the comparison;
#              if the original convention differs (e.g. trimming the start
#              instead of prepending), update apply_leading_drift() to match
#              exactly, everything else in this script is independent of
#              that choice.
#
# Change Log: 1.0 - Initial version.
# GPU Required: RECOMMENDED (not strict). 200 clips x (1 embed + 6 detects)
#                = 1400 AudioSeal forward passes; fast on a T4, slower but
#                feasible on CPU.
# ==============================================================================

import os
import sys
import json
import hashlib
import shutil
import subprocess

# --- 0. Dynamic Environment Setup ---
try:
    import audioseal  # noqa: F401
except ImportError:
    print("\n--- Installing AudioSeal and dependencies ---")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "audioseal", "librosa", "soundfile", "numpy", "pandas"],
                   check=True)
    print("--- Environment setup complete ---\n")

import numpy as np
import pandas as pd
import torch
import librosa
from audioseal import AudioSeal

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# --- 1. Configuration ---
PROJECT_DIR = "/content/drive/MyDrive/ipc-envelope/"  # Persistent storage
LOCAL_WORKSPACE = "/content/spl03_workspace"
os.makedirs(LOCAL_WORKSPACE, exist_ok=True)

SR = 16000
N_CLIPS = 200
DRIFT_LEVELS_SAMPLES = [0, 10, 50, 100, 500, 1000]
AUDIOSEAL_BITS = 16

CHECKPOINT_FILE = "spl03_checkpoint.json"

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

# --- 3. Checkpoint ---
def load_checkpoint():
    """Load checkpoint from Google Drive project folder."""
    local_ckpt = os.path.join(LOCAL_WORKSPACE, CHECKPOINT_FILE)
    if load_from_drive(CHECKPOINT_FILE, local_ckpt):
        with open(local_ckpt) as f:
            state = json.load(f)
        print(f"--- Resumed checkpoint: {len(state['processed'])} clips done ---")
        return state
    return {"processed": [], "results": []}

def save_checkpoint(state):
    """Save checkpoint to Google Drive project folder."""
    local_ckpt = os.path.join(LOCAL_WORKSPACE, CHECKPOINT_FILE)
    with open(local_ckpt, "w") as f:
        json.dump(state, f, indent=2)
    save_to_drive(local_ckpt, CHECKPOINT_FILE)

# --- 4. Payload ---
def get_payload_audioseal(clip_id, bits=AUDIOSEAL_BITS):
    seed = int(hashlib.md5(clip_id.encode()).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=bits)

# --- 5. AudioSeal Embed / Decode ---
def embed_audioseal(y, payload, generator):
    audio = torch.from_numpy(y).float().reshape(1, 1, -1).to(DEVICE)
    message = torch.from_numpy(payload.astype(np.int64)).reshape(1, -1).to(DEVICE)
    with torch.no_grad():
        wm = generator.get_watermark(audio, sample_rate=SR, message=message)
        y_wm = (audio + wm).squeeze().detach().cpu().numpy()
    return y_wm

def decode_audioseal(y, detector):
    audio = torch.from_numpy(y).float().reshape(1, 1, -1).to(DEVICE)
    with torch.no_grad():
        _, message = detector.detect_watermark(audio, sample_rate=SR)
    bits = message.squeeze().detach().cpu().numpy()
    return (bits > 0.5).astype(int)

# --- 6. Drift ---
def apply_leading_drift(y, d):
    """Prepend d zero samples, truncate to original length L.
    SEE HEADER NOTE: verify against Table II's drift convention."""
    if d == 0:
        return y.copy()
    L = len(y)
    return np.concatenate([np.zeros(d, dtype=y.dtype), y])[:L]

# --- 7. Load Models ---
print("\n--- Loading AudioSeal models ---")
generator = AudioSeal.load_generator("audioseal_wm_16bits").to(DEVICE)
detector = AudioSeal.load_detector("audioseal_detector_16bits").to(DEVICE)
generator.eval()
detector.eval()
print("AudioSeal models loaded.")

# --- 8. Main Loop ---
# Mount Google Drive first
from google.colab import drive
drive.mount('/content/drive')

state = load_checkpoint()
processed_set = set(state["processed"])

# Get list of available files on Drive once
drive_files = list_drive_files()

for idx in range(N_CLIPS):
    clip_id = f"musicgen_p{idx:03d}_v0"
    filename = f"gen_{clip_id}.wav"

    if clip_id in processed_set:
        continue

    local_path = os.path.join(LOCAL_WORKSPACE, filename)

    if filename not in drive_files:
        print(f"  [WARN] {filename} not found on Drive. Skipping.")
        continue

    if not load_from_drive(filename, local_path):
        print(f"  [WARN] Could not download {filename} from Drive. Skipping.")
        continue

    y, _ = librosa.load(local_path, sr=SR)
    os.remove(local_path)

    payload = get_payload_audioseal(clip_id)
    y_wm = embed_audioseal(y, payload, generator)

    row = {"clip_id": clip_id}
    for d in DRIFT_LEVELS_SAMPLES:
        y_drift = apply_leading_drift(y_wm, d)
        decoded = decode_audioseal(y_drift, detector)
        ber = float(np.mean(decoded != payload))
        row[f"ber_d{d}"] = ber

    state["results"].append(row)
    state["processed"].append(clip_id)

    if len(state["processed"]) % 20 == 0:
        print(f"  Processed {len(state['processed'])}/{N_CLIPS}")
        save_checkpoint(state)

save_checkpoint(state)
print(f"--- Done: {len(state['processed'])}/{N_CLIPS} clips processed ---")

# --- 9. Aggregate Table III ---
if len(state["processed"]) < N_CLIPS:
    print("[WARN] Not all clips processed; aggregating partial results.")

df = pd.DataFrame(state["results"])

rows = []
for d in DRIFT_LEVELS_SAMPLES:
    col = f"ber_d{d}"
    mean_ber = df[col].mean() * 100
    std_ber = df[col].std() * 100
    rows.append({
        "drift_samples": d,
        "drift_ms": d / 16.0,
        "mean_ber": mean_ber,
        "std_ber": std_ber,
    })

table3_df = pd.DataFrame(rows)
print("\n--- Aggregating Table III ---")
print(table3_df.to_string(index=False))

results_path = os.path.join(LOCAL_WORKSPACE, "spl03_table3_results.json")
with open(results_path, "w") as f:
    json.dump(state["results"], f, indent=2)
save_to_drive(results_path, "spl03_table3_results.json")

tex_lines = []
tex_lines.append(r"\begin{table}[t]")
tex_lines.append(r"\centering")
tex_lines.append(r"\caption{AudioSeal (16-bit) BER Under Leading-Sample Drift ($N=200$)}")
tex_lines.append(r"\label{tab:audioseal_drift}")
tex_lines.append(r"\begin{tabular}{ccc}")
tex_lines.append(r"\toprule")
tex_lines.append(r"Drift (ms) & Mean BER (\%) & Std (\%) \\")
tex_lines.append(r"\midrule")
for _, r in table3_df.iterrows():
    tex_lines.append(f"{r['drift_ms']:.3f} & {r['mean_ber']:.2f} & {r['std_ber']:.2f} \\\\")
tex_lines.append(r"\bottomrule")
tex_lines.append(r"\end{tabular}")
tex_lines.append(r"\end{table}")
tex_content = "\n".join(tex_lines)
print("\n" + tex_content)

tex_path = os.path.join(LOCAL_WORKSPACE, "spl03_table3.tex")
with open(tex_path, "w") as f:
    f.write(tex_content)
save_to_drive(tex_path, "spl03_table3.tex")

# --- Sync to ensure all writes are flushed ---
print("\n[SYNC] Flushing file system buffers...")
os.sync()
print("[SYNC] Complete.")

print("\n[SUCCESS] Table III complete. Results: spl03_table3_results.json, "
      "LaTeX: spl03_table3.tex (saved to Drive).")