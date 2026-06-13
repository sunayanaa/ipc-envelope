# ==============================================================================
# Program Name: spl01_baseline_comparison.py
# Version: 1.0
# Description: SPL Letter reproducibility script - generates Table I
#              (Imperceptibility and Clean-Channel Detection). Compares the
#              proposed IPC watermark against a Spread-Spectrum baseline and
#              WavMark, evaluated WITHIN A SINGLE LOOP on a single shared
#              249-clip stratified sample (83 clips per generator: MusicGen,
#              AudioLDM-2, Stable Audio Open) drawn directly from the Stage 0
#              corpus (00_data_generation.py outputs on Google Drive).
#
#              #
#              The 249-clip sample is defined here from first principles
#              (sorted filename list + groupby('generator').sample(n=83,
#              random_state=42)), so it is independently reconstructible
#              without depending on a pre-built baseline_audio.zip. 
#
# Change Log: 1.0 - Initial version. 
# GPU Required: YES (for WavMark encode/decode; PEAQ and IPC run on CPU)
# ==============================================================================

import os
import subprocess
import sys
import time

PEAQ_BINARY_PATH = "/content/peaqb-fast/src/peaqb"

# --- 0. Dynamic Environment Setup (identical to 06_unified_baselines.py) ---
if not os.path.exists(PEAQ_BINARY_PATH):
    print("\n--- Environment missing. Installing dependencies and compiling PEAQ ---")
    subprocess.run("apt-get update -y", shell=True, check=True)
    subprocess.run("apt-get install -y lame libsndfile1-dev", shell=True, check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "wavmark==0.0.3", "soxr", "librosa", "soundfile",
                    "pandas", "numpy", "scikit-learn"], check=True)
    subprocess.run("rm -rf /content/peaqb-fast", shell=True)
    subprocess.run("git clone https://github.com/akinori-ito/peaqb-fast.git /content/peaqb-fast", shell=True, check=True)
    subprocess.run("cd /content/peaqb-fast && ./configure && make", shell=True, check=True)
    print("--- Environment setup complete ---\n")

import json
import hashlib
import shutil
import pandas as pd
import numpy as np
import torch
import librosa
import soundfile as sf
import soxr
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)

# --- 1. GPU Check ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device Initialization: {DEVICE}")
if DEVICE.type != "cuda":
    print("\n[WARNING] No GPU detected. WavMark encode/decode will run on CPU "
          "and will be substantially slower for 249 clips.")

# --- 2. Configuration ---
PROJECT_DIR = "/content/drive/MyDrive/ipc-envelope/"  # Persistent storage
LOCAL_WORKSPACE = "/content/spl01_workspace"
os.makedirs(LOCAL_WORKSPACE, exist_ok=True)

CHECKPOINT_FILE = "spl01_checkpoint.json"
RESULTS_FILE = "spl01_table1_results.json"
TABLE_FILE = "spl01_table1.tex"

GENERATORS = ["musicgen", "audioldm2", "stableaudio"]
TARGET_PROMPTS = 200
VARIATIONS_PER_PROMPT = 5
N_PER_GENERATOR = 83  # 83 x 3 = 249, matches the TASLP draft's stratified sample size

# --- Audio & Watermark Params ---
SR = 16000
N_FFT = 2048
HOP_LENGTH = 512
H_STARS = 8
B_STARS = 32
ALPHA_SS = 0.01
B_PAYLOAD_WAVMARK = 16
WAVMARK_MIN_SAMPLES = 17600
WAVMARK_VERSION = "0.0.3"

# --- 3. Google Drive Helper Functions ---
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

# --- 4. Build the 249-clip Stratified Sample (deterministic, no zip dependency) ---
print("\n--- Building Stratified Sample (83 per generator, random_state=42) ---")
records = []
for gen in GENERATORS:
    for i in range(TARGET_PROMPTS):
        for var in range(VARIATIONS_PER_PROMPT):
            records.append({"filename": f"gen_{gen}_p{i:03d}_v{var}.wav", "generator": gen})

# Sort by filename BEFORE sampling so the sample is independent of any
# filesystem/glob ordering and is reproducible from this script alone.
df_meta = pd.DataFrame(records).sort_values("filename").reset_index(drop=True)
df_sample = (df_meta.groupby("generator", group_keys=False)
                     .sample(n=N_PER_GENERATOR, random_state=42)
                     .reset_index(drop=True))
print(f"Sample size: {len(df_sample)} clips ({N_PER_GENERATOR} per generator)")

# --- 5. IPC Core (verbatim from 02_exp1_imperceptibility.py / 03_exp2_exp4_detection.py) ---
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

def embed_watermark_ipc(y, sr, H, B, watermark_bits):
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

def extract_reference_phase_ipc(y_orig, sr, H):
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

def detect_watermark_ipc(y_deg, sr, H, B, f0, ref_phase):
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

def compute_ber(ref_bits, rec_bits, n_bits):
    assert len(ref_bits) == len(rec_bits) == n_bits
    errors = np.sum(np.asarray(ref_bits) != np.asarray(rec_bits))
    return float(errors / n_bits)

# --- 6. Spread-Spectrum Core (verbatim from 06_unified_baselines.py) ---
def get_ss_carriers(n_freqs, n_frames, bits=B_STARS, seed=42):
    np.random.seed(seed)
    return np.random.randn(bits, n_freqs, n_frames).astype(np.float32)

def embed_spread_spectrum(y_orig, payload_bits):
    D = librosa.stft(y_orig, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mag, phase = librosa.magphase(D)
    log_mag = np.log1p(mag)

    n_freqs, n_frames = log_mag.shape
    carriers = get_ss_carriers(n_freqs, n_frames, bits=B_STARS)

    for i, bit in enumerate(payload_bits):
        log_mag += ALPHA_SS * bit * carriers[i]

    mag_wm = np.expm1(log_mag)
    mag_wm = np.maximum(mag_wm, 0)

    D_wm = mag_wm * phase
    y_wm = librosa.istft(D_wm, hop_length=HOP_LENGTH, length=len(y_orig))
    return y_wm, n_frames

def detect_spread_spectrum(y_deg, ref_frames):
    D = librosa.stft(y_deg, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mag, _ = librosa.magphase(D)
    log_mag = np.log1p(mag)

    if log_mag.shape[1] < ref_frames:
        pad_width = ref_frames - log_mag.shape[1]
        log_mag = np.pad(log_mag, ((0, 0), (0, pad_width)), mode='constant')
    else:
        log_mag = log_mag[:, :ref_frames]

    n_freqs = log_mag.shape[0]
    carriers = get_ss_carriers(n_freqs, ref_frames, bits=B_STARS)

    rec_bits = []
    for i in range(B_STARS):
        correlation = np.sum(log_mag * carriers[i])
        rec_bits.append(1 if correlation > 0 else -1)

    return np.array(rec_bits).astype(np.float32)

def get_payload_ss(generator_id: str, bits=B_STARS):
    seed = int(hashlib.md5(generator_id.encode()).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)
    return rng.choice([-1, 1], size=bits).astype(np.float32)

# --- 7. WavMark Core (verbatim from 06_unified_baselines.py) ---
print("\n--- Loading WavMark Model ---")
import wavmark
try:
    assert wavmark.__version__ == WAVMARK_VERSION
except AttributeError:
    print("WARNING: wavmark.__version__ not available.")

wavmark_model = wavmark.load_model().to(DEVICE)
print("WavMark Model Loaded.")

def get_payload_wavmark(generator_id: str):
    seed = int(hashlib.md5(generator_id.encode()).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)
    return rng.choice([0, 1], size=B_PAYLOAD_WAVMARK).tolist()

def encode_wavmark_wrapper(y_orig, generator_id):
    if len(y_orig) < WAVMARK_MIN_SAMPLES:
        pad = np.zeros(WAVMARK_MIN_SAMPLES - len(y_orig), dtype=np.float32)
        y_orig = np.concatenate([y_orig, pad])

    wm_payload = get_payload_wavmark(generator_id)
    y_np = np.array(y_orig, dtype=np.float32)

    with torch.no_grad():
        encoded_signal, info = wavmark.encode_watermark(
            wavmark_model, y_np, wm_payload, show_progress=False
        )
    return encoded_signal, info

def decode_wavmark_wrapper(y_deg):
    if len(y_deg) < WAVMARK_MIN_SAMPLES:
        pad = np.zeros(WAVMARK_MIN_SAMPLES - len(y_deg), dtype=np.float32)
        y_deg = np.concatenate([y_deg, pad])

    y_np = np.array(y_deg, dtype=np.float32)

    with torch.no_grad():
        decoded_payload, _ = wavmark.decode_watermark(
            wavmark_model, y_np, show_progress=False
        )

    if decoded_payload is None:
        return np.random.randint(0, 2, size=B_PAYLOAD_WAVMARK).astype(np.float32)

    result = np.array(decoded_payload, dtype=np.float32).flatten()
    assert result.shape == (B_PAYLOAD_WAVMARK,), \
        f"Unexpected decode shape: {result.shape}"
    return result

def compute_ber_wavmark(generator_id, rec_bits):
    ref_bits = np.array(get_payload_wavmark(generator_id), dtype=np.float32)
    rec_bits = np.array(rec_bits, dtype=np.float32).flatten()
    return float(np.sum(ref_bits != rec_bits) / B_PAYLOAD_WAVMARK)

# --- 8. PEAQ ODG (verbatim from 06_unified_baselines.py: peaqb-fast, 48k stereo) ---
def write_peaq_wav(y_mono_16k, path):
    y_48k = soxr.resample(y_mono_16k, SR, 48000)
    y_stereo = np.stack([y_48k, y_48k], axis=1)
    sf.write(path, y_stereo, 48000, subtype='PCM_16')

def calculate_peaq_odg(ref_path, deg_path):
    """
    peaqb-fast prints a running, cumulative-average ODG/DI once per PEAQ
    frame (~468 frames for a 10s clip at 48kHz); each successive line
    converges toward the whole-clip value as more frames are averaged in.
    The FINAL frame's value is the converged whole-clip ODG.

    NOTE: an earlier version of this function returned on the FIRST "ODG:"
    match (frame 1's single-frame, unconverged value), which is dominated by
    onset transients/silence and is not representative of the clip. Table I
    must be regenerated after this fix; the old checkpoint's odg_* values
    are not whole-clip scores.
    """
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

# --- 9. Unified Evaluation Pipeline ---
print("\n--- Starting SPL Table I Evaluation (IPC vs SS vs WavMark) ---")

# Mount Google Drive first
from google.colab import drive
drive.mount('/content/drive')

state = load_checkpoint()
processed = set(state["processed"])
if len(processed) >= len(df_sample):
    print(f"Already processed {len(processed)} clips. Target reached.")
else:
    if processed:
        print(f"Resuming from checkpoint: {len(processed)} clips already done.")

    # Get list of available files on Drive once
    drive_files = list_drive_files()

    for _, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
        filename = row["filename"]
        if filename in processed:
            continue

        generator_id = row["generator"]
        local_orig = os.path.join(LOCAL_WORKSPACE, filename)

        if filename not in drive_files:
            print(f"\n[FATAL] Required clip '{filename}' was not found on Drive "
                  f"under '{PROJECT_DIR}'.")
            print("The 249-clip stratified sample (83 per generator) is fixed "
                  "by this script's sampling code; a missing clip means Table I "
                  "would be computed over an incomplete and unbalanced sample.")
            print("Run spl00_data_generation.py first to generate the full "
                  "required corpus for this paper, then re-run this script.")
            sys.exit(1)

        if not load_from_drive(filename, local_orig):
            print(f"\n[FATAL] Could not download clip '{filename}' from Drive.")
            sys.exit(1)

        y_orig, _ = librosa.load(local_orig, sr=SR)
        os.remove(local_orig)

        # --- IPC ---
        payload_ipc = generate_watermark(generator_id)
        f0_track, ref_phase = extract_reference_phase_ipc(y_orig, SR, H_STARS)
        y_ipc = embed_watermark_ipc(y_orig, SR, H_STARS, B_STARS, payload_ipc)

        rec_ipc_pos = detect_watermark_ipc(y_ipc, SR, H_STARS, B_STARS, f0_track, ref_phase)
        rec_ipc_neg = detect_watermark_ipc(y_orig, SR, H_STARS, B_STARS, f0_track, ref_phase)
        ber_ipc_pos = compute_ber(payload_ipc, rec_ipc_pos, B_STARS)
        ber_ipc_neg = compute_ber(payload_ipc, rec_ipc_neg, B_STARS)

        # --- Spread Spectrum ---
        payload_ss = get_payload_ss(generator_id)
        y_ss, n_frames = embed_spread_spectrum(y_orig, payload_ss)

        rec_ss_pos = detect_spread_spectrum(y_ss, n_frames)
        rec_ss_neg = detect_spread_spectrum(y_orig, n_frames)
        ber_ss_pos = compute_ber(payload_ss, rec_ss_pos, B_STARS)
        ber_ss_neg = compute_ber(payload_ss, rec_ss_neg, B_STARS)

        # --- WavMark ---
        y_wm, _ = encode_wavmark_wrapper(y_orig, generator_id)
        rec_wm_pos = decode_wavmark_wrapper(y_wm)
        rec_wm_neg = decode_wavmark_wrapper(y_orig)
        ber_wm_pos = compute_ber_wavmark(generator_id, rec_wm_pos)
        ber_wm_neg = compute_ber_wavmark(generator_id, rec_wm_neg)

        # --- PEAQ ODG (same 48k-stereo pipeline for all three) ---
        tmp_orig = os.path.join(LOCAL_WORKSPACE, "tmp_orig.wav")
        tmp_ipc = os.path.join(LOCAL_WORKSPACE, "tmp_ipc.wav")
        tmp_ss = os.path.join(LOCAL_WORKSPACE, "tmp_ss.wav")
        tmp_wm = os.path.join(LOCAL_WORKSPACE, "tmp_wm.wav")

        write_peaq_wav(y_orig, tmp_orig)
        write_peaq_wav(y_ipc, tmp_ipc)
        write_peaq_wav(y_ss, tmp_ss)
        write_peaq_wav(y_wm, tmp_wm)

        odg_ipc = calculate_peaq_odg(tmp_orig, tmp_ipc)
        odg_ss = calculate_peaq_odg(tmp_orig, tmp_ss)
        odg_wm = calculate_peaq_odg(tmp_orig, tmp_wm)

        for p in [tmp_orig, tmp_ipc, tmp_ss, tmp_wm]:
            if os.path.exists(p):
                os.remove(p)

        state["results"].append({
            "filename": filename,
            "generator": generator_id,
            "odg_ipc": odg_ipc, "odg_ss": odg_ss, "odg_wm": odg_wm,
            "ber_ipc_pos": ber_ipc_pos, "ber_ipc_neg": ber_ipc_neg,
            "ber_ss_pos": ber_ss_pos, "ber_ss_neg": ber_ss_neg,
            "ber_wm_pos": ber_wm_pos, "ber_wm_neg": ber_wm_neg,
        })
        state["processed"].append(filename)
        save_checkpoint(state)

# --- 10. Aggregation: Table I ---
print("\n--- Aggregating Table I ---")
df = pd.DataFrame(state["results"])

local_results = os.path.join(LOCAL_WORKSPACE, RESULTS_FILE)
df.to_json(local_results, orient="records", indent=4)
save_to_drive(local_results, RESULTS_FILE)

odg_stats = {}
auc_results = {}
for method, odg_col, pos_col, neg_col in [
    ("IPC (proposed)", "odg_ipc", "ber_ipc_pos", "ber_ipc_neg"),
    ("Spread Spectrum", "odg_ss", "ber_ss_pos", "ber_ss_neg"),
    ("WavMark (Neural)", "odg_wm", "ber_wm_pos", "ber_wm_neg"),
]:
    odg_stats[method] = {
        "mean": df[odg_col].mean(),
        "std": df[odg_col].std(),
        "median": df[odg_col].median(),
    }
    y_true = np.concatenate([np.ones(len(df)), np.zeros(len(df))])
    y_scores = np.concatenate([-df[pos_col].values, -df[neg_col].values])
    auc_results[method] = roc_auc_score(y_true, y_scores)

table1_tex = (
    "\\begin{table}[t]\n"
    "\\centering\n"
    "\\caption{Imperceptibility and Clean-Channel Detection}\n"
    "\\label{tab:baseline}\n"
    "\\begin{tabular}{lccc}\n"
    "\\toprule\n"
    "Method & Mean ODG & Median ODG & ROC AUC \\\\\n"
    "\\midrule\n"
)
for method in ["IPC (proposed)", "Spread Spectrum", "WavMark (Neural)"]:
    s = odg_stats[method]
    table1_tex += (
        f"{method} & ${s['mean']:.3f}\\pm{s['std']:.3f}$ & "
        f"${s['median']:.3f}$ & {auc_results[method]:.4f} \\\\\n"
    )
table1_tex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"

print(table1_tex)
local_table = os.path.join(LOCAL_WORKSPACE, TABLE_FILE)
with open(local_table, "w") as f:
    f.write(table1_tex)
save_to_drive(local_table, TABLE_FILE)

# --- Sync to ensure all writes are flushed ---
print("\n[SYNC] Flushing file system buffers...")
os.sync()
print("[SYNC] Complete.")

print(f"\n[SUCCESS] Table I complete. Results: {RESULTS_FILE}, LaTeX: {TABLE_FILE} (saved to Drive).")