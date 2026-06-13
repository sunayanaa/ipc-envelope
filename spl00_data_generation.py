# ==============================================================================
# Program Name: spl00_data_generation.py
# Version: 1.1
# Description: Stage 0 of the reproducibility pipeline for "Phase-Coherent Audio Watermarking Has a Near-Zero Synchronisation Tolerance". Generates the audio corpus used by
#              spl01_baseline_comparison.py (Table I) and
#              spl02_sync_sensitivity.py (Table II / Fig. 2).
#
#              200 text prompts are sampled from MusicCaps
#              (df.sample(n=200, random_state=42)) and used to generate 431
#              10-second clips across three generative models:
#                - MusicGen-medium:   265 clips (all 200 prompts at variation
#                                     0, plus a stratified subset at
#                                     variations 1-4)
#                - AudioLDM-2:         83 clips (72 unique prompts)
#                - Stable Audio Open:  83 clips (68 unique prompts)
#              The exact (prompt, variation) pairs per generator are computed
#              deterministically at the top of this script via
#              groupby(...).sample(n=83, random_state=42), so the corpus is
#              fully reproducible from this script alone. All clips are
#              saved to a dedicated Google Drive folder ("ipc-envelope/").
#
# Change Log: 1.1 - HuggingFace login runs once at the start, before any model
#                   download, since unauthenticated Hub requests are rate-
#                   limited; this speeds up the MusicGen-medium and
#                   AudioLDM-2 weight downloads as well as granting access to
#                   the gated Stable Audio Open checkpoint.
#             1.0 - Initial version. HF token is requested interactively
#                   rather than hardcoded.
# GPU Required: YES (T4 minimum, A100 recommended for speed)
# ==============================================================================

!pip install -q torch torchvision torchaudio torchsde transformers diffusers accelerate scipy soundfile pandas openpyxl

import sys
import os
import gc
import json
import shutil
import zipfile
import torch
import pandas as pd
import soundfile as sf
from google.colab import drive

# --- 1. GPU Check ---
if not torch.cuda.is_available():
    print("\n[ERROR] GPU not detected!")
    print("This script requires a GPU.")
    print("Please switch your Colab runtime to a T4 GPU and restart.")
    sys.exit(1)
print("CUDA available: True. Proceeding...")

# --- 1b. HuggingFace Authentication (applies to ALL downloads below) ---
# Logging in once at the start, not just for Stable Audio Open's gated
# checkpoint, matters for MusicGen-medium and AudioLDM-2 too: unauthenticated
# Hub requests are subject to a much lower rate limit, so an authenticated
# session noticeably speeds up the (large) model-weight downloads for all
# three generators, not only the gated one.
#
# NOTE: do not hardcode tokens in this script; it is requested interactively
# below (or read from the HF_TOKEN environment variable).
from huggingface_hub import login
from getpass import getpass
hf_token = os.environ.get("HF_TOKEN") or getpass(
    "Enter your HuggingFace token (also needed for gated "
    "stabilityai/stable-audio-open-1.0 access): "
)
login(token=hf_token)

# --- 2. Configuration ---
PROJECT_DIR = "/content/drive/MyDrive/ipc-envelope/"  # Persistent storage
DRIVE_DIR = "/content/drive/MyDrive/datasets"
MUSICCAPS_ZIP = os.path.join(DRIVE_DIR, "MusicCaps.zip")
LOCAL_TEMP_DIR = "/content/temp_data"

TARGET_PROMPTS = 200
VARIATIONS_PER_PROMPT = 5
DURATION_SEC = 10

CHECKPOINT_FILE = "spl00_generation_checkpoint.json"

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
    local_cp = os.path.join(LOCAL_TEMP_DIR, "temp_checkpoint.json")
    if load_from_drive(CHECKPOINT_FILE, local_cp):
        with open(local_cp, "r") as f:
            return json.load(f)
    return {"completed": {}}

def save_checkpoint(state):
    """Save checkpoint to Google Drive project folder."""
    local_cp = os.path.join(LOCAL_TEMP_DIR, "temp_checkpoint.json")
    with open(local_cp, "w") as f:
        json.dump(state, f)
    save_to_drive(local_cp, CHECKPOINT_FILE)

# --- 4. MusicCaps Prompt Sampling (deterministic) ---
print("\n--- Loading MusicCaps prompts (n=200, random_state=42) ---")
drive.mount('/content/drive')
os.makedirs(LOCAL_TEMP_DIR, exist_ok=True)

with zipfile.ZipFile(MUSICCAPS_ZIP, 'r') as zip_ref:
    zip_ref.extract("musiccaps-public.csv", LOCAL_TEMP_DIR)

df_mc = pd.read_csv(os.path.join(LOCAL_TEMP_DIR, "musiccaps-public.csv"))
sampled_df = df_mc.sample(n=TARGET_PROMPTS, random_state=42)
prompts = sampled_df['caption'].tolist()
print(f"Loaded {len(prompts)} prompts. prompts[0][:80] = {prompts[0][:80]!r}")

# --- 4b. Export the Prompt List (for submission / provenance) ---
# prompt_idx (p000-p199) is the index used throughout this script and in
# spl01/spl02's filenames (gen_{generator}_p{i:03d}_v{var}.wav). ytid/start_s/
# end_s identify the source MusicCaps clip the caption was drawn from; caption
# is the literal text prompt passed to each generative model.
PROMPT_EXPORT_COLS = ["ytid", "start_s", "end_s", "caption"]
available_cols = [c for c in PROMPT_EXPORT_COLS if c in sampled_df.columns]

df_prompts = sampled_df[available_cols].reset_index(drop=True)
df_prompts.insert(0, "prompt_idx", [f"p{i:03d}" for i in range(len(df_prompts))])

PROMPTS_CSV = "musiccaps_prompts_used.csv"
PROMPTS_XLSX = "musiccaps_prompts_used.xlsx"
local_prompts_csv = os.path.join(LOCAL_TEMP_DIR, PROMPTS_CSV)
local_prompts_xlsx = os.path.join(LOCAL_TEMP_DIR, PROMPTS_XLSX)

df_prompts.to_csv(local_prompts_csv, index=False)
df_prompts.to_excel(local_prompts_xlsx, index=False)

save_to_drive(local_prompts_csv, PROMPTS_CSV)
save_to_drive(local_prompts_xlsx, PROMPTS_XLSX)

print(f"Saved {len(df_prompts)} prompts (prompt_idx + {available_cols}) to "
      f"{PROMPTS_CSV} / {PROMPTS_XLSX} (saved to Drive:{PROJECT_DIR}).")

# --- 5. Required (prompt_idx, variation) Pairs per Generator ---
# Reproduces spl01's 249-clip stratified sample (83/generator, sorted filename
# list + groupby(...).sample(n=83, random_state=42)) and unions it with
# spl02's requirement (musicgen, all 200 prompts, variation 0).
print("\n--- Computing required clip list (spl01 union spl02) ---")
GENERATORS = ["musicgen", "audioldm2", "stableaudio"]

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

required = {gen: set() for gen in GENERATORS}
for _, row in df_sample.iterrows():
    required[row["generator"]].add((row["prompt_idx"], row["variation"]))

# spl02: all 200 MusicGen prompts at variation 0
required["musicgen"] |= {(i, 0) for i in range(TARGET_PROMPTS)}

for gen in GENERATORS:
    required[gen] = sorted(required[gen])
    print(f"  {gen}: {len(required[gen])} clips "
          f"({len(set(p for p, _ in required[gen]))} unique prompts)")

TOTAL_CLIPS = sum(len(v) for v in required.values())
print(f"  TOTAL: {TOTAL_CLIPS} clips")

# --- Helper to clear VRAM ---
def flush_vram():
    gc.collect()
    torch.cuda.empty_cache()

state = load_checkpoint()

# ==============================================================================
# A. MusicGen
# ==============================================================================
gen_id = "musicgen"
done = set(tuple(x) for x in state["completed"].get(gen_id, []))
todo = [(i, v) for (i, v) in required[gen_id] if (i, v) not in done]

if todo:
    print(f"\n--- Starting MusicGen Generation ({len(todo)} clips remaining) ---")
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    processor = AutoProcessor.from_pretrained("facebook/musicgen-medium")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-medium").to("cuda")

    for (i, var) in todo:
        prompt = prompts[i]
        print(f"MusicGen: prompt p{i:03d}, variation v{var}")

        inputs = processor(text=[prompt], padding=True, return_tensors="pt").to("cuda")
        # 512 tokens corresponds to roughly 10 seconds for MusicGen
        audio_values = model.generate(**inputs, max_new_tokens=512)

        audio_data = audio_values[0, 0].cpu().numpy()
        sample_rate = model.config.audio_encoder.sampling_rate

        filename = f"gen_{gen_id}_p{i:03d}_v{var}.wav"
        local_path = os.path.join(LOCAL_TEMP_DIR, filename)
        sf.write(local_path, audio_data, sample_rate)

        save_to_drive(local_path, filename)
        os.remove(local_path)

        state["completed"].setdefault(gen_id, []).append([i, var])
        save_checkpoint(state)

    del model
    del processor
    flush_vram()
else:
    print(f"\n--- MusicGen: all {len(required[gen_id])} required clips already done ---")

# ==============================================================================
# B. AudioLDM-2
# ==============================================================================
gen_id = "audioldm2"
done = set(tuple(x) for x in state["completed"].get(gen_id, []))
todo = [(i, v) for (i, v) in required[gen_id] if (i, v) not in done]

if todo:
    print(f"\n--- Starting AudioLDM-2 Generation ({len(todo)} clips remaining) ---")
    from diffusers import AudioLDM2Pipeline
    from transformers import AutoModelForCausalLM

    correct_lm = AutoModelForCausalLM.from_pretrained(
        "cvssp/audioldm2",
        subfolder="language_model",
        torch_dtype=torch.float16
    )
    pipe = AudioLDM2Pipeline.from_pretrained(
        "cvssp/audioldm2",
        language_model=correct_lm,
        torch_dtype=torch.float16
    ).to("cuda")

    for (i, var) in todo:
        prompt = prompts[i]
        print(f"AudioLDM-2: prompt p{i:03d}, variation v{var}")

        audio = pipe(prompt, num_inference_steps=200, audio_length_in_s=DURATION_SEC).audios[0]

        filename = f"gen_{gen_id}_p{i:03d}_v{var}.wav"
        local_path = os.path.join(LOCAL_TEMP_DIR, filename)
        sf.write(local_path, audio, 16000)

        save_to_drive(local_path, filename)
        os.remove(local_path)

        state["completed"].setdefault(gen_id, []).append([i, var])
        save_checkpoint(state)

    del pipe
    del correct_lm
    flush_vram()
else:
    print(f"\n--- AudioLDM-2: all {len(required[gen_id])} required clips already done ---")

# ==============================================================================
# C. Stable Audio Open
# ==============================================================================
gen_id = "stableaudio"
done = set(tuple(x) for x in state["completed"].get(gen_id, []))
todo = [(i, v) for (i, v) in required[gen_id] if (i, v) not in done]

if todo:
    print(f"\n--- Starting Stable Audio Open Generation ({len(todo)} clips remaining) ---")

    from diffusers import StableAudioPipeline

    pipe = StableAudioPipeline.from_pretrained(
        "stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16
    ).to("cuda")

    for (i, var) in todo:
        prompt = prompts[i]
        print(f"StableAudio: prompt p{i:03d}, variation v{var}")

        audio = pipe(prompt, num_inference_steps=100, audio_end_in_s=DURATION_SEC).audios[0]

        filename = f"gen_{gen_id}_p{i:03d}_v{var}.wav"
        local_path = os.path.join(LOCAL_TEMP_DIR, filename)
        sf.write(local_path, audio.cpu().to(torch.float32).numpy().T, 44100)

        save_to_drive(local_path, filename)
        os.remove(local_path)

        state["completed"].setdefault(gen_id, []).append([i, var])
        save_checkpoint(state)

    del pipe
    flush_vram()
else:
    print(f"\n--- Stable Audio Open: all {len(required[gen_id])} required clips already done ---")

# --- Sync to ensure all writes are flushed ---
print("\n[SYNC] Flushing file system buffers...")
os.sync()
print("[SYNC] Complete.")

print("\n[SUCCESS] All required clips generated and saved to "
      f"Drive:{PROJECT_DIR}. spl01_baseline_comparison.py and "
      "spl02_sync_sensitivity.py can now be run against this corpus.")