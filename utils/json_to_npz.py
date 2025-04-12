import os
import json
import numpy as np
from tqdm import tqdm
import logging
from multiprocessing import Pool, cpu_count

# === CONFIGURATION ===
EXPECTED_KEYPOINTS = 299  # 84 hands (42 left + 42 right) + 140 face + 75 body
RAW_DIR = "data/raw/Validation/json"
SAVE_DIR = "data/processed/Val"
LOG_FILE = "keypoint_extraction.log"

# === SETUP LOGGING ===
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

def load_keypoints(file_path):
    try:
        data = np.load(file_path)
        return data['keypoints']  # shape: (T, 299)
    except Exception as e:
        print(f"[ERROR] Failed to load {file_path}: {e}")
        return None  # Let Dataset handle None case

# === EXTRACT KEYPOINTS FROM ONE FRAME ===
def extract_keypoints_from_frame(json_file):
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)

        if not data["people"]:
            logging.warning(f"No people detected in {json_file}")
            return [0.0] * EXPECTED_KEYPOINTS  # blank frame

        person = data["people"][0]

        keypoints = []
        # Hand keypoints: 42 (left) + 42 (right) = 84
        keypoints += person.get("hand_left_keypoints_2d", [0.0] * 63)[:42]   # x, y for 21 keypoints
        keypoints += person.get("hand_right_keypoints_2d", [0.0] * 63)[:42]  # x, y for 21 keypoints
        # Face keypoints: 140 (x, y for 70 keypoints)
        keypoints += person.get("face_keypoints_2d", [0.0] * 210)[:140]      # x, y for 70 keypoints
        # Body keypoints: 75 (x, y, confidence for 25 keypoints)
        keypoints += person.get("pose_keypoints_2d", [0.0] * 75)[:75]        # x, y, confidence for 25 keypoints

        # Log keypoint count for debugging
        logging.info(f"Extracted {len(keypoints)} keypoints from {json_file}")

        # Pad if needed (handles cases where keypoints are fewer than expected)
        if len(keypoints) < EXPECTED_KEYPOINTS:
            logging.warning(f"Padding {json_file}: got {len(keypoints)}, expected {EXPECTED_KEYPOINTS}")
            keypoints += [0.0] * (EXPECTED_KEYPOINTS - len(keypoints))
        elif len(keypoints) > EXPECTED_KEYPOINTS:
            logging.warning(f"Truncating {json_file}: got {len(keypoints)}, expected {EXPECTED_KEYPOINTS}")
            keypoints = keypoints[:EXPECTED_KEYPOINTS]

        return keypoints
    except Exception as e:
        logging.error(f"Failed to process {json_file}: {e}")
        return [0.0] * EXPECTED_KEYPOINTS

# === PROCESS A SINGLE VIDEO FOLDER ===
def process_single_folder(video_folder):
    video_path = os.path.join(RAW_DIR, video_folder)
    if not os.path.isdir(video_path):
        logging.warning(f"Skipping {video_path}: not a directory")
        return

    try:
        frame_files = sorted([f for f in os.listdir(video_path) if f.endswith('.json')])
        keypoints_seq = []

        for frame_file in frame_files:
            json_path = os.path.join(video_path, frame_file)
            keypoints = extract_keypoints_from_frame(json_path)
            keypoints_seq.append(keypoints)

        keypoints_array = np.array(keypoints_seq)
        npz_path = os.path.join(SAVE_DIR, f"{video_folder}.npz")
        np.savez_compressed(npz_path, keypoints=keypoints_array)
        logging.info(f"✅ Processed: {video_folder} ({len(frame_files)} frames, shape: {keypoints_array.shape})")
    except Exception as e:
        logging.error(f"[ERROR] {video_folder}: {e}")

# === MAIN FUNCTION WITH MULTIPROCESSING ===
def convert_all_video_folders():
    os.makedirs(SAVE_DIR, exist_ok=True)
    video_folders = [f for f in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, f))]
    logging.info(f"Found {len(video_folders)} video folders to process")

    with Pool(cpu_count()) as pool:
        list(tqdm(pool.imap_unordered(process_single_folder, video_folders), total=len(video_folders)))

if __name__ == "__main__":
    convert_all_video_folders()
    print(f"\n✅ All done! See log: {LOG_FILE}")
    logging.info("All done!")
    logging.shutdown()