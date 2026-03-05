from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
import numpy as np
from decord import VideoReader, cpu
from tqdm import tqdm
import logging

SPLIT = "train"
SOURCE_ROOT = Path(f"NEW_{SPLIT}")
DEST_ROOT = Path(f"NEW_{SPLIT}_frames")
MAX_WORKERS = 18
SHORT_SIDE = 240
MAX_FPS = 30
JPEG_QUALITY = 95
BATCH_SIZE = 100
LOG_FILE = f"skipped_{SPLIT}_videos.txt"

# -------------------- LOGGER SETUP --------------------
logging.basicConfig(
    filename=LOG_FILE,
    filemode="a",
    level=logging.ERROR,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)
# ------------------------------------------------------


def resize_frame(frame, short_side=SHORT_SIDE):
    """Resize frame keeping aspect ratio so shortest side = short_side."""
    h, w = frame.shape[:2]
    if h < w:
        new_h = short_side
        new_w = int(w * short_side / h)
    else:
        new_w = short_side
        new_h = int(h * short_side / w)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def extract_frames(video_path, output_dir):
    """Extract frames - skips entire video if any frame fails."""
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    except Exception as e:
        logger.error(f"Skipped {video_path}: Failed to open video - {e}")
        return

    try:
        fps = vr.get_avg_fps()
        total_frames = len(vr)
    except Exception as e:
        logger.error(f"Skipped {video_path}: Failed to get video info - {e}")
        return

    try:
        if fps > MAX_FPS:
            target_frame_count = int(total_frames * MAX_FPS / fps)
            indices = np.linspace(
                0, total_frames - 1, target_frame_count, dtype=int
            ).tolist()
        else:
            indices = list(range(total_frames))
    except Exception as e:
        logger.error(f"Skipped {video_path}: Failed to calculate frame indices - {e}")
        return

    for batch_start in range(0, len(indices), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(indices))
        batch_indices = indices[batch_start:batch_end]

        try:
            frames = vr.get_batch(batch_indices).asnumpy()
        except Exception as e:
            logger.error(
                f"Skipped {video_path}: Failed to decode batch {batch_start}-{batch_end} - {e}"
            )
            return

        for i, frame in enumerate(frames):
            frame_idx = batch_start + i

            try:
                resized = resize_frame(frame)
                bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)

                out_path = str(output_dir / f"{frame_idx+1:05d}.jpg")
                success = cv2.imwrite(
                    out_path, bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                )

                if not success:
                    logger.error(
                        f"Skipped {video_path}: Failed to save frame {frame_idx+1}"
                    )
                    return

            except Exception as e:
                logger.error(
                    f"Skipped {video_path}: Failed to process frame {frame_idx+1} - {e}"
                )
                return


def main():
    tasks = []

    for class_dir in sorted(SOURCE_ROOT.iterdir()):
        if not class_dir.is_dir():
            continue

        videos = list(class_dir.glob("*.mp4"))
        if not videos:
            continue

        for vid in sorted(videos):
            output_dir = DEST_ROOT / class_dir.name / vid.stem
            tasks.append((vid, output_dir))

    print(f"Processing {len(tasks)} videos with {MAX_WORKERS} workers...")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(extract_frames, vid, out_dir): vid
            for vid, out_dir in tasks
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting"):
            vid = futures[future]
            try:
                future.result()
            except Exception as e:
                logger.error(f"Skipped {vid}: Unexpected error - {e}")

    print("Done.")


if __name__ == "__main__":
    main()