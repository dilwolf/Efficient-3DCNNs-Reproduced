"""
Kinetics LMDB Dataset Cache Builder
Pre-builds dataset_cache.pkl for each split (train, valid) using multiprocessing
to scan per-video LMDB files in parallel.
"""
import pickle
import lmdb
import struct
import logging
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


# Logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("build_cache.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)



# Mirrors _read_lmdb_len() from Kinetics dataloader

def _read_lmdb_len(lmdb_path: str) -> int:
    """Open LMDB briefly just to read __len__, then close."""
    env = lmdb.open(lmdb_path, readonly=True, lock=False, meminit=False)
    with env.begin(write=False) as txn:
        raw = txn.get(b"__len__")
        if raw is None:
            raise IOError(f"Missing __len__ key in {lmdb_path}")
        n = struct.unpack(">I", raw)[0]
    env.close()
    return n



# Worker: scan one class directory → return list of samples

def scan_class_dir(args):
    """
    Worker function: scans all .lmdb files in one class folder.

    Args:
        args: tuple of (class_dir: Path, class_idx: int, sample_duration: int, sampling_step: int)

    Returns:
        (class_name, samples, error_message)
        samples: list of (lmdb_path_str, n_frames, class_idx)
    """
    class_dir, class_idx, sample_duration, sampling_step = args
    samples = []

    try:
        lmdb_files = sorted(class_dir.glob("*.lmdb"))
        effective_length = sample_duration * sampling_step

        for lmdb_path in lmdb_files:
            try:
                n_frames = _read_lmdb_len(str(lmdb_path))

                if n_frames < effective_length:
                    continue  # mirrors _process_video_lmdb() skip logic

                samples.append((str(lmdb_path), n_frames, class_idx))

            except IOError as e:
                logger.warning(f"Skipped {lmdb_path.name}: {e}")
                continue

        return (class_dir.name, samples, None)

    except Exception as e:
        return (class_dir.name, [], str(e))



# Build cache for one split

def build_cache_for_split(
    split_path: Path,
    num_classes: int,
    sample_duration: int,
    sampling_step: int,
    num_workers: int
):
    logger.info(f"Building cache for split: {split_path}")

    if not split_path.exists():
        raise FileNotFoundError(f"Split path not found: {split_path}")

    cache_path = split_path / "dataset_cache.pkl"
    if cache_path.exists():
        logger.info(f"Cache already exists, skipping: {cache_path}")
        return

    #  Get class directories
    class_dirs = sorted([d for d in split_path.iterdir() if d.is_dir()])

    if not class_dirs:
        raise ValueError(f"No class directories found in {split_path}")

    if len(class_dirs) != num_classes:
        raise ValueError(
            f"Found {len(class_dirs)} class folders, expected {num_classes}"
        )

    #  Build class mappings
    class_to_idx = {cls_dir.name: idx for idx, cls_dir in enumerate(class_dirs)}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    #  Scan class dirs in parallel 
    jobs = [
        (class_dir, class_to_idx[class_dir.name], sample_duration, sampling_step)
        for class_dir in class_dirs
    ]

    all_samples = []
    errors = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(scan_class_dir, job): job for job in jobs}

        for future in tqdm(as_completed(futures), total=len(jobs), desc=f"[{split_path.name}]"):
            job = futures[future]
            try:
                class_name, samples, error = future.result()
                if error:
                    errors.append((class_name, error))
                    logger.error(f"ERROR | {class_name} | {error}")
                else:
                    all_samples.extend(samples)
            except Exception as e:
                errors.append((str(job[0].name), str(e)))
                logger.error(f"CRASH | {job[0].name} | {e}")

    if not all_samples:
        raise ValueError(f"No valid samples found in {split_path}")

    #  Save cache — same structure as _build_dataset()
    with open(cache_path, 'wb') as f:
        pickle.dump({
            'samples':      all_samples,
            'class_to_idx': class_to_idx,
            'idx_to_class': idx_to_class,
        }, f)

    #  Summary
    logger.info("=" * 60)
    logger.info(f"Cache saved       : {cache_path}")
    logger.info(f"  Total samples   : {len(all_samples)}")
    logger.info(f"  Total classes   : {len(class_to_idx)}")
    if errors:
        logger.info(f"  Errors          : {len(errors)}")
        for cls, msg in errors:
            logger.info(f"    - {cls}: {msg}")
    logger.info("=" * 60)



# Entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-build Kinetics LMDB dataset cache files")
    parser.add_argument("--root",            type=str, default="/mnt/HDD10TB/kinetics600_lmdb", help="Root path to Kinetics LMDB dataset")
    parser.add_argument("--num_classes",     type=int, default=600,  help="Expected number of classes")
    parser.add_argument("--sample_duration", type=int, default=16,   help="Number of frames per clip (must match dataloader)")
    parser.add_argument("--sampling_step",   type=int, default=1,    help="Stride between frames (must match dataloader)")
    parser.add_argument("--workers",         type=int, default=4,    help="Number of parallel worker processes")
    args = parser.parse_args()

    root = Path(args.root)

    for split in ["train", "valid"]:
        build_cache_for_split(
            split_path      = root / split,
            num_classes     = args.num_classes,
            sample_duration = args.sample_duration,
            sampling_step   = args.sampling_step,
            num_workers     = args.workers,
        )
    logger.info("All splits done.")