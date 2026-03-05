"""
Efficient frame-based Kinetics dataset loader
"""

import os
import cv2
import torch
import random
import logging
import numpy as np
from pathlib import Path
import torch.utils.data as data
from typing import Tuple, Optional

# Setup logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


class Kinetics(data.Dataset):
    """
    Modern PyTorch Dataset for video classification from pre-extracted frames.
    Compatible with PyTorch 2.5+ and optimized for efficiency.
    
    Args:
        root_path: Root directory containing class folders
        input_size: model input size
        subset: 'train', 'valid'
        sample_duration: Number of frames per clip
        sampling_step: Stride between frames (for temporal subsampling)
        transform: Transforms to apply to each frame
        num_classes: Number of classes
    """
    
    def __init__(
        self,
        root_path: str,
        input_size: int = 224,
        mode: str = 'train',
        sample_duration: int = 16,
        sampling_step: int = 1,
        num_classes: int = 600,
        transform = None
):
        self.transform = transform or VideoAugmentation(mode=mode, input_size=input_size)
        self.path = Path(root_path) / Path(mode)
        self.mode = mode
        self.sample_duration = sample_duration
        self.sampling_step = sampling_step
        self.num_classes = num_classes
        
        # Build dataset
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        self._build_dataset()
        
        # logger.info(f"Dataset initialized: {len(self.samples)} samples from {len(self.class_to_idx)} classes")
    
    def _build_dataset(self):
        """Build dataset by scanning directory structure"""
        
        if not self.path.exists():
            raise FileNotFoundError(f"Root path does not exist: {self.path}")
        
        # Get all class directories
        class_dirs = sorted([d for d in self.path.iterdir() if d.is_dir()])
        
        if not class_dirs:
            raise ValueError(f"No class directories found in {self.path}")
        
        if len(class_dirs) != self.num_classes:
            raise ValueError(f"The number of class folders: {len(class_dirs)} is not equal to expected number of classes: {self.num_classes}")
        
        # Create class mappings
        self.class_to_idx = {cls_dir.name: idx for idx, cls_dir in enumerate(class_dirs)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
        
        # Scan each class directory
        for class_dir in class_dirs:
            class_name = class_dir.name
            class_idx = self.class_to_idx[class_name]
            
            # Get all video directories within this class
            video_dirs = sorted([v for v in class_dir.iterdir() if v.is_dir()])
            
            for video_dir in video_dirs:
                self._process_video_directory(video_dir, class_idx)
        
        if not self.samples:
            raise ValueError(f"No valid samples found in {self.path}")
        
    def _get_frame_range(self, video_dir: str):
        min_idx = None
        max_idx = None

        with os.scandir(video_dir) as it:
            for entry in it:
                if entry.is_file() and entry.name.endswith(".jpg"):
                    num = int(entry.name[:-4])

                    if min_idx is None:
                        min_idx = num
                        max_idx = num
                    else:
                        if num < min_idx:
                            min_idx = num
                        elif num > max_idx:
                            max_idx = num

        if min_idx is None:
            raise IOError(f"No frames found in {video_dir}")

        return min_idx, max_idx
    
    def load_frames(self, frame_paths: list[str]) -> list[np.ndarray]:
        """
        Load multiple frames efficiently using OpenCV
        
        Args:
            frame_paths: List of frame paths to load
            
        Returns:
            List of numpy arrays (RGB format)
        """
        frames = []
        
        for path in frame_paths:
            frame = cv2.imread(path)

            if frame is None:
                # logger.error(f"Image not found or unreadable: {path}")
                raise IOError(f"Image not found or unreadable: {path}") # we need to catch and stop if the frame is not found

            # convert to RGB color for augmentation later
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    
        return frames
    
    def _process_video_directory(self, video_dir: Path, class_idx: int):
        """Process a single video directory and create samples"""
        
        # Get all frame paths
        start_idx, end_idx = self._get_frame_range(video_dir)
        n_frames = end_idx - start_idx + 1
        effective_length = self.sample_duration * self.sampling_step
        if n_frames < effective_length: # Skip if total frames is smaller than effective_length
            return
        
        # Create samples based on sampling strategy
        self.samples.append(
            (str(video_dir), start_idx, end_idx, n_frames, class_idx)
        )
    
    def _sample_paths(self, video_dir: str, n_frames: int, start_idx: int):

        effective_length = self.sample_duration * self.sampling_step
        max_start = n_frames - effective_length

        if self.mode == "train":
            offset = random.randint(0, max_start)
        else:
            offset = max_start // 2

        start_frame = start_idx + offset

        return [
            os.path.join(video_dir, f"{start_frame + i * self.sampling_step:05d}.jpg")
            for i in range(self.sample_duration)
        ]
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Get a video clip and its label
        
        Args:
            index: Sample index
            
        Returns:
            Tuple of (clip_tensor, label)
            clip_tensor shape: (C, T, H, W)
        """
        video_dir, start_idx, _, n_frames, label = self.samples[index]
        frame_paths_to_load = self._sample_paths(video_dir, n_frames, start_idx)
        frames = self.load_frames(frame_paths_to_load)
        
        # Randomize parameters once per clip for temporal consistency
        if hasattr(self.transform, 'randomize_parameters') and self.mode == 'train':
            self.transform.randomize_parameters()
        
        frames = self.transform(frames)
        frames = [torch.from_numpy(np.ascontiguousarray(f)).permute(2, 0, 1) for f in frames]
        
        clip = torch.stack(frames, dim=0).permute(1, 0, 2, 3)
        
        return clip, label
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset"""
        return len(self.samples)


class VideoAugmentation:
    """
    Video augmentation class using pure OpenCV for efficiency.
    Handles both spatial and temporal consistency across video frames.
    
    Args:
        mode: 'train' or 'valid'
        input_size: Target size for output frames (default: 224)
        scales: List of crop scales for multiscale cropping (default: [1.0, 0.84, 0.71, 0.59, 0.5])
        crop_type: 'random', 'corner', or 'center' (default: 'random' for train, 'center' for val)
        mean: Mean values for normalization (RGB order)
        std: Std values for normalization (RGB order)
        horizontal_flip: Whether to apply random horizontal flip (default: True)
        color_jitter: Whether to apply color jitter (default: True for train, False for val)
        brightness: Brightness jitter range (default: 0.4)
        contrast: Contrast jitter range (default: 0.4)
        saturation: Saturation jitter range (default: 0.4)
        hue: Hue jitter range in degrees (default: 10)
    """
    
    def __init__(
        self,
        mode: str = 'train',
        input_size: int = 224,
        crop_type: Optional[str] = None,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        horizontal_flip: bool = True,
        color_jitter: bool = None,
        brightness: float = 0.4,
        contrast: float = 0.4,
        saturation: float = 0.4,
        hue: float = 10.0
    ):
        self.mode = mode
        self.input_size = input_size
        self.scales = [1.0, 0.84, 0.71, 0.59, 0.5]
        
        # Set defaults based on mode
        if crop_type is None:
            self.crop_type = 'random' if mode == 'train' else 'center'
        else:
            self.crop_type = crop_type
            
        if color_jitter is None:
            self.color_jitter = True if mode == 'train' else False
        else:
            self.color_jitter = color_jitter
        
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.horizontal_flip = horizontal_flip and (mode == 'train')
        
        # Color jitter parameters
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        
        # Parameters that will be randomized once per clip
        self.scale = None
        self.crop_position = None
        self.flip = None
        self.brightness_factor = None
        self.contrast_factor = None
        self.saturation_factor = None
        self.hue_factor = None
    
    def randomize_parameters(self):
        """Randomize augmentation parameters once per video clip"""
        
        # Random scale
        self.scale = random.choice(self.scales)
        
        # Random crop position
        if self.crop_type == 'random':
            self.crop_position = 'random'
            self.tl_x = random.random()
            self.tl_y = random.random()
        elif self.crop_type == 'corner':
            self.crop_position = random.choice(['c', 'tl', 'tr', 'bl', 'br'])
        else:  # center
            self.crop_position = 'c'
        
        # Random horizontal flip
        self.flip = random.random() < 0.5 if self.horizontal_flip else False
        
        # Random color jitter parameters
        if self.color_jitter:
            self.brightness_factor = random.uniform(1 - self.brightness, 1 + self.brightness)
            self.contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast)
            self.saturation_factor = random.uniform(1 - self.saturation, 1 + self.saturation)
            self.hue_factor = random.uniform(-self.hue, self.hue)
        else:
            self.brightness_factor = 1.0
            self.contrast_factor = 1.0
            self.saturation_factor = 1.0
            self.hue_factor = 0.0
            
    def _resize(self, img):
        """Resize to target size"""
        img = cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        
        return img
    
    def _apply_multiscale_crop(self, img: np.ndarray) -> np.ndarray:
        """Apply multiscale crop to image"""
        h, w = img.shape[:2]
        min_length = min(h, w)
        crop_size = int(min_length * self.scale)
        
        # Determine crop coordinates
        if self.crop_position == 'random':
            x1 = int(self.tl_x * (w - crop_size))
            y1 = int(self.tl_y * (h - crop_size))
        elif self.crop_position == 'c':
            x1 = (w - crop_size) // 2
            y1 = (h - crop_size) // 2
        elif self.crop_position == 'tl':
            x1, y1 = 0, 0
        elif self.crop_position == 'tr':
            x1, y1 = w - crop_size, 0
        elif self.crop_position == 'bl':
            x1, y1 = 0, h - crop_size
        elif self.crop_position == 'br':
            x1, y1 = w - crop_size, h - crop_size
        
        x2 = x1 + crop_size
        y2 = y1 + crop_size
        
        # Crop
        img = img[y1:y2, x1:x2]
        
        return img
    
    def _apply_horizontal_flip(self, img: np.ndarray) -> np.ndarray:
        """Apply horizontal flip to image"""
        if self.flip:
            return cv2.flip(img, 1)
        return img
    
    def _apply_color_jitter(self, img: np.ndarray) -> np.ndarray:
        if not self.color_jitter:
            return img
        
        # Work entirely in float32 [0, 255] throughout
        img = img.astype(np.float32)
        
        # Brightness
        img = np.clip(img * self.brightness_factor, 0, 255)
        
        # Contrast (per-channel around channel mean)
        channel_mean = img.mean(axis=(0, 1), keepdims=True)
        img = np.clip((img - channel_mean) * self.contrast_factor + channel_mean, 0, 255)
        
        # Saturation & Hue (HSV space expects uint8)
        img_hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] * self.saturation_factor, 0, 255)
        img_hsv[:, :, 0] = (img_hsv[:, :, 0] + self.hue_factor) % 180
        
        img = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        return img
    
    def _normalize(self, img: np.ndarray) -> np.ndarray:
        """Normalize image with mean and std"""
        # Convert to float32 and scale to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Normalize
        img = (img - self.mean) / self.std
        
        return img
    
    def __call__(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """
        Apply augmentation to a single frame
        
        Args:
            frames: Input images as a lits of numpy arrays (T, H, W, C) in RGB format
            
        Returns:
            Augmented and normalized images as a lits of numpy array (T, H, W, C)
        """
        final_frames = []
        
        if self.mode == 'train':  
            for img in frames:
                img = self._apply_multiscale_crop(img)
                img = self._resize(img)
                img = self._apply_horizontal_flip(img)
                img = self._apply_color_jitter(img)
                img = self._normalize(img)
                final_frames.append(img)
        else:
            self.scale = 1.0
            self.crop_position = 'c'
            for img in frames:
                img = self._apply_multiscale_crop(img)
                img = self._resize(img)
                img = self._normalize(img)
                final_frames.append(img)
        
        return final_frames
    

def get_training_set(opt):
    """
    Get training dataset
    
    Args:
        opt: Configuration object with attributes:
            - input_size: Modlel input size
            - root_path: Path to dataset root
            - sample_duration: Number of frames per clip
            - sampling_step: Temporal stride (optional, default: 1)
            - num_classes: Number of classes
    """
    training_data = Kinetics(
        input_size = opt.input_size,
        root_path=opt.root_path,
        mode='train',
        sample_duration=opt.sample_duration,
        sampling_step=opt.sampling_step,
        num_classes=opt.num_classes
    )
    
    return training_data


def get_validation_set(opt):
    """
    Get validation dataset
    
    Args:
        opt: Configuration object with attributes:
            - input_size: Modlel input size
            - root_path: Path to dataset root
            - sample_duration: Number of frames per clip
            - sampling_step: Temporal stride (optional, default: 1)
            - num_classes: Number of classes
    """
    validation_data = Kinetics(
        input_size = opt.input_size,
        root_path=opt.root_path,
        mode='valid',
        sample_duration=opt.sample_duration,
        sampling_step=opt.sampling_step,
        num_classes=opt.num_classes
    )
    
    return validation_data