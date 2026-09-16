import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from typing import List, Tuple, Optional
import random


class BoundingBox:
    """
    Bounding box representation for augmentation.
    
    Stores box in (x_min, y_min, x_max, y_max) format with
    normalized coordinates [0, 1] relative to image dimensions.
    """
    
    def __init__(self, x_min: float, y_min: float, x_max: float, y_max: float, class_id: int = 0):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.class_id = class_id
    
    def clip(self):
        """Clip coordinates to [0, 1] range."""
        self.x_min = max(0.0, min(1.0, self.x_min))
        self.y_min = max(0.0, min(1.0, self.y_min))
        self.x_max = max(0.0, min(1.0, self.x_max))
        self.y_max = max(0.0, min(1.0, self.y_max))
        return self
    
    def area(self) -> float:
        """Compute box area (normalized)."""
        return max(0, self.x_max - self.x_min) * max(0, self.y_max - self.y_min)
    
    def is_valid(self, min_area: float = 0.001) -> bool:
        """Check if box is valid (positive area above threshold)."""
        return self.area() > min_area and self.x_max > self.x_min and self.y_max > self.y_min
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x_min, self.y_min, self.x_max, self.y_max, self.class_id])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> "BoundingBox":
        return cls(arr[0], arr[1], arr[2], arr[3], int(arr[4]) if len(arr) > 4 else 0)


class AugmentationPipeline:
    """
    Comprehensive image augmentation pipeline for object detection.
    
    Applies a configurable sequence of random transformations while
    maintaining consistency between image pixels and bounding box
    coordinates.
    
    Usage:
        pipeline = AugmentationPipeline(
            horizontal_flip_prob=0.5,
            brightness_range=(0.7, 1.3),
            enable_weather=True,
        )
        
        aug_image, aug_boxes = pipeline.augment(image, boxes)
    """
    
    def __init__(
        self,
        # Geometric augmentations
        horizontal_flip_prob: float = 0.5,
        random_crop_prob: float = 0.3,
        crop_min_ratio: float = 0.7,
        random_scale_prob: float = 0.3,
        scale_range: Tuple[float, float] = (0.8, 1.2),
        random_translate_prob: float = 0.2,
        translate_range: float = 0.1,
        
        # Photometric augmentations
        brightness_prob: float = 0.5,
        brightness_range: Tuple[float, float] = (0.6, 1.4),
        contrast_prob: float = 0.5,
        contrast_range: Tuple[float, float] = (0.6, 1.4),
        saturation_prob: float = 0.4,
        saturation_range: Tuple[float, float] = (0.5, 1.5),
        hue_prob: float = 0.2,
        hue_range: float = 0.05,
        noise_prob: float = 0.2,
        noise_std: float = 10.0,
        
        # Domain-specific
        cutout_prob: float = 0.2,
        cutout_max_size: float = 0.15,
        cutout_num_patches: int = 3,
        motion_blur_prob: float = 0.1,
        fog_prob: float = 0.05,
        rain_prob: float = 0.05,
        
        # General
        enable_weather: bool = False,
        seed: Optional[int] = None,
    ):
        """
        Initialize the augmentation pipeline.
        
        All probability parameters control how often each augmentation
        is applied (0.0 = never, 1.0 = always).
        """
        self.horizontal_flip_prob = horizontal_flip_prob
        self.random_crop_prob = random_crop_prob
        self.crop_min_ratio = crop_min_ratio
        self.random_scale_prob = random_scale_prob
        self.scale_range = scale_range
        self.random_translate_prob = random_translate_prob
        self.translate_range = translate_range
        
        self.brightness_prob = brightness_prob
        self.brightness_range = brightness_range
        self.contrast_prob = contrast_prob
        self.contrast_range = contrast_range
        self.saturation_prob = saturation_prob
        self.saturation_range = saturation_range
        self.hue_prob = hue_prob
        self.hue_range = hue_range
        self.noise_prob = noise_prob
        self.noise_std = noise_std
        
        self.cutout_prob = cutout_prob
        self.cutout_max_size = cutout_max_size
        self.cutout_num_patches = cutout_num_patches
        self.motion_blur_prob = motion_blur_prob
        self.fog_prob = fog_prob
        self.rain_prob = rain_prob
        self.enable_weather = enable_weather
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def augment(
        self,
        image: Image.Image,
        boxes: List[BoundingBox],
    ) -> Tuple[Image.Image, List[BoundingBox]]:
        """
        Apply the full augmentation pipeline to an image and its boxes.
        
        Augmentations are applied in a fixed order but each is randomly
        activated based on its probability parameter.
        
        Args:
            image: Input PIL Image.
            boxes: List of bounding boxes (normalized coordinates).
            
        Returns:
            Tuple of (augmented_image, augmented_boxes).
        """
        img = image.copy()
        bxs = [BoundingBox(b.x_min, b.y_min, b.x_max, b.y_max, b.class_id) for b in boxes]
        
        # 1. Geometric augmentations (modify both image and boxes)
        if random.random() < self.horizontal_flip_prob:
            img, bxs = self._horizontal_flip(img, bxs)
        
        if random.random() < self.random_crop_prob:
            img, bxs = self._random_crop(img, bxs)
        
        # 2. Photometric augmentations (modify image only)
        if random.random() < self.brightness_prob:
            img = self._random_brightness(img)
        
        if random.random() < self.contrast_prob:
            img = self._random_contrast(img)
        
        if random.random() < self.saturation_prob:
            img = self._random_saturation(img)
        
        if random.random() < self.noise_prob:
            img = self._add_gaussian_noise(img)
        
        # 3. Domain-specific augmentations
        if random.random() < self.cutout_prob:
            img = self._random_cutout(img)
        
        if random.random() < self.motion_blur_prob:
            img = self._motion_blur(img)
        
        if self.enable_weather:
            if random.random() < self.fog_prob:
                img = self._simulate_fog(img)
            if random.random() < self.rain_prob:
                img = self._simulate_rain(img)
        
        # Filter out invalid boxes
        bxs = [b.clip() for b in bxs if b.is_valid()]
        
        return img, bxs
    
    # ─── Geometric Augmentations ─────────────────────────────────
    
    def _horizontal_flip(
        self, image: Image.Image, boxes: List[BoundingBox],
    ) -> Tuple[Image.Image, List[BoundingBox]]:
        """
        Flip image horizontally (left-right mirror).
        
        For autonomous driving, this simulates driving on the opposite
        side of the road. Bounding box x-coordinates are mirrored.
        
        Transformation: x' = 1 - x (for normalized coordinates)
        """
        flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        new_boxes = []
        for box in boxes:
            new_boxes.append(BoundingBox(
                x_min=1.0 - box.x_max,
                y_min=box.y_min,
                x_max=1.0 - box.x_min,
                y_max=box.y_max,
                class_id=box.class_id,
            ))
        
        return flipped, new_boxes
    
    def _random_crop(
        self, image: Image.Image, boxes: List[BoundingBox],
    ) -> Tuple[Image.Image, List[BoundingBox]]:
        """
        Randomly crop a portion of the image.
        
        The crop region is randomly selected to be at least
        crop_min_ratio of the original dimensions. Boxes that
        fall mostly outside the crop are removed.
        """
        w, h = image.size
        
        # Random crop dimensions
        crop_ratio = random.uniform(self.crop_min_ratio, 1.0)
        crop_w = int(w * crop_ratio)
        crop_h = int(h * crop_ratio)
        
        # Random crop position
        x_offset = random.randint(0, w - crop_w)
        y_offset = random.randint(0, h - crop_h)
        
        # Crop image
        cropped = image.crop((x_offset, y_offset, x_offset + crop_w, y_offset + crop_h))
        cropped = cropped.resize((w, h), Image.BICUBIC)
        
        # Transform boxes
        new_boxes = []
        x_off_norm = x_offset / w
        y_off_norm = y_offset / h
        
        for box in boxes:
            new_box = BoundingBox(
                x_min=(box.x_min - x_off_norm) / crop_ratio,
                y_min=(box.y_min - y_off_norm) / crop_ratio,
                x_max=(box.x_max - x_off_norm) / crop_ratio,
                y_max=(box.y_max - y_off_norm) / crop_ratio,
                class_id=box.class_id,
            )
            new_box.clip()
            
            # Keep box if at least 50% of its area is still visible
            original_area = box.area()
            if original_area > 0 and new_box.area() / original_area > 0.5:
                new_boxes.append(new_box)
        
        return cropped, new_boxes
    
    # ─── Photometric Augmentations ───────────────────────────────
    
    def _random_brightness(self, image: Image.Image) -> Image.Image:
        """
        Randomly adjust image brightness.
        
        Simulates different lighting conditions:
        - factor < 1.0: darker (shadow, tunnel, overcast)
        - factor > 1.0: brighter (direct sunlight, glare)
        """
        factor = random.uniform(*self.brightness_range)
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)
    
    def _random_contrast(self, image: Image.Image) -> Image.Image:
        """
        Randomly adjust image contrast.
        
        Simulates different camera exposure settings and
        atmospheric conditions affecting visibility.
        """
        factor = random.uniform(*self.contrast_range)
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)
    
    def _random_saturation(self, image: Image.Image) -> Image.Image:
        """
        Randomly adjust color saturation.
        
        Simulates different color renderings from various
        camera sensors and white balance settings.
        """
        factor = random.uniform(*self.saturation_range)
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(factor)
    
    def _add_gaussian_noise(self, image: Image.Image) -> Image.Image:
        """
        Add random Gaussian noise to simulate sensor noise.
        
        Camera sensors produce noise especially in low-light conditions.
        noise ~ N(0, σ²) where σ is noise_std parameter.
        """
        img_array = np.array(image, dtype=np.float32)
        noise = np.random.normal(0, self.noise_std, img_array.shape)
        noisy = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)
    
    # ─── Domain-Specific Augmentations ───────────────────────────
    
    def _random_cutout(self, image: Image.Image) -> Image.Image:
        """
        Apply random rectangular cutout (occlusion simulation).
        
        Theory: Cutout forces the network to learn from partial
        information, improving robustness to occlusion — common
        in driving scenarios where vehicles partially block each other.
        
        Reference: DeVries & Taylor, "Improved Regularization of
        Convolutional Neural Networks with Cutout" (2017)
        """
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        for _ in range(self.cutout_num_patches):
            # Random patch size
            patch_h = int(h * random.uniform(0.02, self.cutout_max_size))
            patch_w = int(w * random.uniform(0.02, self.cutout_max_size))
            
            # Random position
            y = random.randint(0, h - patch_h)
            x = random.randint(0, w - patch_w)
            
            # Fill with random gray value (more realistic than black)
            fill_value = random.randint(50, 200)
            img_array[y:y+patch_h, x:x+patch_w] = fill_value
        
        return Image.fromarray(img_array)
    
    def _motion_blur(self, image: Image.Image) -> Image.Image:
        """
        Apply horizontal motion blur to simulate camera shake or fast movement.
        
        In driving scenarios, motion blur occurs when the vehicle
        is moving at high speeds or during sudden maneuvers.
        """
        kernel_size = random.choice([3, 5, 7])
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size // 2, :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        
        # Apply using PIL filter
        return image.filter(ImageFilter.Kernel(
            size=(kernel_size, kernel_size),
            kernel=kernel.flatten().tolist(),
            scale=1,
            offset=0,
        ))
    
    def _simulate_fog(self, image: Image.Image) -> Image.Image:
        """
        Simulate foggy conditions by blending with white overlay.
        
        Fog reduces visibility non-uniformly — closer objects are
        clearer than distant ones (atmospheric scattering).
        """
        img_array = np.array(image, dtype=np.float32)
        
        # Fog intensity
        fog_intensity = random.uniform(0.1, 0.4)
        
        # Create gradient fog (thicker at top = distance)
        h, w = img_array.shape[:2]
        fog_gradient = np.linspace(fog_intensity, fog_intensity * 0.3, h)
        fog_gradient = fog_gradient.reshape(h, 1, 1)
        
        # Blend
        white = np.ones_like(img_array) * 255
        fogged = img_array * (1 - fog_gradient) + white * fog_gradient
        
        return Image.fromarray(np.clip(fogged, 0, 255).astype(np.uint8))
    
    def _simulate_rain(self, image: Image.Image) -> Image.Image:
        """
        Simulate rain by adding streak-like noise and slight blur.
        
        Rain affects driving detection by:
        1. Adding visual noise (raindrops)
        2. Reducing contrast (wet surfaces reflect)
        3. Partial occlusion of objects
        """
        img_array = np.array(image, dtype=np.float32)
        h, w = img_array.shape[:2]
        
        # Create rain streaks
        num_drops = random.randint(100, 500)
        for _ in range(num_drops):
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 10)
            length = random.randint(5, 15)
            
            y_end = min(h, y + length)
            img_array[y:y_end, x] = np.clip(
                img_array[y:y_end, x] + 60, 0, 255
            )
        
        # Slight brightness reduction (overcast)
        img_array = img_array * 0.85
        
        result = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
        
        # Slight blur for wet lens effect
        result = result.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        return result
    
    @classmethod
    def driving_preset(cls) -> "AugmentationPipeline":
        """
        Factory method: Augmentation preset optimized for driving data.
        
        Enables weather simulation and uses parameters tuned for
        typical dashcam footage variations.
        """
        return cls(
            horizontal_flip_prob=0.5,
            random_crop_prob=0.3,
            crop_min_ratio=0.75,
            brightness_prob=0.6,
            brightness_range=(0.5, 1.5),
            contrast_prob=0.5,
            saturation_prob=0.4,
            noise_prob=0.3,
            noise_std=15.0,
            cutout_prob=0.3,
            cutout_num_patches=2,
            motion_blur_prob=0.15,
            enable_weather=True,
            fog_prob=0.1,
            rain_prob=0.1,
        )
    
    @classmethod
    def light_preset(cls) -> "AugmentationPipeline":
        """Light augmentation for validation tuning."""
        return cls(
            horizontal_flip_prob=0.5,
            random_crop_prob=0.0,
            brightness_prob=0.3,
            brightness_range=(0.8, 1.2),
            contrast_prob=0.3,
            contrast_range=(0.8, 1.2),
            saturation_prob=0.0,
            noise_prob=0.1,
            noise_std=5.0,
            cutout_prob=0.0,
            motion_blur_prob=0.0,
            enable_weather=False,
        )
