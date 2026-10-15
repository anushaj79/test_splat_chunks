"""Core algorithmic module handling geographical data splitting."""
import math
from pathlib import Path
from typing import List
import config
from models import DroneImageMetadata
from utils import extract_drone_gps, safely_copy_file

class SpatialChunker:
    """Handles sorting and segmenting images into spatial chunks based on layout."""
    
    def __init__(self, source_dir: Path, output_dir: Path):
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.image_pool: List[DroneImageMetadata] = []

    def gather_and_sort_images(self) -> None:
        """Discovers, parses, and aligns imagery along the principal flight path."""
        for item in self.source_dir.iterdir():
            if item.suffix.lower() in config.VALID_EXTENSIONS:
                meta = extract_drone_gps(item)
                if meta:
                    self.image_pool.append(meta)

        if not self.image_pool:
            raise FileNotFoundError(f"Zero georeferenced data profiles within: {self.source_dir}")

        # Compute geometric variance boundaries
        lats = [img.latitude for img in self.image_pool]
        lons = [img.longitude for img in self.image_pool]
        
        # Sort along the field's longest axis (North-South vs East-West)
        if (max(lats) - min(lats)) > (max(lons) - min(lons)):
            self.image_pool.sort(key=lambda img: img.latitude)
        else:
            self.image_pool.sort(key=lambda img: img.longitude)

    def distribute_chunks(self, num_chunks: int, overlap_pct: float) -> None:
        """Splits data into individual chunk folders with overlapping margins."""
        total_files = len(self.image_pool)
        base_size = total_files / num_chunks
        overlap_size = math.ceil(base_size * overlap_pct)

        for i in range(num_chunks):
            start_idx = int(i * base_size)
            end_idx = int((i + 1) * base_size) if i < num_chunks - 1 else total_files
            
            # Apply buffer boundaries
            buffered_start = max(0, start_idx - overlap_size)
            buffered_end = min(total_files, end_idx + overlap_size)
            
            chunk_slice = self.image_pool[buffered_start:buffered_end]
            chunk_folder = self.output_dir / f"chunk_{i+1:02d}"
            
            for img in chunk_slice:
                safely_copy_file(img.file_path, chunk_folder / img.file_path.name)
