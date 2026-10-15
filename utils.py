"""Utility functions handling metadata parsing and file system operations."""
import os
import shutil
from pathlib import Path
from typing import Optional, Tuple
from exif import Image
from models import DroneImageMetadata

def convert_to_decimal(coords: tuple, ref: str) -> float:
    """Converts standard EXIF degree/minute/second tuples to pure decimal format."""
    decimal = coords[0] + coords[1] / 60.0 + coords[2] / 3600.0
    if ref in ['S', 'W']:
        decimal = -decimal
    return decimal

def extract_drone_gps(image_path: Path) -> Optional[DroneImageMetadata]:
    """Parses individual image files to retrieve physical flight coordinates."""
    try:
        with open(image_path, 'rb') as src:
            img = Image(src)
            
        if not img.has_exif or 'gps_latitude' not in dir(img):
            return None

        lat = convert_to_decimal(img.gps_latitude, img.gps_latitude_ref)
        lon = convert_to_decimal(img.gps_longitude, img.gps_longitude_ref)
        return DroneImageMetadata(file_path=image_path, coordinates=(lat, lon))
    except Exception:
        # Suppress corrupt data logs silently to clean up terminal streaming stdout
        return None

def resolve_next_output_dir(parent_dir: Path) -> Path:
    """Increments folder counts on disk sequentially without overwriting historic data."""
    counter = 1
    while (parent_dir / f"output{counter}").exists():
        counter += 1
    return parent_dir / f"output{counter}"

def safely_copy_file(source: Path, destination: Path) -> None:
    """Creates a 0-byte system shortcut pointing to the source image, saving storage."""
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Remove old broken files or symlinks if they exist from a past run
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        # Create a zero-byte symlink shortcut back to your main DJI folder
        destination.symlink_to(source.resolve())
    except Exception as e:
        raise SystemError(f"Critical operational shortcut lockup writing to {destination}: {e}")

