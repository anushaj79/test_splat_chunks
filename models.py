"""Data schemas for drone metadata layers."""
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

@dataclass(frozen=True)
class DroneImageMetadata:
    """Immutable record capturing geographical positioning of individual frames."""
    file_path: Path
    coordinates: Tuple[float, float]  # (Latitude, Longitude) in decimal degrees

    @property
    def latitude(self) -> float:
        return self.coordinates[0]

    @property
    def longitude(self) -> float:
        return self.coordinates[1]
