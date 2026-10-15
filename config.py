"""Central configuration module for the spatial chunker pipeline."""
from pathlib import Path

# 1. Base Project Directory
BASE_DIR = Path.home() / "Desktop/CSE120/splat_chunks"

# 2. Source Drone Directory (Steps back out of splat_chunks to find your images)
SOURCE_DIR = BASE_DIR.parent / "9:22 drone images/DJI_202609271005_001_Basil-smart-oblique"

# 3. Corrected Output Target Path
OUTPUT_PARENT_DIR = BASE_DIR / "outputs"

# Algorithmic Parameters
DEFAULT_NUM_CHUNKS = 20         # Forces exactly 20 manageable chunks (~100 images per block)
DEFAULT_OVERLAP_MARGIN = 0.20  # 20% overlap padding index buffer

# File System Constraints
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.dng'}

# Training Pipeline Hyperparameters (Nerfstudio Splatfacto Optimized for M1 Mac)
MAX_TRAINING_ITERATIONS = 10000
TRAINER_ENGINE = "splatfacto"

# Absolute local binary mapping
COLMAP_BINARY_PATH = "/opt/homebrew/bin/colmap"
