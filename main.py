"""Main entry point orchestrating data chunking and training."""
import sys
import math  # Crucial for ceiling calculations
import config
from chunker import SpatialChunker
from trainer import HeadlessBatchTrainer
from utils import resolve_next_output_dir

def main() -> None:
    """Coordinates validation, spatial segmentation, and automated training."""
    print("🎬 Initializing Agricultural Gaussian Splatting Script Pipeline...")
    
    if not config.SOURCE_DIR.exists():
        print(f"❌ Error: Raw image source path missing at {config.SOURCE_DIR}")
        sys.exit(1)

    # 1. Resolve Output Path
    active_output_run = resolve_next_output_dir(config.OUTPUT_PARENT_DIR)
    print(f"📂 Resolved Output Run Target Path: {active_output_run}")

    # 2. Execute Data Segmentation Chunking
    try:
        # Count the number of valid drone photos in the folder dynamically
        valid_files = [
            f for f in config.SOURCE_DIR.iterdir() 
            if f.suffix.lower() in config.VALID_EXTENSIONS
        ]
        total_images = len(valid_files)
        
        if total_images == 0:
            print(f"❌ Error: No valid images found in {config.SOURCE_DIR}")
            sys.exit(1)
            
        # DYNAMIC ALGEBRA: Total Images / N = 100  -->  N = Total Images / 100
        TARGET_IMAGES_PER_CHUNK = 100
        DYNAMIC_N = math.ceil(total_images / TARGET_IMAGES_PER_CHUNK)
        
        print(f"📊 Dataset Analysis: Found {total_images} raw drone photos.")
        print(f"🔄 Solving Allocation: N = {DYNAMIC_N} chunks (~{total_images // DYNAMIC_N} base images per chunk)")

        # Initialize and sort
        chunker = SpatialChunker(source_dir=config.SOURCE_DIR, output_dir=active_output_run)
        chunker.gather_and_sort_images()
        
        # Inject the freshly calculated chunk count directly into the cluster distribution loop
        chunker.distribute_chunks(
            num_chunks=DYNAMIC_N, 
            overlap_pct=config.DEFAULT_OVERLAP_MARGIN
        )
        print("📊 Data successfully partitioned into overlapping chunks on disk.")
    except Exception as e:
        print(f"❌ Pipeline Failed during data chunking phase: {e}")
        sys.exit(1)

    # 3. Execute Headless Sequential Model Training
    trainer = HeadlessBatchTrainer(run_directory=active_output_run)
    trainer.execute_loop()
    
    print("\n🏁 Script executed successfully! All output objects are stored on disk.")

if __name__ == "__main__":
    main()
