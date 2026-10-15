"""Automated model orchestration handling direct COLMAP and training blocks."""
import os
import subprocess
from pathlib import Path
import config

class HeadlessBatchTrainer:
    """Manages individual model executions sequentially to isolate VRAM."""
    
    def __init__(self, run_directory: Path):
        self.run_directory = run_directory
        self.target_chunks = sorted(
            [d for d in run_directory.iterdir() if d.is_dir() and d.name.startswith("chunk_")]
        )

    def execute_loop(self) -> None:
        """Sequentially processes data and triggers Nerfstudio training for each chunk."""
        if not self.target_chunks:
            print(f"⚠️ Terminating training task: No valid chunks inside {self.run_directory}")
            return

        custom_env = os.environ.copy()
        custom_env["PATH"] = f"/opt/homebrew/bin:/usr/local/bin:{custom_env.get('PATH', '')}"

        for chunk in self.target_chunks:
            print(f"\n" + "="*60)
            print(f"📦 PROCESSING AND TRAINING TARGET: {chunk.name.upper()}")
            print("="*60)
            
            # Formulate the EXACT directory layout that standard Nerfstudio parsers require
            colmap_dir = chunk / "colmap"
            db_path = colmap_dir / "database.db"
            
            # CRUCIAL PATH RE-FIX: Output directly to the parent colmap_dir.
            # COLMAP's mapper automatically appends the 'sparse/0' structure under this directory!
            sparse_output_base = colmap_dir
            
            # Clean up old broken folders if they exist from previous runs
            old_sparse = colmap_dir / "sparse"
            if old_sparse.exists():
                subprocess.run(["rm", "-rf", str(old_sparse.resolve())])
            
            colmap_dir.mkdir(exist_ok=True)

            # --- PHASE 1: DIRECT STABLE CPU COLMAP EXTRACTION ---
            print(f"🛠️ Step 1a: Extracting SIFT visual keypoints via native COLMAP...")
            extract_cmd = [
                "colmap", "feature_extractor",
                "--database_path", str(db_path.resolve()),
                "--image_path", str(chunk.resolve()),
                "--ImageReader.single_camera", "1",
                "--ImageReader.camera_model", "OPENCV"
            ]
            
            try:
                res = subprocess.run(extract_cmd, env=custom_env, capture_output=True, text=True)
                if res.returncode != 0:
                    print(f"❌ COLMAP Feature Extraction failed for {chunk.name}:\n{res.stderr}")
                    continue
                print(f"✅ Features extracted successfully.")
                
                # --- PHASE 2: DIRECT FEATURE MATCHING ---
                print(f"🛠️ Step 1b: Running exhaustive sequence feature matching...")
                match_cmd = [
                    "colmap", "exhaustive_matcher",
                    "--database_path", str(db_path.resolve())
                ]
                res = subprocess.run(match_cmd, env=custom_env, capture_output=True, text=True)
                if res.returncode != 0:
                    print(f"❌ COLMAP Matching failed for {chunk.name}:\n{res.stderr}")
                    continue
                print(f"✅ Feature matching complete.")

                # --- PHASE 3: SPARSE MAPPER ---
                print(f"🛠️ Step 1c: Generating camera matrices via COLMAP mapper...")
                map_cmd = [
                    "colmap", "mapper",
                    "--database_path", str(db_path.resolve()),
                    "--image_path", str(chunk.resolve()),
                    "--output_path", str(sparse_output_base.resolve())
                ]
                res = subprocess.run(map_cmd, env=custom_env, capture_output=True, text=True)
                if res.returncode != 0:
                    print(f"❌ COLMAP Mapper failed for {chunk.name}:\n{res.stderr}")
                    continue
                print(f"✅ Camera sparse alignment complete.")

            except Exception as e:
                print(f"❌ Camera trajectory setup crashed for {chunk.name}: {e}")
                continue

            # --- PHASE 4: SPLATFACTO TRAINING FROM RAW COLMAP DIRECTORY ---
            print(f"🏋️ Step 2: Training 3D Gaussian Splat model directly via colmap dataparser...")
            model_out = chunk / "trained_model"
            model_out.mkdir(exist_ok=True)

            train_command = [
                "ns-train", config.TRAINER_ENGINE,
                "--output-dir", str(model_out.resolve()),
                "--viewer.quit-on-train-completion", "True",
                "--max-num-iterations", str(config.MAX_TRAINING_ITERATIONS),
                "--data", str(chunk.resolve()),
                "colmap",                             
                "--images-path", str(chunk.resolve()) 
                # "--colmap-path", "colmap/sparse/0" # Points directly to colmap/0
                ]

            try:
                # ST_DOUT tracking allows us to watch errors live as they happen
                with subprocess.Popen(
                    train_command, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT, 
                    text=True,
                    env=custom_env
                ) as process:
                    if process.stdout:
                        for line in process.stdout:
                            # Print all output lines so we catch missing argument flags instantly
                            print(f"[{chunk.name}-Train]: {line.strip()}")
                process.wait()
                
                if process.returncode == 0:
                    print(f"✅ SUCCESS: Finished training model for {chunk.name}!")
                else:
                    print(f"❌ ERROR: Nerfstudio train process failed with exit status code {process.returncode}")
                    break # Break immediately so we can fix the bug rather than cycling 26 failures
            except Exception as e:
                print(f"❌ Critical system termination processing {chunk.name}: {e}")
                break
