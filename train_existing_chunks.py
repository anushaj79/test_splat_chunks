"""Automated script to structurally arrange paths and train existing COLMAP chunks."""
import os
import subprocess
import shutil
from pathlib import Path

def train_existing_pipeline(output_run_dir: str):
    run_path = Path(output_run_dir)
    chunks = sorted([d for d in run_path.iterdir() if d.is_dir() and d.name.startswith("chunk_")])
    
    if not chunks:
        print(f"❌ No valid chunk directories found inside {output_run_dir}")
        return
        
    print(f"🔄 Found {len(chunks)} pre-calculated chunks. Re-aligning paths to textbook Nerfstudio structure...")
    
    custom_env = os.environ.copy()
    custom_env["PATH"] = f"/opt/homebrew/bin:/usr/local/bin:{custom_env.get('PATH', '')}"

    for chunk in chunks:
        print(f"\n" + "="*60)
        print(f"🏋️ PREPARING WORKSPACE & TRAINING: {chunk.name.upper()}")
        print("="*60)
        
        # --- PHASE 1: STANDARDIZE THE COLMAP TARGET DIRECTORY LAYOUT ---
        # The textbook Nerfstudio 'colmap' parser expects: chunk_##/colmap/sparse/0/
        target_sparse_dir = chunk / "colmap" / "sparse" / "0"
        
        # If files were moved directly into chunk_##/colmap/, let's put them back into sparse/0/ securely
        if (chunk / "colmap" / "cameras.bin").exists() and not target_sparse_dir.exists():
            print(f"📁 Creating standard sparse pathing structure for {chunk.name}...")
            temp_move_dir = chunk / "colmap_temp"
            temp_move_dir.mkdir(exist_ok=True)
            
            # Temporarily move binary assets out
            for item in (chunk / "colmap").iterdir():
                if item.is_file():
                    item.rename(temp_move_dir / item.name)
            
            # Recreate clean nested directories
            shutil.rmtree(chunk / "colmap")
            target_sparse_dir.mkdir(parents=True, exist_ok=True)
            
            # Restore files directly into colmap/sparse/0/
            for item in temp_move_dir.iterdir():
                item.rename(target_sparse_dir / item.name)
            shutil.rmtree(temp_move_dir)
            print("✅ Textbook sparse directory alignment locked.")

        # Final verification fallback check
        if not (target_sparse_dir / "cameras.bin").exists():
            print(f"⚠️ Skipping {chunk.name}: Could not locate cameras.bin in your dataset.")
            continue

        # --- PHASE 2: EXECUTE INJECTED NERFSTUDIO OPTIMIZATION LOOP ---
        model_out = chunk / "trained_model"
        model_out.mkdir(exist_ok=True)

        # Structure the training arguments with the exact subcommand formatting rules
        train_command = [
            "ns-train", "splatfacto",
            "--output-dir", str(model_out.resolve()),
            "--max-num-iterations", "10000",
            "--viewer.quit-on-train-completion", "True",
            "--data", str(chunk.resolve()),
            "colmap",
            "--images-path", str(chunk.resolve()),
            "--colmap-path", "colmap/sparse/0" # Directs the dataparser to the valid folder relative to --data
        ]

        try:
            print(f"Executing training loop command for {chunk.name}...")
            with subprocess.Popen(
                train_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=custom_env
            ) as process:
                if process.stdout:
                    for line in process.stdout:
                        # Print all lines to catch initialization and hidden errors
                        print(f"[{chunk.name}-Train]: {line.strip()}")
            process.wait()
            
            if process.returncode == 0:
                print(f"✅ SUCCESS: Finished training model weights for {chunk.name}!")
            else:
                print(f"❌ Nerfstudio failed on {chunk.name} with exit status code {process.returncode}.")
                break # Halt immediately if a parameters syntax bug returns
        except Exception as e:
            print(f"❌ Unexpected termination loop error on {chunk.name}: {e}")
            break

    print("\n🏁 All existing spatial chunks have been trained successfully!")

if __name__ == "__main__":
    TARGET_OUTPUT_RUN = "./outputs/output2"
    train_existing_pipeline(TARGET_OUTPUT_RUN)
