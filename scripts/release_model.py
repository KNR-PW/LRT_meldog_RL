#!/usr/bin/env python3
# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Release a trained model to the releases/ directory.

Packages the best weights and configuration files into a clean folder.

Usage:
    python scripts/release_model.py \
        --domain locomotion \
        --log-dir logs/locomotion/LM_rough_sim_2026-02-06_14-23-29 \
        --tag v1.0-rough
"""

import argparse
import os
import shutil
import glob
import re
from pathlib import Path


def find_best_locomotion_model(log_dir: Path) -> Path:
    """Find the model with the highest iteration number."""
    models = list(log_dir.glob("model_*.pt"))
    if not models:
        raise FileNotFoundError(f"No model_*.pt files found in {log_dir}")
    
    max_iter = -1
    best_model = None
    
    for model in models:
        # Extract number from model_1234.pt
        match = re.search(r"model_(\d+)\.pt", model.name)
        if match:
            iteration = int(match.group(1))
            if iteration > max_iter:
                max_iter = iteration
                best_model = model
                
    if best_model is None:
        raise FileNotFoundError(f"Could not parse iteration numbers in {log_dir}")
        
    return best_model


def find_best_perception_model(log_dir: Path) -> Path:
    """Find model_best.pt, fallback to highest epoch."""
    best_model = log_dir / "model_best.pt"
    if best_model.exists():
        return best_model
        
    # Fallback
    models = list(log_dir.glob("model_ep*.pt"))
    if not models:
        raise FileNotFoundError(f"No model_best.pt or model_ep*.pt found in {log_dir}")
        
    max_ep = -1
    best_ep_model = None
    
    for model in models:
        match = re.search(r"model_ep(\d+)\.pt", model.name)
        if match:
            ep = int(match.group(1))
            if ep > max_ep:
                max_ep = ep
                best_ep_model = model
                
    if best_ep_model is None:
        raise FileNotFoundError(f"Could not parse epoch numbers in {log_dir}")
        
    return best_ep_model


def main():
    parser = argparse.ArgumentParser(description="Release a model.")
    parser.add_argument("--domain", type=str, required=True, choices=["locomotion", "perception"],
                        help="Domain of the model.")
    parser.add_argument("--log-dir", type=str, required=True,
                        help="Path to the training log directory.")
    parser.add_argument("--tag", type=str, required=True,
                        help="Release tag (e.g., v1.0-rough, v2.1-flat-obs).")
    parser.add_argument("--message", "-m", type=str, default=None,
                        help="Release notes or message to include in the README.md")
    
    args = parser.parse_args()
    log_dir = Path(args.log_dir)
    
    if not log_dir.exists() or not log_dir.is_dir():
        print(f"[ERROR] Log directory does not exist: {log_dir}")
        return
        
    # Create release directory
    release_dir = Path("releases") / args.domain / args.tag
    if release_dir.exists():
        print(f"[ERROR] Release tag '{args.tag}' already exists at {release_dir}")
        print("Please use a different tag or manually delete the existing release.")
        return
        
    print(f"[INFO] Preparing release '{args.tag}' for {args.domain}...")
    release_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Find and copy the best model weights
        if args.domain == "locomotion":
            best_model_path = find_best_locomotion_model(log_dir)
        else:
            best_model_path = find_best_perception_model(log_dir)
            
        print(f"  -> Selected best weights: {best_model_path.name}")
        shutil.copy2(best_model_path, release_dir / "model.pt")
        
        # 2. Copy configuration YAMLs
        params_dir = log_dir / "params"
        if params_dir.exists():
            for yaml_file in params_dir.glob("*.yaml"):
                print(f"  -> Copying config: {yaml_file.name}")
                shutil.copy2(yaml_file, release_dir / yaml_file.name)
        else:
            print(f"  -> [WARNING] No 'params/' directory found in {log_dir}!")
            
        # 3. Create README.md with release notes
        readme_path = release_dir / "README.md"
        with open(readme_path, "w") as f:
            f.write(f"# Release: {args.tag}\n\n")
            f.write(f"**Domain:** {args.domain}\n")
            f.write(f"**Source Log:** {log_dir.name}\n\n")
            f.write("## Release Notes\n")
            if args.message:
                f.write(f"{args.message}\n")
            else:
                f.write("*(Add your release notes here)*\n")
        print(f"  -> Created README.md")
            
        print(f"\n[SUCCESS] Model released successfully to: {release_dir}")
        print("You can now safely commit this folder to Git.")
        
    except Exception as e:
        print(f"\n[ERROR] Release failed: {e}")
        # Clean up the partially created directory
        if release_dir.exists():
            shutil.rmtree(release_dir)


if __name__ == "__main__":
    main()
