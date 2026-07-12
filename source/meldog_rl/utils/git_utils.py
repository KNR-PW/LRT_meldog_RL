# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Git utilities for tracking policy origins."""

import subprocess
from pathlib import Path
import yaml


def get_git_suffix() -> str:
    """Get the short git commit hash to append to directory names.
    
    Returns:
        e.g., "a1b2c3d" or "a1b2c3d-dirty". Empty string if git fails.
    """
    try:
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        
        is_dirty = subprocess.run(
            ["git", "diff", "--quiet"], stderr=subprocess.DEVNULL
        ).returncode != 0
        
        if is_dirty:
            return f"{commit_hash}-dirty"
        return commit_hash
    except Exception:
        return ""


def save_git_metadata(log_dir: str | Path):
    """Save comprehensive git metadata to params/git_info.yaml.
    
    Captures:
    - commit_hash
    - branch
    - is_dirty
    - uncommitted_diff (if dirty)
    
    Args:
        log_dir: Path to the log directory (e.g. logs/locomotion/LM_...)
    """
    log_dir = Path(log_dir)
    params_dir = log_dir / "params"
    params_dir.mkdir(parents=True, exist_ok=True)
    
    git_info = {}
    try:
        git_info["commit_hash"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        
        git_info["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        
        is_dirty = subprocess.run(
            ["git", "diff", "--quiet"], stderr=subprocess.DEVNULL
        ).returncode != 0
        
        git_info["is_dirty"] = is_dirty
        
        if is_dirty:
            git_info["uncommitted_diff"] = subprocess.check_output(
                ["git", "diff"], stderr=subprocess.DEVNULL
            ).decode("utf-8")
            
    except Exception as e:
        git_info["error"] = f"Failed to extract git info: {e}"
    
    with open(params_dir / "git_info.yaml", "w") as f:
        yaml.dump(git_info, f, default_flow_style=False)
