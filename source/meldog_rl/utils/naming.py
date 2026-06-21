# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Naming utilities for consistent output directory naming.

Naming convention:
    {PREFIX}_{details}_{YYYY-MM-DD}_{HH-MM-SS}

Prefixes:
    LM = Locomotion Model (training output)
    LE = Locomotion Evaluation
    PM = Perception Model (training output)
    PE = Perception Evaluation
    PD = Perception Dataset
"""

from datetime import datetime
from pathlib import Path
from typing import Literal

from meldog_rl.utils.git_utils import get_git_suffix


# Type alias for prefixes
OutputPrefix = Literal["LM", "LE", "PM", "PE", "PD"]


def get_timestamp() -> str:
    """Get current timestamp in standard format.
    
    Returns:
        Timestamp string: YYYY-MM-DD_HH-MM-SS
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def make_output_name(
    prefix: OutputPrefix,
    *details: str,
    timestamp: str | None = None,
) -> str:
    """Create output directory name following naming convention.
    
    Args:
        prefix: Output type prefix (LM, LE, PM, PE, PD)
        *details: Variable details to include (terrain, model type, etc.)
        timestamp: Optional timestamp, defaults to current time
        
    Returns:
        Formatted name: {PREFIX}_{details}_{timestamp}
        
    Examples:
        >>> make_output_name("LM", "rough", "sim")
        'LM_rough_sim_2026-01-19_12-00-00'
        
        >>> make_output_name("PM", "hmv6", "rough")
        'PM_hmv6_rough_2026-01-19_12-00-00'
        
        >>> make_output_name("PD", "flat")
        'PD_flat_2026-01-19_12-00-00'
    """
    if timestamp is None:
        timestamp = get_timestamp()
    
    parts = [prefix] + list(details) + [timestamp]
    base_name = "_".join(parts)
    
    git_suffix = get_git_suffix()
    if git_suffix:
        return f"{base_name}_{git_suffix}"
    return base_name


def make_locomotion_log_dir(
    terrain: str,
    domain: str = "sim",
    base_dir: str | Path = "logs/locomotion",
    timestamp: str | None = None,
) -> Path:
    """Create locomotion training log directory path.
    
    Args:
        terrain: Terrain type (flat, rough, flat_obs, rough_obs)
        domain: Domain type (sim, real)
        base_dir: Base logs directory
        timestamp: Optional timestamp
        
    Returns:
        Path to log directory
        
    Example:
        >>> make_locomotion_log_dir("rough", "sim")
        PosixPath('logs/locomotion/LM_rough_sim_2026-01-19_12-00-00')
    """
    name = make_output_name("LM", terrain, domain, timestamp=timestamp)
    return Path(base_dir) / name


def make_perception_log_dir(
    model_type: str,
    dataset_terrain: str,
    base_dir: str | Path = "logs/perception",
    timestamp: str | None = None,
) -> Path:
    """Create perception training log directory path.
    
    Args:
        model_type: Model architecture (hmv5, hmv6, voxel, slam)
        dataset_terrain: Terrain the dataset was collected on
        base_dir: Base logs directory
        timestamp: Optional timestamp
        
    Returns:
        Path to log directory
        
    Example:
        >>> make_perception_log_dir("hmv6", "rough")
        PosixPath('logs/perception/PM_hmv6_rough_2026-01-19_12-00-00')
    """
    name = make_output_name("PM", model_type, dataset_terrain, timestamp=timestamp)
    return Path(base_dir) / name


def make_dataset_dir(
    terrain: str,
    base_dir: str | Path = "datasets",
    timestamp: str | None = None,
) -> Path:
    """Create dataset collection directory path.
    
    Args:
        terrain: Terrain type for dataset
        base_dir: Base datasets directory
        timestamp: Optional timestamp
        
    Returns:
        Path to dataset directory
        
    Example:
        >>> make_dataset_dir("rough")
        PosixPath('datasets/PD_rough_2026-01-19_12-00-00')
    """
    name = make_output_name("PD", terrain, timestamp=timestamp)
    return Path(base_dir) / name


def make_evaluation_dir(
    eval_type: Literal["locomotion", "perception"],
    *details: str,
    base_dir: str | Path | None = None,
    timestamp: str | None = None,
) -> Path:
    """Create evaluation output directory path.
    
    Args:
        eval_type: Type of evaluation (locomotion, perception)
        *details: Additional details (model name, terrain, etc.)
        base_dir: Base directory (defaults based on eval_type)
        timestamp: Optional timestamp
        
    Returns:
        Path to evaluation directory
    """
    prefix: OutputPrefix = "LE" if eval_type == "locomotion" else "PE"
    
    if base_dir is None:
        base_dir = f"logs/{eval_type}"
    
    name = make_output_name(prefix, *details, timestamp=timestamp)
    return Path(base_dir) / name


def parse_output_name(name: str) -> dict:
    """Parse an output directory name into components.
    
    Args:
        name: Output directory name
        
    Returns:
        Dictionary with prefix, details, and timestamp
        
    Example:
        >>> parse_output_name("LM_rough_sim_2026-01-19_12-00-00")
        {'prefix': 'LM', 'details': ['rough', 'sim'], 'timestamp': '2026-01-19_12-00-00'}
    """
    parts = name.split("_")
    
    # Check if the last part is a git suffix (e.g., a1b2c3d or a1b2c3d-dirty)
    # A standard timestamp is YYYY-MM-DD_HH-MM-SS (2 parts)
    # If the last part has no hyphens or only one hyphen but is short, it might be git.
    # But a cleaner way: if we assume standard format, the timestamp starts with 202...
    
    # Let's find the timestamp part which usually starts with '20' (year)
    timestamp_idx = -1
    for i, p in enumerate(parts):
        if p.startswith("20") and len(p.split("-")) == 3:
            timestamp_idx = i
            break
            
    if timestamp_idx != -1 and timestamp_idx + 1 < len(parts):
        timestamp = f"{parts[timestamp_idx]}_{parts[timestamp_idx+1]}"
        details = parts[1:timestamp_idx]
        git_suffix = "_".join(parts[timestamp_idx+2:])
    else:
        # Fallback to old behavior
        timestamp = "_".join(parts[-2:])
        details = parts[1:-2]
        git_suffix = ""
    
    return {
        "prefix": parts[0],
        "details": details,
        "timestamp": timestamp,
        "git_suffix": git_suffix,
    }


# Prefix descriptions for documentation
PREFIX_DESCRIPTIONS = {
    "LM": "Locomotion Model - Training output from locomotion policy",
    "LE": "Locomotion Evaluation - Evaluation results for locomotion",
    "PM": "Perception Model - Training output from perception model",
    "PE": "Perception Evaluation - Evaluation results for perception",
    "PD": "Perception Dataset - Collected dataset for perception training",
}
