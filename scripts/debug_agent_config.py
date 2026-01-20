#!/usr/bin/env python3
"""Debug script to check agent config structure.

Run with Isaac Lab:
    python scripts/debug_agent_config.py
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Debug agent config")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch minimal app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Now test the config
from meldog_rl.agents import MeldogPPORunnerCfg

print("\n" + "=" * 60)
print("AGENT CONFIG DEBUG")
print("=" * 60)

# Instantiate config
cfg = MeldogPPORunnerCfg()
print(f"\nConfig class: {type(cfg).__name__}")

# Convert to dict
cfg_dict = cfg.to_dict()

print(f"\nTop-level keys: {list(cfg_dict.keys())}")

# Check policy
if "policy" in cfg_dict:
    print(f"\nPolicy keys: {list(cfg_dict['policy'].keys())}")
    if "class_name" in cfg_dict["policy"]:
        print(f"✓ policy.class_name = {cfg_dict['policy']['class_name']}")
    else:
        print("✗ MISSING: policy.class_name")
else:
    print("✗ MISSING: policy section")

# Check algorithm  
if "algorithm" in cfg_dict:
    print(f"\nAlgorithm keys: {list(cfg_dict['algorithm'].keys())}")
    if "class_name" in cfg_dict["algorithm"]:
        print(f"✓ algorithm.class_name = {cfg_dict['algorithm']['class_name']}")
    else:
        print("✗ MISSING: algorithm.class_name")
else:
    print("✗ MISSING: algorithm section")

# Print full dict for debugging
print("\n" + "=" * 60)
print("FULL CONFIG DICT:")
print("=" * 60)
import json
print(json.dumps(cfg_dict, indent=2, default=str))

simulation_app.close()
