#!/usr/bin/env python3
"""Debug script to check agent config structure."""

import argparse
import json
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Debug agent config")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from meldog_rl.agents import MeldogPPORunnerCfg

print("\nConfig class:", type(cfg).__name__)

cfg = MeldogPPORunnerCfg()
cfg_dict = cfg.to_dict()

print("Top-level keys:", list(cfg_dict.keys()))

if "policy" in cfg_dict:
    print("\nPolicy keys:", list(cfg_dict['policy'].keys()))
    if "class_name" in cfg_dict["policy"]:
        print(f"  policy.class_name = {cfg_dict['policy']['class_name']}")
    else:
        print("  MISSING: policy.class_name")
else:
    print("MISSING: policy section")

if "algorithm" in cfg_dict:
    print("\nAlgorithm keys:", list(cfg_dict['algorithm'].keys()))
    if "class_name" in cfg_dict["algorithm"]:
        print(f"  algorithm.class_name = {cfg_dict['algorithm']['class_name']}")
    else:
        print("  MISSING: algorithm.class_name")
else:
    print("MISSING: algorithm section")

print("\nFull config dict:")
print(json.dumps(cfg_dict, indent=2, default=str))

simulation_app.close()
