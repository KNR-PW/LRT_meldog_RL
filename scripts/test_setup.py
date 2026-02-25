#!/usr/bin/env python3
"""Test meldog_rl setup."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))

def test_imports():
    """Test basic imports."""
    print("\nTEST 1: Basic imports")

    try:
        from meldog_rl.utils.naming import make_output_name, make_locomotion_log_dir
        print("  meldog_rl.utils.naming imports OK")

        name = make_output_name("LM", "rough", "sim")
        print(f"  Naming utility works: {name}")

        log_dir = make_locomotion_log_dir("rough", "sim")
        print(f"  Log dir utility works: {log_dir}")

    except Exception as e:
        print(f"  Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def test_configs():
    """Test config instantiation (requires Isaac Lab)."""
    print("\nTEST 2: Config instantiation")

    try:
        from meldog_rl.envs.configs.simulation.rough_cfg import RoughSimCfg
        cfg = RoughSimCfg()
        print("  RoughSimCfg instantiated")
        print(f"    episode_length_s: {cfg.episode_length_s}")
        print(f"    action_scale: {cfg.action_scale}")
        print(f"    num_envs: {cfg.scene.num_envs}")

        from meldog_rl.envs.configs.sim2real.rough_cfg import RoughRealCfg
        cfg2 = RoughRealCfg()
        print("  RoughRealCfg instantiated")

        from meldog_rl.envs.configs.dataset.rough_cfg import RoughDatasetCfg
        cfg3 = RoughDatasetCfg()
        print("  RoughDatasetCfg instantiated")
        print(f"    Cameras enabled: {cfg3.tiled_camera_front is not None}")

    except ImportError as e:
        print(f"  Isaac Lab not available (run with --with-isaaclab)")
        print(f"  Error: {e}")
        return None
    except Exception as e:
        print(f"  Config error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def test_task_registration():
    """Test task registration (requires Isaac Lab)."""
    print("\nTEST 3: Task registration")

    try:
        import gymnasium as gym
        from meldog_rl import envs

        expected_tasks = [
            "Meldog-RL-Locomotion-Flat-Sim-v0",
            "Meldog-RL-Locomotion-Rough-Sim-v0",
            "Meldog-RL-Locomotion-Rough-Real-v0",
            "Meldog-RL-Dataset-Rough-v0",
        ]

        for task in expected_tasks:
            spec = gym.spec(task)
            print(f"  {task} registered")

            env_cfg_cls = spec.kwargs.get("env_cfg_entry_point")
            agent_cfg_cls = spec.kwargs.get("rsl_rl_cfg_entry_point")
            print(f"    env_cfg: {env_cfg_cls.__name__}")
            print(f"    agent_cfg: {agent_cfg_cls.__name__}")

            env_cfg = env_cfg_cls()
            agent_cfg = agent_cfg_cls()
            print("    Both configs instantiate OK")

    except ImportError as e:
        print(f"  Isaac Lab not available")
        print(f"  Error: {e}")
        return None
    except Exception as e:
        print(f"  Registration error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def test_usd_path():
    """Test USD path."""
    print("\nTEST 4: USD Path")

    usd_path = os.environ.get("MELDOG_USD_PATH")
    if usd_path:
        print(f"  MELDOG_USD_PATH set: {usd_path}")
        if os.path.exists(usd_path):
            print("  File exists")
        else:
            print("  File NOT found")
            return False
    else:
        print("  MELDOG_USD_PATH not set, checking fallbacks")

        fallbacks = [
            "/home/frydjak/Downloads/Meldog-1.4-no-ground-plane.usd",
            "assets/robots/meldog/Meldog-1.4-no-ground-plane.usd",
        ]

        found = False
        for path in fallbacks:
            if os.path.exists(path):
                print(f"  Found USD at: {path}")
                found = True
                break

        if not found:
            print("  USD not found")
            print("  Set MELDOG_USD_PATH or copy USD to assets/robots/meldog/")
            return False

    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-isaaclab", action="store_true",
                        help="Run full Isaac Lab tests")
    args = parser.parse_args()

    print("\nMELDOG_RL SETUP TEST")

    results = {}

    results["basic_imports"] = test_imports()
    results["usd_path"] = test_usd_path()

    if args.with_isaaclab:
        results["configs"] = test_configs()
        results["registration"] = test_task_registration()
    else:
        print("\nSkipping Isaac Lab tests (run with --with-isaaclab)")

    print("\nSummary:")
    for test, result in results.items():
        if result is True:
            print(f"  {test}: PASSED")
        elif result is False:
            print(f"  {test}: FAILED")
        else:
            print(f"  {test}: SKIPPED")

    if any(r is False for r in results.values()):
        sys.exit(1)
    print("\nTests passed")
