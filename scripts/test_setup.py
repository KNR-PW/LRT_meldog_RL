#!/usr/bin/env python3
"""Test script to verify meldog_rl setup.

Run this BEFORE Isaac Lab to check imports work:
    python scripts/test_setup.py

Run this WITH Isaac Lab to check full setup:
    isaaclab -p scripts/test_setup.py --with-isaaclab
"""

import sys
import os

# Add source to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))

def test_imports():
    """Test basic imports work (no Isaac Lab needed)."""
    print("=" * 50)
    print("TEST 1: Basic imports (no Isaac Lab)")
    print("=" * 50)
    
    try:
        # These should work without Isaac Lab
        from meldog_rl.utils.naming import make_output_name, make_locomotion_log_dir
        print("✓ meldog_rl.utils.naming imports OK")
        
        # Test naming utility
        name = make_output_name("LM", "rough", "sim")
        print(f"✓ Naming utility works: {name}")
        
        log_dir = make_locomotion_log_dir("rough", "sim")
        print(f"✓ Log dir utility works: {log_dir}")
        
    except Exception as e:
        print(f"✗ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_configs():
    """Test config classes can be instantiated (requires Isaac Lab)."""
    print("\n" + "=" * 50)
    print("TEST 2: Config instantiation (requires Isaac Lab)")
    print("=" * 50)
    
    try:
        from meldog_rl.envs.configs.simulation.rough_cfg import RoughSimCfg
        cfg = RoughSimCfg()
        print(f"✓ RoughSimCfg instantiated")
        print(f"  - episode_length_s: {cfg.episode_length_s}")
        print(f"  - action_scale: {cfg.action_scale}")
        print(f"  - num_envs: {cfg.scene.num_envs}")
        
        from meldog_rl.envs.configs.sim2real.rough_cfg import RoughRealCfg
        cfg2 = RoughRealCfg()
        print(f"✓ RoughRealCfg instantiated")
        
        from meldog_rl.envs.configs.dataset.rough_cfg import RoughDatasetCfg
        cfg3 = RoughDatasetCfg()
        print(f"✓ RoughDatasetCfg instantiated")
        print(f"  - Cameras enabled: {cfg3.tiled_camera_front is not None}")
        
    except ImportError as e:
        print(f"⚠ Isaac Lab not available (run with: isaaclab -p scripts/test_setup.py)")
        print(f"  Error: {e}")
        return None  # Can't test without Isaac Lab
    except Exception as e:
        print(f"✗ Config error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_task_registration():
    """Test gymnasium task registration (requires Isaac Lab)."""
    print("\n" + "=" * 50)
    print("TEST 3: Task registration (requires Isaac Lab)")
    print("=" * 50)
    
    try:
        import gymnasium as gym
        
        # Import to trigger registration
        from meldog_rl import envs
        
        # Check tasks are registered
        expected_tasks = [
            "Meldog-RL-Locomotion-Flat-Sim-v0",
            "Meldog-RL-Locomotion-Rough-Sim-v0",
            "Meldog-RL-Locomotion-Rough-Real-v0",
            "Meldog-RL-Dataset-Rough-v0",
        ]
        
        for task in expected_tasks:
            spec = gym.spec(task)
            print(f"✓ {task} registered")
            
            # Check config entry points exist
            env_cfg_cls = spec.kwargs.get("env_cfg_entry_point")
            agent_cfg_cls = spec.kwargs.get("rsl_rl_cfg_entry_point")
            print(f"  - env_cfg: {env_cfg_cls.__name__}")
            print(f"  - agent_cfg: {agent_cfg_cls.__name__}")
            
            # Try instantiating
            env_cfg = env_cfg_cls()
            agent_cfg = agent_cfg_cls()
            print(f"  - Both configs instantiate OK")
        
    except ImportError as e:
        print(f"⚠ Isaac Lab not available")
        print(f"  Error: {e}")
        return None
    except Exception as e:
        print(f"✗ Registration error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_usd_path():
    """Test USD path is set correctly."""
    print("\n" + "=" * 50)
    print("TEST 4: USD Path Check")
    print("=" * 50)
    
    # Check environment variable
    usd_path = os.environ.get("MELDOG_USD_PATH")
    if usd_path:
        print(f"✓ MELDOG_USD_PATH set: {usd_path}")
        if os.path.exists(usd_path):
            print(f"✓ File exists!")
        else:
            print(f"✗ File NOT found!")
            return False
    else:
        print("⚠ MELDOG_USD_PATH not set, will use fallback")
        
        # Check fallback locations
        fallbacks = [
            "/home/frydjak/Downloads/Meldog-1.4-no-ground-plane.usd",
            "assets/robots/meldog/Meldog-1.4-no-ground-plane.usd",
        ]
        
        found = False
        for path in fallbacks:
            if os.path.exists(path):
                print(f"✓ Found USD at fallback: {path}")
                found = True
                break
        
        if not found:
            print("✗ USD not found at any fallback location!")
            print("  Please either:")
            print("    1. Set MELDOG_USD_PATH=/path/to/your/robot.usd")
            print("    2. Copy USD to assets/robots/meldog/Meldog-1.4-no-ground-plane.usd")
            return False
    
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-isaaclab", action="store_true", 
                        help="Run full Isaac Lab tests (requires simulator)")
    args = parser.parse_args()
    
    print("\n" + "=" * 50)
    print("MELDOG_RL SETUP TEST")
    print("=" * 50)
    
    results = {}
    
    # Test 1: Always run (no Isaac Lab needed)
    results["basic_imports"] = test_imports()
    
    # Test 4: USD path check (no Isaac Lab needed)
    results["usd_path"] = test_usd_path()
    
    # Test 2-3: Need Isaac Lab
    if args.with_isaaclab:
        results["configs"] = test_configs()
        results["registration"] = test_task_registration()
    else:
        print("\n" + "=" * 50)
        print("SKIPPING Isaac Lab tests (run with --with-isaaclab)")
        print("=" * 50)
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for test, result in results.items():
        if result is True:
            print(f"✓ {test}: PASSED")
        elif result is False:
            print(f"✗ {test}: FAILED")
        else:
            print(f"⚠ {test}: SKIPPED")
    
    # Exit code
    if any(r is False for r in results.values()):
        sys.exit(1)
    print("\nBasic tests passed!")
