# Meldog RL - Gemini Assistant Skills & Environment Setup

## 1. Running Isaac Lab Commands
The user's setup requires a very specific initialization sequence to expose the `isaaclab` module and properly bind the Vulkan/Python environment variables. **Do not run standard `python` or `conda run`**.

Whenever you need to run a python script (e.g., for training or evaluation), you **must** use the `run_command` tool and wrap your execution in this exact `bash -c` string:

```bash
bash -c 'source ~/.bashrc; eval "$(conda shell.bash hook)"; conda activate isaac-sim; cd /home/frydjak/IsaacLab; source _isaac_sim/setup_conda_env.sh; cd /home/frydjak/KNR/Meldog/LRT_meldog_RL; <YOUR_PYTHON_COMMAND_HERE>'
```

**Example:**
To run training:
```bash
bash -c 'source ~/.bashrc; eval "$(conda shell.bash hook)"; conda activate isaac-sim; cd /home/frydjak/IsaacLab; source _isaac_sim/setup_conda_env.sh; cd /home/frydjak/KNR/Meldog/LRT_meldog_RL; python scripts/locomotion/train_locomotion.py --task Meldog-RL-Locomotion-Rough-Sim-v0 --num_envs 2 --max_iterations 10'
```

## 2. File Structure Assumptions
*   **Locomotion Scripts:** `scripts/locomotion/`
*   **Perception Scripts:** `scripts/perception/`
*   **Env Configs:** `source/meldog_rl/envs/configs/`
*   **Registered Envs:** `source/meldog_rl/envs/__init__.py`
*   **Perception Models:** `source/meldog_rl/models/perception/`

## 3. General Rules & Documentation
*   **Manual Maintenance:** The file `manual.md` serves as the comprehensive user manual for this framework. You **MUST** update `manual.md` whenever you introduce new features, scripts, CLI tools, or make fundamental changes to how the repository is structured or used. Ensure the documentation stays perfectly in sync with the codebase.

## 4. Agentic Workflow & Shared Memory
To enable seamless cross-agent collaboration (e.g., between Gemini and Claude) and provide version-controlled rationale for code changes, all AI planning, task tracking, and experiment results must be stored directly in the repository.

*   **Active Tasks & Experiments:** Create tracking files directly in the `agentic/` directory. You must use descriptive names prepended with the current date, e.g., `YYYY-MM-DD_Task_descriptive_name.md` or `YYYY-MM-DD_Experiment_results.md`.
*   **Archiving:** Once a task is finished or an experiment concludes, you **MUST** move the corresponding file into `agentic/archive/`.
*   **Context Management:** When assessing the current state of the project, agents should read the active `.md` files in the root of `agentic/`. **DO NOT** indiscriminately read the contents of `agentic/archive/` as this will blow out your context window. Only read archived files if specifically instructed to review past historical data.
