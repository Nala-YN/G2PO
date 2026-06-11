import os
import numpy as np
import time
import logging
import concurrent.futures
from datetime import datetime
from agent_system.environments.env_manager import *
from omegaconf import OmegaConf
#api api.deepseek.com
import litellm
litellm.set_verbose = False
litellm.suppress_debug_info = True
logging.getLogger('LiteLLM').setLevel(logging.WARNING)

def build_env(env_name, env_num=1):
    group_n = 1
    config = OmegaConf.load("verl/trainer/config/ppo_trainer.yaml")
    if env_name == "alfworld":
        # Test AlfWorldEnvironmentManager
        from agent_system.environments.env_package.alfworld import alfworld_projection
        from agent_system.environments.env_package.alfworld import build_alfworld_envs
        alf_config_path = os.path.join(os.path.dirname(__file__), '../../agent_system/environments/env_package/alfworld/configs/config_tw.yaml')
        env_kwargs = {
            'eval_dataset': "eval_in_distribution", # 'eval_in_distribution' or 'eval_out_of_distribution'
        }
        resources_per_worker = {"num_cpus": 0.05, "num_gpus": 0.0}
        envs = build_alfworld_envs(alf_config_path, seed=1, env_num=env_num, group_n=group_n, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        env_manager = AlfWorldEnvironmentManager(envs, alfworld_projection, config)
    else:
        raise ValueError(f"Unsupported environment name: {env_name}")
    
    return env_manager

class Agent:
    def __init__(self, model_name="openai/coder"):
        self.model_name = model_name
        self.api_base = "http://localhost:8000/v1"
        self.api_key = "mykey"
    """ def __init__(self, model_name="deepseek/deepseek-chat"):
        self.model_name = model_name
        self.api_base = "https://api.deepseek.com"
        self.api_key = "sk-f051af6b41614647b26067206c2899ad" """
    def get_action_from_gpt(self, obs):
        response = litellm.completion(
            model=self.model_name,
            messages=[
                {
                    "role": "user", 
                    "content": obs
                }
            ],
            api_base=self.api_base,
            api_key=self.api_key,
            temperature=0.4,
            n=1,
            stop=None
        )
        
        action = response.choices[0].message.content.strip()
        return action

if __name__ == "__main__":
    # -------- logging ----------
    os.makedirs("logs/alfworld", exist_ok=True)
    log_fp = os.path.join(
        "logs/alfworld", f"run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[logging.FileHandler(log_fp, encoding="utf-8"), logging.StreamHandler()],
    )

    # -------- Parameters ----------
    max_steps = 50
    env_num = 128
    test_times = 3
    env_name = "alfworld" 

    # Keywords for 6 subtasks
    TASKS = [
        "pick_and_place",
        "pick_two_obj_and_place",
        "look_at_obj_in_light",
        "pick_heat_then_place_in_recep",
        "pick_cool_then_place_in_recep",
        "pick_clean_then_place_in_recep",
    ]

    # -------- Environment and agent setup ----------
    env_manager = build_env(env_name, env_num)
    agent = Agent()

    # Accumulated statistics
    overall_success_rates = []         # Overall success per round
    task_success_history = defaultdict(list)  # Subtask success per round

    # ======================= Main Loop =======================
    for test_idx in range(test_times):
        logging.info(f"\n========== Start test {test_idx} ==========")
        start_time = time.time()
        
        kwargs = {}
        obs, infos = env_manager.reset(kwargs)
        env_dones = [False] * env_num

        # Statistics for single round
        overall_success_this_round = np.zeros(env_num, dtype=bool)
        task_success_cnt = defaultdict(int)
        task_total_cnt = defaultdict(int)

        for step_idx in range(max_steps):
            logging.info(f"Step {step_idx}; Dones ({np.array(env_dones).sum().item()}/{env_num}); SR {overall_success_this_round.mean().item()}")

            # --- Assemble actions ---
            actions = ["None"] * env_num
            active_indices = [i for i in range(env_num) if not env_dones[i]]
            
            if active_indices:
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(active_indices), 128)) as executor:
                    # Submit all tasks
                    future_to_idx = {
                        executor.submit(agent.get_action_from_gpt, obs["text"][i]): i 
                        for i in active_indices
                    }
                    
                    # Collect results as they complete
                    for future in concurrent.futures.as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        try:
                            actions[idx] = future.result()
                        except Exception as exc:
                            logging.error(f"Generated an exception for env {idx}: {exc}")
                            actions[idx] = "look" # Fallback action or handle appropriately
            if len(actions)>=2:
                print(actions[0], actions[1])
            # --- Environment stepping ---
            obs, rewards, dones, infos = env_manager.step(actions)

            # --- Determine endings and successes ---
            for i in range(env_num):
                if env_dones[i]:
                    continue

                if dones[i]:
                    env_dones[i] = True
                    won = bool(infos[i].get("won", False))
                    overall_success_this_round[i] = won

                    # Parse task type
                    gamefile = infos[i].get("extra.gamefile", "")
                    matched = False
                    for task in TASKS:
                        if task in gamefile:
                            task_total_cnt[task] += 1
                            if won:
                                task_success_cnt[task] += 1
                            matched = True
                            break
                    if not matched:
                        # Unrecognized tasks are also counted in total
                        task_total_cnt["other"] += 1
                        if won:
                            task_success_cnt["other"] += 1

            if all(env_dones):
                logging.info("All environments finished early!")
                break

        # -------- Single round results --------
        round_success_rate = overall_success_this_round.mean()
        overall_success_rates.append(round_success_rate)

        logging.info(f"Test {test_idx} overall success: {round_success_rate:.4f}")

        for task in TASKS + ["other"]:
            if task_total_cnt.get(task, 0) > 0:
                rate = task_success_cnt[task] / task_total_cnt[task]
                task_success_history[task].append(rate)
                logging.info(
                    f"    {task:<35s}: {rate:.4f} "
                    f"({task_success_cnt[task]}/{task_total_cnt[task]})"
                )

        logging.info(
            f"Test {test_idx} time elapsed: {time.time() - start_time:.2f}s\n"
        )

    # ======================= Final Summary =======================
    logging.info("=============== Final Summary ===============")
    logging.info(
        f"Total tests: {test_times} | Envs / test: {env_num} | Total envs: {env_num * test_times}"
    )
    logging.info(
        f"Overall success avg ± std: "
        f"{np.mean(overall_success_rates):.4f} ± {np.std(overall_success_rates):.4f}"
    )

    for task in TASKS + ["other"]:
        if task_success_history.get(task):
            logging.info(
                f"{task:<35s}: "
                f"{np.mean(task_success_history[task]):.4f} ± "
                f"{np.std(task_success_history[task]):.4f}"
            )
