import os
import numpy as np
import time
import logging
import concurrent.futures
from datetime import datetime
from agent_system.environments.env_manager import *
from openai import OpenAI
from omegaconf import OmegaConf

import litellm
litellm.set_verbose = False
litellm.suppress_debug_info = True
logging.getLogger('LiteLLM').setLevel(logging.WARNING)

def build_env(env_name, env_num=1):
    group_n = 1
    config = OmegaConf.load("verl/trainer/config/ppo_trainer.yaml")
    if "webshop" in env_name:
        from agent_system.environments.env_package.webshop import build_webshop_envs, webshop_projection
        file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle_1000.json')
        attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2_1000.json')
        env_kwargs = {
                    'observation_mode': 'text', 
                    'num_products': None, 
                    'human_goals': False,
                    'file_path': file_path,
                    'attr_path': attr_path
                    }
        
        config.env.seed=0
        config.data.val_batch_size=env_num
        resources_per_worker = {"num_cpus": 0.05, "num_gpus": 0.0}
        _val_envs = build_webshop_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_kwargs=None, resources_per_worker=resources_per_worker)

        projection_f = partial(webshop_projection)
        val_envs = WebshopEnvironmentManager(_val_envs, projection_f, config)
        import time
        time.sleep((config.data.train_batch_size * group_n + config.data.val_batch_size) * 0.1) # wait for the envs to be ready
        return val_envs
    else:
        raise ValueError(f"Unsupported environment name: {env_name}")
    
    return env_manager

class Agent:
    def __init__(self, model_name="deepseek/Deepseek-V3.2"):
        self.model_name = model_name
        self.api_base = "https://songm-maqi2ez1-eastus2.services.ai.azure.com/openai/v1/"
        self.api_key = "8pfoDncPJ1pNv4eTNx6YDISijSIkjvt88ao8yanYT30kbZifbbKRJQQJ99BEACHYHv6XJ3w3AAAAACOG1VXY"
        self.max_retries = 100
        self.retry_base_delay = 1.0
        
    def get_action_from_gpt(self, obs):
        last_exception = None
        for attempt in range(1, self.max_retries + 1):
            try:
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
                    n=1,
                    stop=None
                )

                action = response.choices[0].message.content.strip()
                return action
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    sleep_s = self.retry_base_delay * attempt
                    logging.warning(
                        f"LLM request failed (attempt {attempt}/{self.max_retries}): {e}. "
                        f"Retrying in {sleep_s:.1f}s..."
                    )
                    time.sleep(sleep_s)

        raise RuntimeError(
            f"LLM request failed after {self.max_retries} attempts"
        ) from last_exception

if __name__ == "__main__":
    # -------- logging ----------
    os.makedirs("logs/webshop", exist_ok=True)
    log_fp = os.path.join(
        "logs/webshop", f"run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[logging.FileHandler(log_fp, encoding="utf-8"), logging.StreamHandler()],
    )

    # -------- Parameters ----------
    max_steps = 15
    env_num = 96
    test_times = 1
    env_name = "webshop" 

    # -------- Environment and agent setup ----------
    env_manager = build_env(env_name, env_num)
    agent = Agent()

    # Accumulated statistics
    overall_success_rates = []         # Overall success per round
    overall_test_scores=[]
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
        overall_test_score_this_round= np.zeros(env_num, dtype=float)
        task_success_cnt = defaultdict(int)
        task_total_cnt = defaultdict(int)

        for step_idx in range(max_steps):
            logging.info(f"Step {step_idx}; Dones ({np.array(env_dones).sum().item()}/{env_num}); SR {overall_success_this_round.mean().item()}")

            # --- Assemble actions ---
            actions = ["None"] * env_num
            active_indices = [i for i in range(env_num) if not env_dones[i]]
            
            if active_indices:
                logging.info(obs["text"][0])
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(active_indices), 2)) as executor:
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
                            actions[idx] = "None"
            
            if len(actions)>=2:
                logging.info(f"Sample actions: {actions[0]}, {actions[1]}, ...")
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
                    overall_test_score_this_round[i] = infos[i]["task_score"]

            if all(env_dones):
                logging.info("All environments finished early!")
                break

        # -------- Single round results --------
        round_success_rate = overall_success_this_round.mean()
        round_test_score = overall_test_score_this_round.mean()
        overall_success_rates.append(round_success_rate)
        overall_test_scores.append(round_test_score)
        logging.info(f"Test {test_idx} overall success: {round_success_rate:.4f}, average test score: {round_test_score:.4f}")

        logging.info(
            f"Test {test_idx} time elapsed: {time.time() - start_time:.2f}s\n"
        )
    #0.1562, average test score: 0.3166
    # ======================= Final Summary =======================
    logging.info("=============== Final Summary ===============")
    logging.info(
        f"Total tests: {test_times} | Envs / test: {env_num} | Total envs: {env_num * test_times}"
    )
    logging.info(
        f"Overall success avg ± std: "
        f"{np.mean(overall_success_rates):.4f} ± {np.std(overall_success_rates):.4f}"
    )

    logging.info(
        f"Overall test score avg ± std: "
        f"{np.mean(overall_test_scores):.4f} ± {np.std(overall_test_scores):.4f}"
    )
