"""
run_test.py
===========
Config-driven simulation runner for the Bee Foraging DRL evaluation.

Usage
-----
# Run with default config:
    python run_test.py

# Specify a different config file:
    python run_test.py --config path/to/run_test_config.yaml

# Run only one model for a quick check:
    python run_test.py --models qlearning --num_scenarios 2

# Override a single timestep for debugging:
    python run_test.py --models gradient --timesteps 500
"""

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
import pickle
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

from bees_drl_utils import GlobalStateExtractor
from bees_env import BeeForagingEnv
from rl_models.bee_gradient_policy import BeeGradientAgent, ReTaskModel, GossipMessage
from rl_models.bee_policy import BeePolicyWorker, TaskManager
from bee_dqn_policy import HierarchicalAgent
from bee_flower_state import GlobalStateExtractor, BeeStateWrapper
from utils.bee_utils import Utils
from milp_linear import BeeMILP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_test")


# ─────────────────────────────────────────────────────────────────────────────
# Gossip infrastructure (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

class GossipProtocol:
    """Manages gossip communication between bees."""

    def __init__(self, num_bees: int, gossip_interval: int = 3):
        self.num_bees = num_bees
        self.gossip_interval = gossip_interval
        self.message_queues = {i: deque(maxlen=20) for i in range(num_bees)}

    def should_gossip(self, step: int) -> bool:
        return step % self.gossip_interval == 0

    def broadcast_messages(
        self,
        agents: List[BeeGradientAgent],
        current_step: int,
        bee_positions: Dict[str, Tuple[float, float]],
        bee_state: Dict,
        device,
    ):
        messages = []
        for agent in agents:
            if agent.active:
                pos = bee_positions.get(f"bee_{agent.bee_id}", (0, 0))
                msg = agent.create_gossip_message(current_step, pos, bee_state)
                messages.append(msg)
        for agent in agents:
            for msg in messages:
                if msg.sender_id != agent.bee_id:
                    agent.receive_gossip(msg)


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Loaded config from {config_path}")
    return cfg


def _resolve(value, fallback):
    """Return *value* if not None/null, otherwise *fallback*."""
    return value if value is not None else fallback


def _timestep_list(cfg: Dict) -> List[int]:
    """Build the list of timestep values from the config sweep spec."""
    ts_cfg = cfg.get("timestep_sweep", {})
    mode = ts_cfg.get("mode", "range")
    if mode == "list":
        return [int(v) for v in ts_cfg["values"]]
    start = int(ts_cfg.get("start", 100))
    stop  = int(ts_cfg.get("stop",  1000))
    step  = int(ts_cfg.get("step",  100))
    return list(range(start, stop + step, step))


def _output_path(cfg: Dict, filename: str) -> Path:
    out_dir = Path(cfg.get("persistence", {}).get("results_filename", "results_m_steps.pkl")).parent
    out_dir = Path(cfg.get("experiment", {}).get("output_dir", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    stem   = Path(filename).stem
    suffix = Path(filename).suffix
    if cfg.get("persistence", {}).get("append_timestamp", True):
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{stem}_{ts}{suffix}"
    return out_dir / filename


# ─────────────────────────────────────────────────────────────────────────────
# Shared analytics helper (identical to original _process_analytics_from_scenario)
# ─────────────────────────────────────────────────────────────────────────────

def _process_analytics_from_scenario(
    scenario, scenario_reward, time_completed, bees_scenario: dict, flower_scenario: dict
) -> Dict:
    hard_window_flowers           = sum(1 for v in flower_scenario.values() if v.hard_window)
    hard_window_flowers_harvested = sum(1 for v in flower_scenario.values() if v.harvested and v.hard_window)
    high_priority_flowers_harvested = sum(1 for v in flower_scenario.values() if v.harvested and v.priority_level == "High")
    soft_window_flowers_harvested = sum(1 for v in flower_scenario.values() if v.harvested and not v.hard_window)
    soft_window_flowers           = sum(1 for v in flower_scenario.values() if not v.hard_window)
    total_pollen_harvested        = sum(v.current_capacity for v in bees_scenario.values())
    number_flowers_harvested      = sum(1 for v in flower_scenario.values() if v.harvested)
    number_bee_terminated         = [bee.terminated for bee in bees_scenario.values()]
    total_flowers                 = soft_window_flowers + hard_window_flowers

    logger.info(
        f"Scenario {scenario} | reward={scenario_reward:.1f} | t={time_completed} | "
        f"flowers={number_flowers_harvested} | pollen={total_pollen_harvested} | "
        f"hard={hard_window_flowers_harvested}/{hard_window_flowers} | "
        f"soft={soft_window_flowers_harvested}/{soft_window_flowers}"
    )
    return {
        "number_bee_terminated":          sum(number_bee_terminated),
        "number_flowers_harvested":        number_flowers_harvested,
        "total_pollen_harvested":          total_pollen_harvested,
        "soft_window_flowers":             soft_window_flowers,
        "soft_window_flowers_harvested":   soft_window_flowers_harvested,
        "hard_window_flowers_harvested":   hard_window_flowers_harvested,
        "hard_window_flowers":             hard_window_flowers,
        "high_priority_flowers_harvested": high_priority_flowers_harvested,
        "total_flowers_scenario":          total_flowers,
    }


def _empty_results_dict(num_scenarios: int) -> Dict:
    """Allocate a fresh results accumulator for one (model, timestep) cell."""
    z  = lambda: np.zeros(num_scenarios)
    zi = lambda: np.zeros(num_scenarios, dtype=np.int32)
    return {
        "rewards":                         z(),
        "run_steps":                       zi(),
        "num_flowers":                     zi(),
        "num_bees_terminated":             zi(),
        "hard_window_flowers_scen":        zi(),
        "total_pollens_harvested_scen":    z(),
        "soft_window_flowers_scen":        zi(),
        "soft_window_flowers_harvested_scen": zi(),
        "hard_window_flowers_harvested_scen": zi(),
        "total_pollen_per_scenario":       z(),
        "high_priority_flowers_harvested": zi(),
        "objective_milp_list":             z(),
        "total_num_flowers":               zi(),
    }


def _append_scenario_results(container: Dict, idx: int, analytics: Dict,
                              scenario_reward: float, step: int,
                              total_pollen_available: float, milp_objective: float) -> None:
    """Write one scenario's metrics into the results container at position *idx*."""
    container["rewards"][idx]                         = scenario_reward
    container["run_steps"][idx]                       = step
    container["num_flowers"][idx]                     = analytics["number_flowers_harvested"]
    container["num_bees_terminated"][idx]             = analytics["number_bee_terminated"]
    container["hard_window_flowers_scen"][idx]        = analytics["hard_window_flowers"]
    container["total_pollens_harvested_scen"][idx]    = analytics["total_pollen_harvested"]
    container["soft_window_flowers_scen"][idx]        = analytics["soft_window_flowers"]
    container["soft_window_flowers_harvested_scen"][idx] = analytics["soft_window_flowers_harvested"]
    container["hard_window_flowers_harvested_scen"][idx] = analytics["hard_window_flowers_harvested"]
    container["total_pollen_per_scenario"][idx]       = total_pollen_available
    container["high_priority_flowers_harvested"][idx] = analytics["high_priority_flowers_harvested"]
    container["objective_milp_list"][idx]             = milp_objective
    container["total_num_flowers"][idx]               = analytics["total_flowers_scenario"]


def update_bee_agent(bee_vfrl_agents, bee_feature_state: Dict):
    """Sync model agent state from environment bee info."""
    agents_list, terminated_bees = [], []
    for i, agent in enumerate(bee_vfrl_agents):
        bee_id    = f"bee_{i}"
        bee_state = bee_feature_state.get(bee_id, {})
        agent.energy      = bee_state.get("energy_level", 1.0)
        agent.pollen_load = bee_state.get("pollen_collected", 0.0)
        agent.update_agent_status(bee_state.get("is_available", False))
        agent.update_task_dict(bee_state.get("assigned_task", {}))
        if not agent.active:
            terminated_bees.append(i)
        agents_list.append(agent)
    return agents_list, terminated_bees


# ─────────────────────────────────────────────────────────────────────────────
# Q-Learning / DDQN runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_q_learning_simulator(
    model_cfg: Dict,
    num_scenarios: int,
    max_step: int,
    run_index: int = 0,
) -> Dict:
    """Run the DDQN agent for *num_scenarios* scenarios and return a results dict."""

    model_path = model_cfg["model_path"]
    render     = model_cfg.get("render")
    device     = Utils.get_device()
    path       = Path(model_path)

    logger.info(f"[qlearning] Loading checkpoint: {path}")
    bee_q_network = Utils.load_model(path, device)
    metadata      = bee_q_network["metadata"]

    num_sim_bees    = int(_resolve(model_cfg.get("num_bees"),    metadata["num_bees"]))
    num_flower_task = int(_resolve(model_cfg.get("num_flowers"), metadata["num_flowers"]))
    grid_size       = int(_resolve(model_cfg.get("grid_size"),   metadata["grid_size"]))
    max_energy      = int(_resolve(model_cfg.get("max_energy_level"), 2000))
    bee_capacity    = int(_resolve(model_cfg.get("max_bee_capacity"), 100))
    embed_dim       = int(_resolve(model_cfg.get("embed_dim"),   256))
    bee_feat_dim    = int(_resolve(model_cfg.get("bee_feature_dim"), 5))
    flower_feat_dim = int(_resolve(model_cfg.get("flower_feature_dim"), 6))

    env = BeeForagingEnv(
        render_mode=render,
        num_bees=num_sim_bees,
        grid_size=grid_size,
        num_flowers=num_flower_task,
        max_steps=max_step,
        num_scenario=num_scenarios,
        dataset_type="test_data",
        max_energy_level=max_energy,
        bee_capacity=bee_capacity,
    )
    extractor   = GlobalStateExtractor(env)
    wrapped_env = BeeStateWrapper(env, extractor, num_groom_types=2)
    task_manager = TaskManager(num_flowers=num_flower_task)
    beemilp      = BeeMILP()

    dqn_agents = [
        HierarchicalAgent(
            bee_id=i,
            bee_feature_dim=bee_feat_dim,
            flower_feature_dim=flower_feat_dim,
            embed_dim=embed_dim,
            num_flowers=num_flower_task,
        )
        for i in range(num_sim_bees)
    ]
    agent_network = bee_q_network["agents"]
    for agent in dqn_agents:
        agent.goal_selector.load_state_dict(agent_network["goal_selector"])
        agent.worker.online_net.load_state_dict(agent_network["worker_online"])
        agent.epsilon = float(model_cfg.get("epsilon", 0.0))
        agent.set_eval_mode()

    container = _empty_results_dict(num_scenarios)

    for scenario_idx in range(1, num_scenarios + 1):
        env.reset(scenario_idx)
        scenario_reward   = 0.0
        current_timestep  = 1
        done              = False
        total_pollen_avail = sum(v.pollen_amount for v in env.flower_dict.values())
        milp_obj = beemilp.get_dataset_objective_value(env.filepath, scenario_idx)

        while not done:
            bee_info        = extractor.get_bee_info()
            active_agents, _ = update_bee_agent(dqn_agents, bee_info)
            actions_list    = {}

            for agent in active_agents:
                if agent.active:
                    state_dict, _ = extractor.get_global_state(
                        agent_id=agent.agent_id, current_step=current_timestep
                    )
                    action_mask = wrapped_env.get_action_mask(agent.agent_id)
                    action, _, _ = agent.predict(state_dict, action_mask)
                    actions_list[agent.agent_id] = {
                        "action": action,
                        "assigned_task": list(agent.my_tasks.keys()),
                    }
                else:
                    actions_list[agent.agent_id] = {
                        "action": 0,
                        "assigned_task": list(agent.my_tasks.keys()),
                    }

            rewards, _, _, failed_bees = wrapped_env.step(actions_list, current_timestep)
            scenario_reward += sum(rewards.values())

            for agent in active_agents:
                if failed_bees.get(agent.agent_id, False):
                    agent.update_agent_status(False)
                    task_manager.orphan_agent_tasks(agent)

            done = (failed_bees and all(failed_bees.values())) or current_timestep >= max_step
            if render == "human":
                time.sleep(float(model_cfg.get("render_delay", 0.3)))
            if not done:
                current_timestep += 1

        analytics = _process_analytics_from_scenario(
            scenario_idx, scenario_reward, current_timestep,
            env.possible_bee_agents, env.flower_dict,
        )
        _append_scenario_results(
            container, scenario_idx - 1, analytics,
            scenario_reward, current_timestep, total_pollen_avail, milp_obj,
        )

    env.close()
    return container


# ─────────────────────────────────────────────────────────────────────────────
# Policy Gradient runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_gradient_bee_simulator(
    model_cfg: Dict,
    num_scenarios: int,
    max_step: int,
    run_index: int = 0,
) -> Dict:
    """Run the policy gradient agent for *num_scenarios* scenarios."""

    model_path = model_cfg["model_path"]
    render     = model_cfg.get("render")
    device     = Utils.get_device()
    path       = Path(model_path)

    logger.info(f"[gradient] Loading checkpoint: {path}")
    gradient_policy  = Utils.load_model(path, device)
    training_config  = gradient_policy["metadata"]

    num_sim_bees    = int(_resolve(model_cfg.get("num_bees"),    training_config["num_of_bees"]))
    num_flower_task = int(_resolve(model_cfg.get("num_flowers"), training_config["num_flowers"]))
    grid_size       = int(_resolve(model_cfg.get("grid_size"),   training_config["playground_size"]))
    max_energy      = int(_resolve(model_cfg.get("max_energy_level"), 2000))
    bee_capacity    = int(_resolve(model_cfg.get("max_bee_capacity"), 100))
    embed_dim       = int(_resolve(model_cfg.get("embed_dim"),   64))
    flower_feat_dim = int(_resolve(model_cfg.get("flower_feature_dim"), 6))

    retask_model = ReTaskModel()
    retask_model.load_state_dict(gradient_policy["retask_model"])
    retask_model.eval()

    worker_policy = BeePolicyWorker(
        num_groom_types=training_config["num_groom_types"],
        num_flowers=num_flower_task,
        device=device,
    )
    worker_policy.load_state_dict(gradient_policy["worker_policy"])
    worker_policy.eval()

    coordinator_state = gradient_policy.get("coordinators")
    gossip_protocol   = GossipProtocol(num_sim_bees, gossip_interval=3)

    bee_agents = [
        BeeGradientAgent(
            bee_id=i,
            worker_policy=worker_policy,
            retask_reassignment=retask_model,
            device=device,
            grid_size=grid_size,
            flower_feature_dim=flower_feat_dim,
            max_steps=max_step,
            embed_dim=embed_dim,
        )
        for i in range(num_sim_bees)
    ]
    for agent in bee_agents:
        agent.coordinator.load_state_dict(coordinator_state)
        agent.coordinator.eval()

    env = BeeForagingEnv(
        dataset_type="test_data",
        render_mode=render,
        num_bees=num_sim_bees,
        grid_size=grid_size,
        num_flowers=num_flower_task,
        max_steps=max_step,
        num_scenario=num_scenarios,
        max_energy_level=max_energy,
        bee_capacity=bee_capacity,
    )
    extractor    = GlobalStateExtractor(env)
    env_wrapper  = BeeStateWrapper(env, num_groom_types=training_config["num_groom_types"], extractor=extractor)
    task_manager = TaskManager(num_flowers=env.num_flowers)
    beemilp      = BeeMILP()
    container    = _empty_results_dict(num_scenarios)

    for scenario in range(1, num_scenarios + 1):
        env.reset(scenario_num=scenario)
        scenario_reward   = 0.0
        step              = 1
        done              = False
        total_pollen_avail = sum(v.pollen_amount for v in env.flower_dict.values())
        milp_obj = beemilp.get_dataset_objective_value(env.filepath, scenario)

        while not done:
            bee_info = extractor.get_bee_info()
            vfrl_agents, _ = update_bee_agent(bee_agents, bee_feature_state=bee_info)
            bee_current_positions, all_bee_states = extractor.get_bees_states_current_positions(step)

            gossip_protocol.broadcast_messages(
                vfrl_agents, step, bee_current_positions, all_bee_states, device
            )

            actions_list = {}
            for agent in vfrl_agents:
                if agent.active:
                    state_dict, bee_pos = extractor.get_global_state(
                        agent_id=agent.agent_id, current_step=step
                    )
                    action_mask = env_wrapper.get_action_mask(agent.agent_id)
                    agent.perform_task_reassignment(
                        agent.agent_id, state_dict, step,
                        bee_pos, task_manager.orphaned_tasks, is_training=False,
                    )
                    action, _, _ = agent.select_hierarchical_action(
                        state_dict, step, action_mask, bee_pos,
                        task_manager.orphaned_tasks, is_training=False,
                    )
                    actions_list[agent.agent_id] = {
                        "action": action,
                        "assigned_task": agent.my_tasks.keys(),
                    }
                else:
                    actions_list[agent.agent_id] = {
                        "action": 0,
                        "assigned_task": agent.my_tasks.keys(),
                    }

            rewards, _, _, failed_bees = env_wrapper.step(actions_list, step)
            scenario_reward += sum(rewards.values())
            done = (failed_bees and all(failed_bees.values())) or step >= env.max_steps

            for agent in vfrl_agents:
                if failed_bees and failed_bees.get(agent.agent_id, False):
                    agent.update_agent_status(False)
                    task_manager.orphan_agent_tasks(agent)

            step += 1
            if render == "human":
                time.sleep(float(model_cfg.get("render_delay", 0.5)))

        analytics = _process_analytics_from_scenario(
            scenario, scenario_reward, step,
            env.possible_bee_agents, env.flower_dict,
        )
        _append_scenario_results(
            container, scenario - 1, analytics,
            scenario_reward, step, total_pollen_avail, milp_obj,
        )

    env.close()
    return container


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher: route a model key to the correct runner
# ─────────────────────────────────────────────────────────────────────────────

_RUNNERS = {
    "qlearning": _run_q_learning_simulator,
    "gradient":  _run_gradient_bee_simulator,
}


def run_simulation_for_model(
    model_name: str,
    model_cfg: Dict,
    num_scenarios: int,
    max_step: int,
    run_index: int = 0,
) -> Dict:
    """
    Run a single (model, timestep) cell and return the raw results dict.

    Raises ValueError for unknown model names so the sweep loop can skip
    or abort cleanly.
    """
    runner = _RUNNERS.get(model_name)
    if runner is None:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Supported: {list(_RUNNERS.keys())}"
        )
    logger.info(
        f"  → model={model_name}  max_step={max_step}  "
        f"scenarios={num_scenarios}  run={run_index}"
    )
    return runner(model_cfg, num_scenarios, max_step, run_index)


# ─────────────────────────────────────────────────────────────────────────────
# Full evaluation sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_full_evaluation(cfg: Dict) -> Dict:
    """
    Execute the complete timestep × model sweep defined in *cfg*.

    Returns
    -------
    results : Dict
        Structure::

            {
              model_name: {
                timestep_str: {          # e.g. "100", "200", …
                  run_0: <results_dict>,
                  run_1: <results_dict>,
                  …
                  "mean": <aggregated>,  # mean across runs
                  "std":  <aggregated>,  # std  across runs
                },
              },
            }
    """
    exp_cfg        = cfg.get("experiment", {})
    num_scenarios  = int(exp_cfg.get("num_scenarios", 10))
    num_runs       = int(exp_cfg.get("num_runs", 1))
    seed           = int(exp_cfg.get("seed", 0))
    timesteps      = _timestep_list(cfg)
    models_cfg     = cfg.get("models", {})

    if seed:
        np.random.seed(seed)
        torch.manual_seed(seed)
        logger.info(f"Random seed fixed to {seed}")

    # Determine which models are enabled
    active_models = {
        name: mc for name, mc in models_cfg.items()
        if mc.get("enabled", True) and name in _RUNNERS
    }
    if not active_models:
        raise RuntimeError("No enabled models found in config that match a runner.")

    results: Dict = {name: {} for name in active_models}

    total_cells = len(active_models) * len(timesteps) * num_runs
    cell_idx    = 0

    for model_name, model_cfg in active_models.items():
        for ts in timesteps:
            ts_key        = str(ts)
            run_results   = {}

            for run_i in range(num_runs):
                cell_idx += 1
                logger.info(
                    f"[{cell_idx}/{total_cells}] "
                    f"model={model_name}  timestep={ts}  run={run_i}"
                )
                try:
                    run_results[f"run_{run_i}"] = run_simulation_for_model(
                        model_name, model_cfg, num_scenarios, ts, run_index=run_i
                    )
                except Exception as exc:
                    logger.error(
                        f"  FAILED model={model_name} ts={ts} run={run_i}: {exc}"
                    )
                    run_results[f"run_{run_i}"] = None

            # Aggregate across runs (skip None entries from failures)
            valid_runs = [v for v in run_results.values() if v is not None]
            if valid_runs:
                run_results["mean"] = _aggregate_runs(valid_runs, np.mean)
                run_results["std"]  = _aggregate_runs(valid_runs, np.std)
            else:
                run_results["mean"] = None
                run_results["std"]  = None

            results[model_name][ts_key] = run_results

    return results


def _aggregate_runs(run_list: List[Dict], agg_fn) -> Dict:
    """
    Apply *agg_fn* (e.g. np.mean / np.std) element-wise across a list
    of identically-structured results dicts.
    """
    keys = run_list[0].keys()
    out  = {}
    for k in keys:
        arrays = [r[k] for r in run_list if isinstance(r.get(k), np.ndarray)]
        if arrays:
            out[k] = agg_fn(np.stack(arrays, axis=0), axis=0)
        else:
            out[k] = run_list[0].get(k)  # Non-array values: take first
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_results(results: Dict, cfg: Dict) -> Path:
    """Pickle results and return the path written."""
    persist_cfg = cfg.get("persistence", {})
    filename    = persist_cfg.get("results_filename", "results_m_steps.pkl")
    out_path    = _output_path(cfg, filename)
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    logger.info(f"Results saved → {out_path}")
    return out_path


def load_results(path: str) -> Dict:
    """Load a previously saved results pickle."""
    with open(path, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Bee Foraging DRL Evaluation Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config", default="run_test_config.yaml",
        help="Path to YAML configuration file.",
    )
    p.add_argument(
        "--models", nargs="+", default=None,
        help="Restrict evaluation to a subset of model names, e.g. --models qlearning.",
    )
    p.add_argument(
        "--timesteps", nargs="+", type=int, default=None,
        help="Override timestep sweep with explicit values, e.g. --timesteps 200 500 1000.",
    )
    p.add_argument(
        "--num_scenarios", type=int, default=None,
        help="Override num_scenarios from config.",
    )
    p.add_argument(
        "--num_runs", type=int, default=None,
        help="Override num_runs from config.",
    )
    p.add_argument(
        "--load_results", default=None,
        help="Skip simulation; load existing .pkl file and go straight to plotting.",
    )
    p.add_argument(
        "--no_save", action="store_true",
        help="Do not write results to disk.",
    )
    return p


def main():
    parser = _build_parser()
    args   = parser.parse_args()

    cfg = load_config(args.config)

    # CLI overrides
    if args.models:
        for name in list(cfg["models"].keys()):
            if name not in args.models:
                cfg["models"][name]["enabled"] = False

    if args.timesteps:
        cfg["timestep_sweep"] = {"mode": "list", "values": args.timesteps}

    if args.num_scenarios:
        cfg["experiment"]["num_scenarios"] = args.num_scenarios

    if args.num_runs:
        cfg["experiment"]["num_runs"] = args.num_runs

    # ── Run or load ──────────────────────────────────────────
    if args.load_results:
        logger.info(f"Loading existing results from {args.load_results}")
        results = load_results(args.load_results)
    else:
        results = run_full_evaluation(cfg)
        if not args.no_save:
            save_results(results, cfg)

    # ── Plot ─────────────────────────────────────────────────
    from evaluator_2 import plot_episode_results
    # evaluator expects the legacy flat structure; build a compatibility view
    plot_episode_results(_build_plot_input(results))


def _build_plot_input(results: Dict) -> Dict:
    """
    Convert the multi-run results structure into the flat format expected
    by the evaluator's plot_episode_results():

        { model_name: { timestep_str: <mean_results_dict> } }
    """
    flat = {}
    for model_name, ts_dict in results.items():
        flat[model_name] = {}
        for ts_key, run_dict in ts_dict.items():
            mean = run_dict.get("mean")
            if mean is None and "run_0" in run_dict:
                mean = run_dict["run_0"]  # Fallback for num_runs=1
            if mean is not None:
                flat[model_name][ts_key] = mean
    return flat


if __name__ == "__main__":
    main()