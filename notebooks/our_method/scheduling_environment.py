"""Task-aware scheduling environment shared by the tabular RL methods."""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class Envir:
    """Task-aware scheduling environment for reinforcement-learning agents.

    Key modeling choices:
    - Observation contains per-core utilization + normalized features of the CURRENT task.
    - The environment owns the episode task order, so the returned observation always matches
      the task that the next action will schedule.
    - A joint action selects (primary_core, VF level, replica setting).
    - Replica semantics are STANDBY/backup semantics: replicas do not add normal CPU utilization
      merely because they exist. The primary task alone updates the utilization state.
    - Replica count affects reliability and the load-dependent replica penalty. That
      penalty is the model's abstraction for the increasing cost/overlap pressure of deferred
      backup scheduling near deadlines at high load.
    - Capacity-infeasible primary placements are masked before sampling.

    Model scope:
    The assignment model does not include an event timeline, fault-arrival trace, or explicit
    replica start times. The assignment environment therefore does not infer per-replica execution events.
    Fault-triggered backup execution and deadline-window overlap require explicit timing events.
    Replica pressure is represented by a load-dependent penalty rather than by permanently
    charging replica utilization.
    """

    TASK_FEATURE_KEYS = (
        "initial_exec_task_time",
        "deadlines",
        "util",
        "peak",
        "alpha_C",
        "I0_exp_term",
        "base_util",
    )

    def __init__(
        self,
        initial_voltage,
        initial_frequency,
        cores,
        main_tasks,
        alpha,
        util_load,
        capacity_limit=1.1,
        tsp=1.5,
        rep_max=10,
    ):
        self.initial_voltage = float(initial_voltage)
        self.initial_frequency = float(initial_frequency)
        self.cores = int(cores)
        self.main_tasks = int(main_tasks)
        self.alpha = tuple(alpha)
        self.load = float(util_load)
        self.capacity_limit = float(capacity_limit)
        self.TSP = float(tsp)
        self.rep_max = int(rep_max)
        self.threshold_load = 0.7

        self.vf_levels = {
            0: {"frequency": 800e6, "voltage": 0.9},
            1: {"frequency": 1.1e9, "voltage": 0.95},
            2: {"frequency": 1.4e9, "voltage": 1.0},
            3: {"frequency": 1.7e9, "voltage": 1.05},
            4: {"frequency": 2.0e9, "voltage": 1.1},
        }
        self.n_vf = len(self.vf_levels)
        self.n_rep = self.rep_max + 1
        self.n_actions = self.cores * self.n_vf * self.n_rep

        # Reliability parameters for the assignment model.
        self.lambda_0 = 1e-7
        self.Delta = 1.0
        self.V_max = self.vf_levels[max(self.vf_levels)]["voltage"]

        self.data: List[Dict] = []
        self.train_data: List[Dict] = []
        self.feature_min: Optional[np.ndarray] = None
        self.feature_max: Optional[np.ndarray] = None

        self.state = np.zeros(self.cores, dtype=np.float32)
        self.system_tasks: List[float] = []
        self.task_order: List[int] = []
        self.task_cursor = 0
        self.current_task_idx: Optional[int] = None
        self.rng = np.random.default_rng()

        # core utilizations + current-task features + [episode_progress, running_reliability]
        self.obs_dim = self.cores + len(self.TASK_FEATURE_KEYS) + 2

    # ------------------------------------------------------------------
    # Domain helpers
    # ------------------------------------------------------------------
    def fault_rate(self, V):
        return self.lambda_0 * 10 ** ((self.V_max - V) / self.Delta)

    def task_reliability(self, t, V):
        return float(np.exp(-self.fault_rate(V) * t))

    def total_task_reliability(self, task_tuple):
        n, exec_time, voltage = task_tuple
        n = int(n)
        if n <= 0:
            return 0.0
        single = self.task_reliability(exec_time, voltage)
        # Reliability of at least one successful copy.
        return float(1.0 - (1.0 - single) ** n)

    def system_reliability(self, task_total_reliabilities):
        if not task_total_reliabilities:
            return 1.0
        return float(np.mean(task_total_reliabilities))

    def power_sharp(self, x, n=1):
        return float(2 * (x ** n) - 1)

    # ------------------------------------------------------------------
    # Data and feature normalization
    # ------------------------------------------------------------------
    def read_data(self, input_files):
        """Load one task from the first row of each CSV, using the first row of each task CSV."""
        self.data = []
        for file in input_files:
            task_name = os.path.basename(file).replace(".csv", "")
            task_data = pd.read_csv(file)
            if task_data.empty:
                raise ValueError(f"Task CSV is empty: {file}")

            row = task_data.iloc[0]
            required = ["#", "static", "avg_power", "alpha_C", "I0_exp_term", "util", "peak", "deadline", "min"]
            missing = [c for c in required if c not in task_data.columns]
            if missing:
                raise ValueError(f"Missing columns in {file}: {missing}")

            initial_exec_task_time = float(row["#"])
            deadline = float(row["deadline"])
            if deadline <= 0:
                raise ValueError(f"deadline must be positive in {file}")

            self.data.append({
                "name": task_name,
                "alpha_C": float(row["alpha_C"]),
                "I0_exp_term": float(row["I0_exp_term"]),
                "initial_exec_task_time": initial_exec_task_time,
                "deadlines": deadline,
                "util": float(row["util"]),
                "peak": float(row["peak"]),
                "min": float(row["min"]),
                "power": float(row["avg_power"]),
                "static": float(row["static"]),
                "base_util": initial_exec_task_time / deadline,
            })

        self.train_data = list(self.data)
        if not self.train_data:
            raise ValueError("No tasks were loaded.")
        self.main_tasks = len(self.train_data)
        self._fit_task_normalizer()
        return self.train_data

    def _raw_task_features(self, task_idx):
        t = self.train_data[int(task_idx)]
        return np.array([t[k] for k in self.TASK_FEATURE_KEYS], dtype=np.float32)

    def _fit_task_normalizer(self):
        x = np.stack([self._raw_task_features(i) for i in range(len(self.train_data))])
        self.feature_min = np.min(x, axis=0)
        self.feature_max = np.max(x, axis=0)

    def normalized_task_features(self, task_idx):
        if self.feature_min is None or self.feature_max is None:
            raise RuntimeError("Call read_data() before requesting observations.")
        x = self._raw_task_features(task_idx)
        span = self.feature_max - self.feature_min
        safe_span = np.where(span > 1e-12, span, 1.0)
        z = (x - self.feature_min) / safe_span
        z = np.where(span > 1e-12, z, 0.0)
        return np.clip(z, 0.0, 1.0).astype(np.float32)

    def print_task_details(self):
        for idx, task in enumerate(self.train_data):
            peak_i = self.peak_power(idx, vf_select=max(self.vf_levels))
            print(
                f"Task {idx:02d} | {task['name']} | baseline/max-VF peak={peak_i:.6f} "
                f"| deadline={task['deadlines']} | exec={task['initial_exec_task_time']} "
                f"| base_util={task['base_util']:.4f}"
            )
        print(f"\nTotal tasks: {len(self.train_data)}")

    # ------------------------------------------------------------------
    # Observation / episode management
    # ------------------------------------------------------------------
    def _running_reliability(self):
        return self.system_reliability(self.system_tasks)

    def _observation_for(self, task_idx):
        core_part = np.clip(self.state / self.capacity_limit, 0.0, 1.5).astype(np.float32)
        task_part = self.normalized_task_features(task_idx)
        progress = float(self.task_cursor / max(1, len(self.task_order)))
        context = np.array([progress, self._running_reliability()], dtype=np.float32)
        return np.concatenate([core_part, task_part, context]).astype(np.float32)

    def _terminal_observation(self):
        core_part = np.clip(self.state / self.capacity_limit, 0.0, 1.5).astype(np.float32)
        task_part = np.zeros(len(self.TASK_FEATURE_KEYS), dtype=np.float32)
        context = np.array([1.0, self._running_reliability()], dtype=np.float32)
        return np.concatenate([core_part, task_part, context]).astype(np.float32)

    def reset(self, *, seed=None, shuffle_tasks=True):
        if not self.train_data:
            raise RuntimeError("No task data loaded. Call read_data() first.")
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.state = np.zeros(self.cores, dtype=np.float32)
        self.system_tasks = []
        self.task_order = list(range(len(self.train_data)))
        if shuffle_tasks:
            self.rng.shuffle(self.task_order)
        self.task_cursor = 0
        self.current_task_idx = self.task_order[0]
        return self._observation_for(self.current_task_idx), {
            "task": int(self.current_task_idx),
            "task_name": self.train_data[self.current_task_idx]["name"],
        }

    # ------------------------------------------------------------------
    # Joint action encoding and projection
    # ------------------------------------------------------------------
    def encode_action(self, core_select, vf_select, replica_n):
        core_select = int(core_select)
        vf_select = int(vf_select)
        replica_n = int(replica_n)
        if not (0 <= core_select < self.cores):
            raise ValueError("Invalid core index")
        if not (0 <= vf_select < self.n_vf):
            raise ValueError("Invalid VF index")
        if not (0 <= replica_n <= self.rep_max):
            raise ValueError("Invalid replica count")
        return ((core_select * self.n_vf) + vf_select) * self.n_rep + replica_n

    def decode_action(self, action_idx):
        action_idx = int(action_idx)
        if not (0 <= action_idx < self.n_actions):
            raise ValueError(f"action_idx must be in [0, {self.n_actions})")
        replica_n = action_idx % self.n_rep
        q = action_idx // self.n_rep
        vf_select = q % self.n_vf
        core_select = q // self.n_vf
        return int(core_select), int(vf_select), int(replica_n)

    def task_utilization(self, task_idx, vf_select):
        t = self.train_data[int(task_idx)]
        f = self.vf_levels[int(vf_select)]["frequency"]
        return float((t["initial_exec_task_time"] / (f / self.initial_frequency)) / t["deadlines"])

    def peak_power(self, task_idx, vf_select):
        t = self.train_data[int(task_idx)]
        vf = self.vf_levels[int(vf_select)]
        f, v = vf["frequency"], vf["voltage"]
        p_static = t["I0_exp_term"] * v
        return float(t["peak"] * (v ** 2) * f + p_static)

    def project_action(self, action_idx, task_idx=None):
        """Project only the PRIMARY execution onto utilization.

        `replica_n` is a standby reliability decision. It does not multiply current utilization.
        Replica-related execution pressure at high load is represented by the load-dependent
        penalty in `_reliability_terms()` until an explicit timing/fault model is available.
        """
        if task_idx is None:
            if self.current_task_idx is None:
                raise RuntimeError("No current task.")
            task_idx = self.current_task_idx

        core_select, vf_select, replica_n = self.decode_action(action_idx)
        util_inc = self.task_utilization(task_idx, vf_select)

        projected = self.state.astype(np.float64).copy()
        projected[core_select] += util_inc

        return {
            "core": core_select,
            "vf": vf_select,
            "replica_n": replica_n,
            "util_inc": util_inc,
            "projected_state": projected.astype(np.float32),
            "placements": (int(core_select),),
            "primary_resource_copies": 1,
            "replica_execution_mode": "standby",
            "capacity_violation": bool(np.max(projected) > self.capacity_limit + 1e-12),
            "thermal_violation": bool(self.peak_power(task_idx, vf_select) > self.TSP),
        }

    def joint_action_mask(self, task_idx=None):
        """True means the PRIMARY placement is capacity-feasible for the current task.

        Standby replicas are not charged as always-running utilization. Thermal
        safety stays in the reward rather than the hard mask. If every primary placement violates
        capacity, the least-overloading actions are exposed instead of returning an all-False mask.
        """
        if task_idx is None:
            task_idx = self.current_task_idx
        mask = np.zeros(self.n_actions, dtype=np.bool_)
        max_loads = np.empty(self.n_actions, dtype=np.float32)

        for a in range(self.n_actions):
            p = self.project_action(a, task_idx)
            max_loads[a] = float(np.max(p["projected_state"]))
            mask[a] = not p["capacity_violation"]

        if not np.any(mask):
            best = float(np.min(max_loads))
            mask = np.isclose(max_loads, best, rtol=0.0, atol=1e-8)
        return mask

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def _utilization_reward(self, action_idx, task_idx):
        """Reward primary-core choice relative to alternatives for the same VF/replica decision."""
        core_select, vf_select, replica_n = self.decode_action(action_idx)
        candidate_max = []
        chosen_max = None
        for c in range(self.cores):
            a = self.encode_action(c, vf_select, replica_n)
            p = self.project_action(a, task_idx)
            m = float(np.max(p["projected_state"]))
            candidate_max.append(m)
            if c == core_select:
                chosen_max = m

        best = min(candidate_max)
        worst = max(candidate_max)
        if worst - best <= 1e-8:
            return 1.0
        r = 1.0 - 2.0 * (chosen_max - best) / (worst - best)
        return float(np.clip(r, -1.0, 1.0))

    def _reliability_terms(self, task_idx, vf_select, replica_n):
        """Reliability benefit + standby-replica scheduling pressure.

        Replicas are NOT treated as continuously executing copies. Their reliability benefit is
        analytical, while their scheduling cost is represented by the load-dependent
        penalty: cheap at low load, increasingly expensive once the system enters the high-load
        region where deferred backup windows near deadlines are more likely to overlap/compete.

        Fault-triggered execution and overlap duration require an event/fault timeline, which
        is outside the assignment model.
        """
        t = self.train_data[int(task_idx)]
        f = self.vf_levels[int(vf_select)]["frequency"]
        v = self.vf_levels[int(vf_select)]["voltage"]

        # Keep fractional execution time before scaling.
        exec_time_scaled = float(t["initial_exec_task_time"] / (f / self.initial_frequency)) * 1000.0
        task_rel = self.total_task_reliability((int(replica_n), exec_time_scaled, float(v)))
        prospective = self.system_tasks + [task_rel]
        system_r = self.system_reliability(prospective)
        reliability_reward = self.power_sharp(system_r)

        replica_penalty = 0.0
        if replica_n <= 1:
            reliability_reward = -1.0
        elif self.load < self.threshold_load:
            replica_penalty = (
                (1 / self.cores)
                * (self.load ** 4)
                * (2 ** replica_n)
            ) / (2 ** self.rep_max)
            reliability_reward -= replica_penalty
        else:
            # Apply the high-load penalty with numerical protection.
            denom = max(1e-6, 1.1 - self.load)
            replica_penalty = (
                ((1 / denom) ** 2)
                * (2 ** replica_n)
                * (1 / self.cores)
                * (replica_n ** 2)
            ) / (2 ** self.rep_max)
            reliability_reward -= replica_penalty

        reliability_reward = float(np.clip(reliability_reward, -1.0, 1.0))
        return task_rel, system_r, reliability_reward, float(replica_penalty)

    def reward_function(self, action_idx, task_idx):
        core_select, vf_select, replica_n = self.decode_action(action_idx)
        projection = self.project_action(action_idx, task_idx)

        peak = self.peak_power(task_idx, vf_select)
        f = self.vf_levels[vf_select]["frequency"]
        f_min = self.vf_levels[0]["frequency"]
        f_max = self.vf_levels[self.n_vf - 1]["frequency"]
        f_norm = (f - f_min) / (f_max - f_min + 1e-12)
        thermal_reward = -1.0 if peak > self.TSP else float(1.0 - f_norm)

        utilization_reward = self._utilization_reward(action_idx, task_idx)
        task_rel, system_r, reliability_reward, replica_penalty = self._reliability_terms(task_idx, vf_select, replica_n)
        projection["replica_penalty"] = float(replica_penalty)

        alpha1, alpha2, alpha3, alpha4 = self.alpha
        # alpha2 is unused because the reward has no separate performance term.
        total_reward = (
            alpha1 * thermal_reward
            + alpha3 * utilization_reward
            + alpha4 * reliability_reward
        )

        # Capacity is treated as a hard scheduling constraint. Normally masked actions prevent this;
        # this branch handles the rare all-actions-infeasible fallback state.
        if projection["capacity_violation"]:
            total_reward = -1.0

        total_reward = float(np.clip(total_reward, -1.0, 1.0))
        weighted_terms = (
            alpha1 * thermal_reward,
            alpha3 * utilization_reward,
            alpha4 * reliability_reward,
        )
        return total_reward, projection, task_rel, system_r, weighted_terms

    def step(self, action_idx):
        if self.current_task_idx is None:
            raise RuntimeError("Episode is finished. Call reset().")

        task_idx = int(self.current_task_idx)
        task_name = self.train_data[task_idx]["name"]
        reward, projection, task_rel, system_r, weighted_terms = self.reward_function(action_idx, task_idx)

        # Apply only the PRIMARY task utilization. Standby replicas do not automatically consume
        # utilization; their high-load scheduling pressure is handled in the reliability penalty.
        self.state = projection["projected_state"].astype(np.float32)
        self.system_tasks.append(float(task_rel))

        self.task_cursor += 1
        terminated = self.task_cursor >= len(self.task_order)
        if terminated:
            self.current_task_idx = None
            next_obs = self._terminal_observation()
        else:
            self.current_task_idx = int(self.task_order[self.task_cursor])
            next_obs = self._observation_for(self.current_task_idx)

        core_select, vf_select, replica_n = self.decode_action(action_idx)
        info = {
            "task": task_idx,
            "task_name": task_name,
            "action_idx": int(action_idx),
            "core_select": core_select,
            "vf_select": vf_select,
            "replica_n": replica_n,
            "primary_resource_copies": projection["primary_resource_copies"],
            "replica_execution_mode": projection["replica_execution_mode"],
            "placements": tuple(projection["placements"]),
            "primary_util_inc": float(projection["util_inc"]),
            "replica_penalty": float(projection["replica_penalty"]),
            "projected_state": projection["projected_state"].copy(),
            "capacity_violation": projection["capacity_violation"],
            "thermal_violation": projection["thermal_violation"],
            "task_reliability": float(task_rel),
            "system_reliability": float(system_r),
            "reward_terms": tuple(float(x) for x in weighted_terms),
        }
        return next_obs, float(reward), bool(terminated), False, info
