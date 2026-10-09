# Reliability-Aware RL Scheduling for Cyber-Physical Systems

This repository implements task assignment and runtime simulation for periodic
cyber-physical workloads on multicore processors. The schedulers account for processor
utilization, voltage and frequency selection, thermal limits, and task reliability.

## Components

- Synthetic task-set generation from the bundled A15 benchmark measurements.
- Q-learning, Double Q-learning, and PPO scheduling methods.
- Task-aware observations and masked joint actions for selecting a core, a
  voltage/frequency level, and a standby-replica configuration.
- Reliability modeling with load-dependent replica pressure.
- Runtime simulation with EDF dispatch, primary/backup placement, checkpoints,
  rollback, fault injection, and backup activation.
- EDF, FIFO, NSGA-III, particle-swarm, and passive primary-backup baselines.

## Repository layout

```text
.
├── data/
│   ├── benchmarks/                # Source workload measurements
│   └── generated/                 # Generated task sets
├── notebooks/
│   ├── generate_task_sets.ipynb
│   ├── our_method/
│   │   ├── q_learning.ipynb
│   │   ├── double_q_learning.ipynb
│   │   ├── ppo.ipynb
│   │   ├── runtime_simulation.ipynb
│   │   └── scheduling_environment.py
│   └── baselines/
│       ├── with_reliability/
│       └── without_reliability/
└── requirements.txt
```

## Setup

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter lab
```

## Running the experiments

1. Run `notebooks/generate_task_sets.ipynb` to create task sets under
   `data/generated/`.
2. Open one of the scheduling notebooks in `notebooks/our_method/`.
3. Set `RUN_TRAINING = True` in the selected notebook to run its load sweep.
4. Use `runtime_simulation.ipynb` for detailed fault and backup-execution traces.
5. Run the notebooks under `notebooks/baselines/` for comparison experiments.

The default training configurations are computationally intensive. Episode counts and
load-point ranges can be reduced for short validation runs.

## Scheduling model

At each decision step, the learning agent receives per-core utilization and normalized
features for the task being scheduled. A joint action selects:

- the primary execution core;
- a discrete voltage/frequency level; and
- the standby-replica count.

Capacity-infeasible actions are masked. The reward combines thermal behavior, load
distribution, and reliability. Standby replicas do not add normal utilization to the
assignment state; their scheduling pressure is represented by the reliability penalty.

Q-learning stores one action-value table, Double Q-learning uses two cross-evaluated
tables, and PPO uses a neural policy and value function. All three methods use the same
scheduling inputs, action encoding, constraints, and evaluation metrics.

## Outputs

Generated task sets are written to `data/generated/`. Training and simulation outputs
are written to `artifacts/`, including:

- learned policies or action-value tables;
- reward plots and timing summaries;
- per-step evaluation records;
- utilization, power, reliability, and task reports.

The generated directories are excluded from version control and can be recreated from
the notebooks.

## Model assumptions

- The assignment environment models standby replicas analytically and does not include
  explicit replica start times or a fault-event timeline.
- The runtime simulator uses discrete millisecond time steps.
- Power, thermal, execution-time, and fault behavior follow the equations and parameters
  defined in the notebooks.
- Benchmark measurements and generated task sets are intended for the experiment
  configurations provided in this repository.

## Reproducibility

- Task generation and learning routines accept fixed random seeds.
- Data and artifact paths are resolved relative to the repository root.
- Notebook outputs and execution counts are cleared.
- Long-running experiment cells are disabled by default.
