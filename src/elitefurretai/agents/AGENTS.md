# `agents/` — EliteFurretAI Battle Participants

This directory holds **user-facing, instantiable agents** — the things you grab to run a battle in EFA. Eval/analysis players, heuristic baselines, the behavior-cloned player, the human-in-the-loop player, and the subprocess managers that wrap external bots (vgc-bench, foul-play).

It is **not** for classes that subclass `poke_env.player.Player` for training-plumbing reasons. `BatchInferencePlayer` is a `Player` subclass, but it lives in [rl/batch_inference_player.py](../rl/batch_inference_player.py) because its job is dynamic batching for RL training throughput — coupled to the trajectory queue and the inference IPC layer. You'd never grab it to run an ad-hoc battle.

## What's in here

| File | Class | Role |
|---|---|---|
| [`simple_model_player.py`](simple_model_player.py) | `SimpleModelPlayer` | Eval-time Player loading a trained checkpoint and running inference inline in `choose_move`. The default choice for evaluation, analysis, and head-to-head matchups. |
| [`verbose_model_player.py`](verbose_model_player.py) | `VerboseModelPlayer` | `SimpleModelPlayer` + top-k action-probability logging per turn. Use for ad-hoc model-behavior inspection. |
| [`max_damage_player.py`](max_damage_player.py) | `MaxDamagePlayer` | Heuristic baseline: picks the highest-damage action via `poke_env.calc.calculate_damage`. One of the four Stage II graduation baselines. |
| [`bc_player.py`](bc_player.py) | `BCPlayer` | Player loading a supervised (behavior-cloned) checkpoint. Maintains an across-turn trajectory tensor; used as a curriculum opponent and as one of the four Stage II graduation baselines. |
| [`human_player.py`](human_player.py) | `HumanPlayer` | Terminal-driven manual control. Used by `rl/analyze/play_human_vs_model.py` so a human can battle a trained checkpoint locally. |
| [`vgcbench_manager.py`](vgcbench_manager.py) | `VGCBenchManager` | Launches and supervises the external vgc-bench subprocess. EFA challenges by Showdown username. Also exports `_create_vgc_bench_player`, `_temporary_cwd`, `_resolve_vgc_bench_root` for in-process construction under a venv whose poke_env vintage matches vgc-bench's. |
| `foulplay_manager.py` *(landing later)* | `FoulPlayManager` | Same shape as `VGCBenchManager`, for the foul-play-doubles search bot. See `planning/stage2/2026-05-18-16-00-foulplay-eval-integration.md`. |
| [`_vgcbench_subprocess.py`](_vgcbench_subprocess.py) | — (script) | Subprocess entry point launched by `VGCBenchManager` under `../venv-vgcbench/bin/python`. Not user-invokable; the leading underscore is the signal. |
| `_foulplay_subprocess.py` *(landing later)* | — (script) | Same shape, for foul-play. |

## How to use each agent

### `SimpleModelPlayer`

```python
from elitefurretai.agents import SimpleModelPlayer
from poke_env.ps_client import AccountConfiguration, ServerConfiguration

player = SimpleModelPlayer(
    model_path="data/models/supervised/cool-bee-85-finetune_best.pt",
    device="cuda:0",
    battle_format="gen9vgc2024regg",
    probabilistic=False,                              # argmax; True samples
    account_configuration=AccountConfiguration(...),
    server_configuration=ServerConfiguration(...),
    team=team_string,
    accept_open_team_sheet=False,
)
```

Gotchas:
- Loads the model on construction — instantiation cost is not trivial. Reuse the instance across battles when possible.
- `probabilistic=False` is the default for eval (lowest variance). Pass `True` for stochastic play.

### `VerboseModelPlayer`

Same construction as `SimpleModelPlayer`. Prints top-k action probabilities to stdout each turn; use only in interactive contexts (don't run inside a training loop).

### `MaxDamagePlayer`

```python
from elitefurretai.agents import MaxDamagePlayer

player = MaxDamagePlayer(
    battle_format="gen9vgc2024regg",
    account_configuration=...,
    server_configuration=...,
    team=team_string,
    accept_open_team_sheet=False,
)
```

No model. Cheap to instantiate. Picks the action with highest estimated damage via `poke_env.calc.calculate_damage`; falls through to a default ordering when no damaging move is available (e.g. forced switches).

### `BCPlayer`

```python
from elitefurretai.agents import BCPlayer

player = BCPlayer(
    model_filepath="data/models/supervised/curious-darkness-77_best.pt",
    battle_format="gen9vgc2024regg",
    probabilistic=True,                               # sample from softmax
    device="cuda:0",
    verbose=False,
    accept_open_team_sheet=False,
    account_configuration=...,
    server_configuration=...,
    team=team_string,
)
```

Gotchas:
- Maintains an across-turn trajectory tensor per battle. `reset_battles()` clears it.
- The model checkpoint must be the new format (config + state_dict). Old state-dict-only checkpoints raise `ValueError` — migrate via `scripts/prepare/migrate_model_configs.py`.

### `HumanPlayer`

```python
from elitefurretai.agents import HumanPlayer

player = HumanPlayer(
    battle_format="gen9vgc2024regg",
    account_configuration=...,
    server_configuration=...,
    team=team_string,
)
```

Reads keyboard input at each decision point. Practically used via `python -m elitefurretai.rl.analyze.play_human_vs_model` rather than directly — that CLI wires HumanPlayer up to a local Showdown server and pairs it with a model opponent.

### `VGCBenchManager`

Subprocess lifecycle wrapper. EFA-side code never imports vgc-bench; it `/challenge`s the subprocess by Showdown username.

```python
from elitefurretai.agents import VGCBenchManager

manager = VGCBenchManager(config, server_ports=[8000])
usernames = manager.launch()   # subprocess up; returns list of usernames
# ... run training/eval, workers challenge usernames[0] ...
manager.shutdown()             # SIGTERM, close log files
```

Gotchas:
- Requires `../venv-vgcbench/` to exist with vgc-bench installed (configured via `config.curriculum.external_vgcbench_python_executable`).
- Subprocess logs go to `data/logs/vgcbench_runners/runner_*.log` — first place to check if challenges aren't being accepted.
- `WAIT_FOR_SERVER_TIMEOUT_S=180.0` and `STARTUP_WAIT_S=10.0` are the timing parameters; the trainer waits up to `STARTUP_WAIT_S` for the subprocess to log in before issuing the first challenge.

## How to add a new agent

1. Create `src/elitefurretai/agents/<name>.py` with the class.
2. Add a re-export to `src/elitefurretai/agents/__init__.py` and the `__all__` list.
3. Add a row to the "What's in here" table above and a usage section.
4. If the class is a Player subclass usable from the eval CLI, also update `src/elitefurretai/rl/analyze/player_factory.py` to register it.

## What does *not* belong here

- **Training-time plumbing** that happens to subclass `Player` (`BatchInferencePlayer`). Lives in [rl/batch_inference_player.py](../rl/batch_inference_player.py).
- **Model wrappers** (`RNaDAgent`). They're `torch.nn.Module`s, not Players. Lives in [rl/rnad_model.py](../rl/rnad_model.py).
- **Opponent-sampling / curriculum orchestration** (`OpponentPool`, `WorkerOpponentFactory`). They consume agents but aren't ones. Lives in [rl/opponents.py](../rl/opponents.py).
- **CLI scaffolding** for the eval entry point (`player_factory.py`, `team_provider.py`). Lives in [rl/analyze/](../rl/analyze/).
