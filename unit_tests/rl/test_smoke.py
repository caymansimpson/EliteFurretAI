"""
Smoke test for RL training (src/elitefurretai/rl/train.py).

Runs 3 gradient updates end-to-end using the Rust battle backend so no
Showdown server is required.  The test verifies that:
  - the script starts cleanly (config parse, model init, worker spawn)
  - workers produce trajectories and the learner completes updates
  - a final checkpoint is written to disk

Marked `smoke` — run with `pytest -m smoke` or skip with `pytest -m "not smoke"`.
Expected wall-clock time: 20–60 s.
"""

import sys
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Minimal config: Rust backend, tiny LSTM, 3 updates, no external services.
# Paths that depend on tmp_path are filled in by the test via str.format().
_SMOKE_YAML = textwrap.dedent("""\
    algorithm:
      clip_range: 0.2
      ent_coef: 0.01
      ent_coef_end: 0.001
      gae_lambda: 0.95
      gamma: 1
      max_grad_norm: 0.5
      rnad_alpha: 0.25
      vf_coef: 0.5

    portfolio:
      max_portfolio_size: 1
      portfolio_add_interval: 100
      portfolio_update_strategy: recent

    exploration:
      temperature_start: 1.0
      temperature_end: 1.0
      temperature_anneal_steps: 1000
      top_p: 1.0

    optimizer:
      type: adamw
      weight_decay: 0.0001
      warmup_steps: 1
      schedule: cosine
      backbone_lr: 0.0001
      backbone_weight_decay: 0.0001
      heads_lr: 0.0001
      heads_weight_decay: 0.0
      lr: 0.0001

    value_head:
      num_value_bins: 51
      value_min: -1.0
      value_max: 1.0

    architecture:
      early_layers: [64]
      late_layers: [64]
      dropout: 0.0
      grouped_encoder_hidden_dim: 64
      grouped_encoder_aggregated_dim: 64
      pokemon_attention_heads: 1
      teampreview_head_layers: [64]
      teampreview_head_dropout: 0.0
      teampreview_attention_heads: 1
      turn_head_layers: [64]
      max_seq_len: 40
      number_bank_hp_bins: 100
      number_bank_stat_bins: 600
      number_bank_power_bins: 250
      number_bank_embedding_dim: 16
      transformer_layers: 1
      transformer_heads: 1
      transformer_ff_dim: 64
      transformer_dropout: 0.0
      use_decision_tokens: false
      use_causal_mask: false

    hardware:
      batch_size: 1
      batch_timeout: 0.001
      device: cpu
      max_battle_steps: 20
      num_battles_per_pair: 4
      num_players: 2
      num_servers: 1
      num_workers: 1
      showdown_start_port: 8000
      use_mixed_precision: false
      battle_backend: rust_engine

    curriculum:
      battle_format: gen9vgc2024regg
      base_team_path: data/teams
      team_pool_path: null
      agent_team_path: null
      bc_model_path: null
      curriculum_weights:
        self_play: 1.0
      adaptive_curriculum: false
      max_exploiter_models: 0
      max_ghosts: 0
      vgc_bench_checkpoint_path: null
      external_vgcbench_usernames: []
      external_vgcbench_startup_wait_s: 0.0
      auto_launch_external_vgcbench: false
      external_vgcbench_python_executable: python
      external_vgcbench_team_file: ""
      dedicated_vgcbench_workers: 0

    training:
      embedder_feature_set: full
      initialize_path: null
      resume_from: null
      max_updates: 3
      save_dir: "{save_dir}"
      train_batch_size: 2
      checkpoint_interval: 1000
      log_interval: 1
      use_wandb: false
      wandb_project: elitefurretai-test
      wandb_run_name: smoke-test
      train_exploiters: false
""")


@pytest.mark.smoke
def test_rl_train_smoke(tmp_path):
    """RL training runs 3 gradient updates and writes a final checkpoint."""
    save_dir = tmp_path / "models"
    save_dir.mkdir()

    config_path = tmp_path / "smoke_rl.yaml"
    config_path.write_text(_SMOKE_YAML.format(save_dir=save_dir))

    result = subprocess.run(
        [sys.executable, "src/elitefurretai/rl/train.py", "--config", str(config_path)],
        cwd=str(REPO_ROOT),
        timeout=120,
        capture_output=True,
        text=True,
    )

    tail = lambda s: s[-3000:] if len(s) > 3000 else s
    assert result.returncode == 0, (
        f"RL smoke test exited with code {result.returncode}\n"
        f"--- stdout (tail) ---\n{tail(result.stdout)}\n"
        f"--- stderr (tail) ---\n{tail(result.stderr)}"
    )

    # The finally block in main() always saves a checkpoint regardless of
    # checkpoint_interval, so at least one .pt file should exist. Checkpoints
    # land under save_dir/<run_name>/ since the per-run-subdir refactor.
    checkpoints = list(save_dir.rglob("*.pt"))
    assert checkpoints, f"No checkpoint found in {save_dir} after training completed"
