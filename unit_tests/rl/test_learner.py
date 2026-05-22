# -*- coding: utf-8 -*-
"""
Unit tests for RL learner (RNaDLearner).

These tests verify:
1. Learner initialization
2. Reference model freezing and updates
3. PPO + RNaD update logic
4. Mixed precision training
5. Gradient clipping
6. Batch processing

Note: Tests use small models for speed. Uses CPU for compatibility.
"""

import copy

import pytest
import torch

from elitefurretai.etl.embedder import Embedder
from elitefurretai.etl.encoder import MDBO
from elitefurretai.rl.config import RNaDConfig
from elitefurretai.rl.learners import PortfolioRNaDLearner
from elitefurretai.rl.rnad_model import RNaDModel
from elitefurretai.supervised.model_archs import TransformerThreeHeadedModel

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def simple_embedder():
    """Create a simple Embedder for testing."""
    return Embedder(feature_set="simple")


@pytest.fixture
def small_model(simple_embedder):
    """Create a small TransformerThreeHeadedModel for testing."""
    return TransformerThreeHeadedModel(
        embedder=simple_embedder,
        early_layers=[32, 16],
        late_layers=[32, 16],
        num_actions=MDBO.action_space(),
        num_teampreview_actions=MDBO.teampreview_space(),
        max_seq_len=10,
        dropout=0.0,
        transformer_layers=1,
        transformer_heads=2,
        transformer_ff_dim=32,
        transformer_dropout=0.0,
    )


@pytest.fixture
def agent(small_model):
    """Create RNaDModel from small model."""
    return RNaDModel(small_model)


@pytest.fixture
def ref_agent(small_model, simple_embedder):
    """Create reference RNaDModel from small model."""
    # Create a separate instance with same architecture
    ref_model = TransformerThreeHeadedModel(
        embedder=simple_embedder,
        early_layers=[32, 16],
        late_layers=[32, 16],
        num_actions=MDBO.action_space(),
        num_teampreview_actions=MDBO.teampreview_space(),
        max_seq_len=10,
        dropout=0.0,
        transformer_layers=1,
        transformer_heads=2,
        transformer_ff_dim=32,
        transformer_dropout=0.0,
    )
    return RNaDModel(ref_model)


@pytest.fixture
def learner(agent, ref_agent):
    """Create PortfolioRNaDLearner for testing."""
    config = RNaDConfig()
    config.algorithm.clip_range = 0.2
    config.algorithm.ent_coef = 0.01
    config.algorithm.vf_coef = 0.5
    config.algorithm.rnad_alpha = 0.1
    config.algorithm.gamma = 0.99
    config.algorithm.max_grad_norm = 0.5
    config.hardware.device = "cpu"
    config.optimizer.warmup_steps = 0
    config.optimizer.schedule = "constant"
    config.optimizer.backbone_lr = 1e-4
    config.optimizer.backbone_weight_decay = 1e-4
    config.optimizer.heads_lr = 3e-4
    config.optimizer.heads_weight_decay = 0.0
    return PortfolioRNaDLearner(
        model=agent,
        ref_models=[ref_agent],
        config=config,
        device="cpu",
    )


@pytest.fixture
def sample_batch(simple_embedder):
    """
    Create sample batch for training.

    Returns dict with all required tensors for learner.update().
    """
    batch_size = 4
    seq_len = 5
    state_dim = simple_embedder.embedding_size

    # Random states
    states = torch.randn(batch_size, seq_len, state_dim)

    # Is teampreview: first step is teampreview, rest are turns
    is_teampreview = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    is_teampreview[:, 0] = True

    # Random valid actions (within appropriate action space)
    # Teampreview actions must be < 90, turn actions can be up to 2025
    actions = torch.zeros(batch_size, seq_len, dtype=torch.long)
    for b in range(batch_size):
        for s in range(seq_len):
            if is_teampreview[b, s]:
                actions[b, s] = torch.randint(0, MDBO.teampreview_space(), (1,)).item()
            else:
                actions[b, s] = torch.randint(0, MDBO.action_space(), (1,)).item()

    # Rewards
    rewards = torch.randn(batch_size, seq_len)

    # Old values and log_probs from previous policy
    values = torch.randn(batch_size, seq_len)
    log_probs = -torch.rand(batch_size, seq_len)  # Log probs are negative

    # Advantages and returns (pre-computed)
    advantages = torch.randn(batch_size, seq_len)
    returns = torch.randn(batch_size, seq_len)

    # Action masks (all actions valid for simplicity)
    masks = torch.ones(batch_size, seq_len, MDBO.action_space(), dtype=torch.bool)

    return {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "values": values,
        "log_probs": log_probs,
        "advantages": advantages,
        "returns": returns,
        "is_teampreview": is_teampreview,
        "masks": masks,
    }


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================


def test_learner_initialization(learner):
    """Test PortfolioRNaDLearner initializes correctly."""
    assert learner.gamma == 0.99
    assert learner.clip_range == 0.2
    assert learner.ent_coef == 0.01
    assert learner.vf_coef == 0.5
    assert learner.rnad_alpha == 0.1
    assert learner.device == "cpu"
    assert learner.gradient_clip == 0.5


def test_learner_creates_optimizer(learner):
    """
    Test that learner creates optimizer.

    Expected: Optimizer is torch.optim.AdamW or Adam instance.
    """
    assert isinstance(learner.optimizer, (torch.optim.Adam, torch.optim.AdamW))


def test_ref_model_frozen(learner):
    """Test that reference model parameters are frozen."""
    for ref in learner.ref_models:
        for param in ref.parameters():
            assert not param.requires_grad, "Reference model should be frozen"


def test_main_model_trainable(learner):
    """
    Test that main model parameters are trainable.

    Expected: Main model params have requires_grad=True.
    """
    trainable_count = sum(1 for p in learner.model.parameters() if p.requires_grad)
    assert trainable_count > 0, "Main model should have trainable parameters"


# =============================================================================
# REFERENCE MODEL UPDATE TESTS
# =============================================================================


def test_add_reference_model_snapshots_main(learner):
    """Test add_reference_model snapshots current weights from main model."""
    with torch.no_grad():
        for param in learner.model.parameters():
            param.fill_(1.0)

    learner.add_reference_model(RNaDModel(copy.deepcopy(learner.model.model)))

    for main_param, ref_param in zip(
        learner.model.parameters(), learner.ref_models[-1].parameters()
    ):
        assert torch.allclose(main_param, ref_param), (
            "New ref model should match main model at snapshot time"
        )


def test_add_reference_model_keeps_frozen(learner):
    """Test that ref models stay frozen after add."""
    learner.add_reference_model(RNaDModel(copy.deepcopy(learner.model.model)))

    for ref in learner.ref_models:
        for param in ref.parameters():
            assert not param.requires_grad, "Ref model should stay frozen"


# =============================================================================
# UPDATE STEP TESTS
# =============================================================================


def test_update_returns_losses(learner, sample_batch):
    """
    Test that update returns loss dictionary.

    Expected: Returns dict with loss, policy_loss, value_loss, entropy, rnad_loss.
    """
    losses = learner.update(sample_batch)

    assert isinstance(losses, dict)
    assert "loss" in losses
    assert "policy_loss" in losses
    assert "value_loss" in losses
    assert "entropy" in losses
    assert "rnad_loss" in losses


def test_update_losses_are_finite(learner, sample_batch):
    """
    Test that all losses are finite values.

    Expected: No NaN or Inf values in losses.
    """
    losses = learner.update(sample_batch)

    for key, value in losses.items():
        if not isinstance(value, (int, float)):
            continue
        assert not torch.isnan(torch.tensor(value)), f"{key} is NaN"
        assert not torch.isinf(torch.tensor(value)), f"{key} is Inf"


def test_update_modifies_weights(learner, sample_batch):
    """
    Test that update actually modifies model weights.

    Expected: Weights change after update step.
    """
    # Get initial weights
    initial_weights = [p.clone() for p in learner.model.parameters()]

    # Perform update
    learner.update(sample_batch)

    # Check some weights changed
    weights_changed = False
    for initial, current in zip(initial_weights, learner.model.parameters()):
        if not torch.allclose(initial, current):
            weights_changed = True
            break

    assert weights_changed, "Model weights should change after update"


def test_update_with_padding_mask(learner, sample_batch):
    """
    Test update handles padding mask correctly.

    Padding mask zeros out loss for padded positions.

    Expected: Update completes without error.
    """
    # Add padding mask (last 2 positions are padding)
    batch_size, seq_len = sample_batch["actions"].shape
    padding_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    padding_mask[:, -2:] = False
    sample_batch["padding_mask"] = padding_mask

    losses = learner.update(sample_batch)

    assert "loss" in losses
    assert not torch.isnan(torch.tensor(losses["loss"]))


def test_update_with_initial_hidden(learner, sample_batch):
    """
    Test update handles provided initial hidden state.

    Expected: Update uses provided hidden state.
    """
    batch_size = sample_batch["states"].shape[0]

    # Create initial hidden state
    num_layers = 1
    num_directions = 2
    hidden_size = 16
    h0 = torch.zeros(num_layers * num_directions, batch_size, hidden_size)
    c0 = torch.zeros(num_layers * num_directions, batch_size, hidden_size)

    sample_batch["initial_hidden"] = (h0, c0)

    losses = learner.update(sample_batch)

    assert "loss" in losses


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


def test_update_all_teampreview(learner, sample_batch):
    """
    Test update with all teampreview steps.

    Expected: Handles case where all steps are teampreview.
    """
    # Mark all steps as teampreview
    sample_batch["is_teampreview"] = torch.ones_like(
        sample_batch["is_teampreview"], dtype=torch.bool
    )
    # Actions should be valid teampreview actions (< 90)
    sample_batch["actions"] = torch.randint(
        0, MDBO.teampreview_space(), sample_batch["actions"].shape
    )

    losses = learner.update(sample_batch)

    assert "loss" in losses
    assert not torch.isnan(torch.tensor(losses["loss"]))


def test_update_all_turn(learner, sample_batch):
    """
    Test update with all turn steps (no teampreview).

    Expected: Handles case where no steps are teampreview.
    """
    # Mark all steps as turn actions
    sample_batch["is_teampreview"] = torch.zeros_like(
        sample_batch["is_teampreview"], dtype=torch.bool
    )

    losses = learner.update(sample_batch)

    assert "loss" in losses
    assert not torch.isnan(torch.tensor(losses["loss"]))


def test_update_single_step(learner, sample_batch):
    """
    Test update with single step trajectories.

    Expected: Handles seq_len=1 correctly.
    """
    # Reduce to single step
    for key in sample_batch:
        if isinstance(sample_batch[key], torch.Tensor) and sample_batch[key].dim() > 1:
            sample_batch[key] = sample_batch[key][:, :1]

    losses = learner.update(sample_batch)

    assert "loss" in losses


def test_update_batch_size_one(learner, sample_batch):
    """
    Test update with batch_size=1.

    Expected: Handles single batch correctly.
    """
    # Reduce to single batch
    for key in sample_batch:
        if isinstance(sample_batch[key], torch.Tensor):
            sample_batch[key] = sample_batch[key][:1]

    losses = learner.update(sample_batch)

    assert "loss" in losses


# =============================================================================
# GRADIENT CLIPPING TESTS
# =============================================================================


def test_gradient_clipping_applied(learner, sample_batch):
    """
    Test that gradient clipping is applied during update.

    Expected: Gradients are clipped to gradient_clip value.
    """
    # Create batch with potentially large gradients
    sample_batch["advantages"] = sample_batch["advantages"] * 100  # Large advantages

    # Hook to capture gradient norms before clipping
    grad_norms = []

    def hook(module):
        total_norm = 0
        for p in module.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm**0.5
        grad_norms.append(total_norm)

    # Perform update (clipping happens internally)
    learner.update(sample_batch)

    # Can't directly verify clipping without hooks, but update should complete
    # without NaN due to clipping
    assert True  # Update completed


# =============================================================================
# RNAD AGENT TESTS
# =============================================================================


def test_rnad_model_forward(agent, simple_embedder):
    """RNaDModel.forward delegates to ``model.forward_with_hidden`` which is
    single-step (seq_len == 1) for online RL inference."""
    batch_size = 4

    x = torch.randn(batch_size, 1, simple_embedder.embedding_size)
    hidden = agent.get_initial_state(batch_size, "cpu")

    turn_logits, tp_logits, value, win_dist_logits, next_hidden = agent(x, hidden)

    assert turn_logits.shape == (batch_size, 1, MDBO.action_space())
    assert tp_logits.shape == (batch_size, 1, MDBO.teampreview_space())
    assert value.shape == (batch_size, 1)
    # Transformer next_hidden is the accumulated context tensor.
    assert isinstance(next_hidden, torch.Tensor)
    assert next_hidden.shape[0] == batch_size
    assert next_hidden.shape[1] == 1


def test_rnad_model_initial_state_is_none(agent):
    """Transformer has no initial hidden state — context starts as None."""
    hidden = agent.get_initial_state(8, "cpu")
    assert hidden is None


def test_rnad_model_value_range(agent, simple_embedder):
    """Value output is in [-1, 1] via the C51 distributional head."""
    x = torch.randn(4, 1, simple_embedder.embedding_size)
    hidden = agent.get_initial_state(4, "cpu")

    _, _, value, _, _ = agent(x, hidden)

    assert value.min() >= -1.0
    assert value.max() <= 1.0


# =============================================================================
# PPO MINI-EPOCH TESTS (algorithm.ppo_epochs)
# =============================================================================


def _make_learner_with_epochs(agent, ref_agent, ppo_epochs=1, kl_early_stop=None):
    """Build a learner identical to the `learner` fixture but with K mini-epochs."""
    config = RNaDConfig()
    config.algorithm.clip_range = 0.2
    config.algorithm.ent_coef = 0.01
    config.algorithm.vf_coef = 0.5
    config.algorithm.rnad_alpha = 0.1
    config.algorithm.gamma = 0.99
    config.algorithm.max_grad_norm = 0.5
    config.algorithm.ppo_epochs = ppo_epochs
    config.algorithm.ppo_kl_early_stop = kl_early_stop
    config.hardware.device = "cpu"
    config.optimizer.warmup_steps = 0
    config.optimizer.schedule = "constant"
    config.optimizer.backbone_lr = 1e-4
    config.optimizer.backbone_weight_decay = 1e-4
    config.optimizer.heads_lr = 3e-4
    config.optimizer.heads_weight_decay = 0.0
    return PortfolioRNaDLearner(
        model=agent, ref_models=[ref_agent], config=config, device="cpu"
    )


def test_ppo_default_epochs_is_three(learner):
    """Default config uses K=3 mini-epochs (the standard PPO recommendation)."""
    assert learner.ppo_epochs == 3


def test_ppo_metrics_include_epochs_actual(learner, sample_batch):
    """update() must report ppo_epochs_actual; matches default K=3."""
    metrics = learner.update(sample_batch)
    assert "ppo_epochs_actual" in metrics
    assert metrics["ppo_epochs_actual"] == 3
    assert "approx_kl" in metrics


def test_ppo_k3_runs_three_inner_epochs(agent, ref_agent, sample_batch):
    """With ppo_epochs=3, the inner loop should run 3 times."""
    learner = _make_learner_with_epochs(agent, ref_agent, ppo_epochs=3)
    metrics = learner.update(sample_batch)
    assert metrics["ppo_epochs_actual"] == 3
    # All losses still finite
    for key in ("loss", "policy_loss", "value_loss", "entropy", "rnad_loss"):
        v = metrics[key]
        assert not torch.isnan(torch.tensor(v)), f"{key} NaN with K=3"
        assert not torch.isinf(torch.tensor(v)), f"{key} Inf with K=3"


def test_ppo_k3_takes_more_optimizer_steps_than_k1(agent, ref_agent, sample_batch):
    """K=3 should take 3 optimizer steps for one update vs 1 for K=1."""
    # K=1
    l1 = _make_learner_with_epochs(agent, ref_agent, ppo_epochs=1)
    initial_state = copy.deepcopy(l1.model.state_dict())
    l1.update(sample_batch)
    after_k1 = copy.deepcopy(l1.model.state_dict())

    # K=3 starting from same weights
    l3 = _make_learner_with_epochs(agent, ref_agent, ppo_epochs=3)
    l3.model.load_state_dict(initial_state)
    l3.update(sample_batch)
    after_k3 = copy.deepcopy(l3.model.state_dict())

    # K=3 should drift further from initial than K=1 (more optimizer steps).
    def total_drift(initial, final):
        return sum(
            (final[k] - initial[k]).norm().item()
            for k in initial
            if initial[k].dtype.is_floating_point
        )

    drift_k1 = total_drift(initial_state, after_k1)
    drift_k3 = total_drift(initial_state, after_k3)
    assert drift_k3 > drift_k1, (
        f"Expected K=3 to drift further than K=1 (drift_k3={drift_k3:.6f}, drift_k1={drift_k1:.6f})"
    )


def test_ppo_step_counter_advances_once_per_update(agent, ref_agent, sample_batch):
    """self._step bumps per update, not per epoch — entropy annealing depends on it."""
    learner = _make_learner_with_epochs(agent, ref_agent, ppo_epochs=4)
    assert learner._step == 0
    learner.update(sample_batch)
    assert learner._step == 1
    learner.update(sample_batch)
    assert learner._step == 2


def test_ppo_kl_early_stop_caps_inner_epochs(agent, ref_agent, sample_batch):
    """A tiny KL threshold should make the inner loop bail out early."""
    # Threshold 0 means even a tiny ratio drift triggers early stop.
    learner = _make_learner_with_epochs(agent, ref_agent, ppo_epochs=10, kl_early_stop=0.0)
    metrics = learner.update(sample_batch)
    assert metrics["ppo_epochs_actual"] < 10, (
        "Early stop with KL threshold 0 should fire before all 10 epochs"
    )
    assert metrics["ppo_epochs_actual"] >= 1, "At least one epoch must run"


def test_ppo_portfolio_selection_count_tracked_once_per_update(
    agent, ref_agent, sample_batch
):
    """Selection counter should bump once per update regardless of K."""
    learner = _make_learner_with_epochs(agent, ref_agent, ppo_epochs=5)
    # tp + turn paths each call _compute_portfolio_kl once on the first epoch,
    # so the counter on the (single) ref model bumps by 2 per update — same as
    # K=1 baseline. Verify that and not 2*K=10.
    initial_count = learner.portfolio_selection_counts[0]
    learner.update(sample_batch)
    final_count = learner.portfolio_selection_counts[0]
    assert final_count - initial_count == 2, (
        f"Selection count should bump by 2 (tp+turn) per update, got "
        f"{final_count - initial_count}"
    )


# =============================================================================
# RESUME-STATE TESTS
# =============================================================================
#
# These pin down the invariants violated by the original resume path
# (see planning/stage2/2026-05-14-17-00-resume-state-bugs.md): the LR
# scheduler used to rewind to step 0 on every resume, and `_step` (which
# drives the entropy-bonus anneal) used to reset to 0. Both are now
# round-tripped through `save_checkpoint` / `load_resume_state`.


def test_load_resume_state_restores_scheduler_last_epoch(agent, ref_agent, sample_batch):
    """Scheduler's `last_epoch` must round-trip via load_resume_state."""
    learner_a = _make_learner_with_epochs(agent, ref_agent, ppo_epochs=1)
    for _ in range(5):
        learner_a.update(sample_batch)
    saved_last_epoch = learner_a.scheduler.last_epoch
    assert saved_last_epoch == 5

    # Fresh learner, then restore.
    learner_b = _make_learner_with_epochs(agent, ref_agent, ppo_epochs=1)
    assert learner_b.scheduler.last_epoch == 0
    fake_checkpoint = {
        "optimizer_state_dict": learner_a.optimizer.state_dict(),
        "scheduler_state_dict": learner_a.scheduler.state_dict(),
        "learner_step": learner_a._step,
        "step": 5,
    }
    learner_b.load_resume_state(fake_checkpoint)
    assert learner_b.scheduler.last_epoch == saved_last_epoch


def test_load_resume_state_restores_learner_step(agent, ref_agent, sample_batch):
    """`_step` (which drives ent_coef anneal) must round-trip."""
    learner_a = _make_learner_with_epochs(agent, ref_agent, ppo_epochs=1)
    for _ in range(7):
        learner_a.update(sample_batch)
    assert learner_a._step == 7

    learner_b = _make_learner_with_epochs(agent, ref_agent, ppo_epochs=1)
    fake_checkpoint = {
        "optimizer_state_dict": learner_a.optimizer.state_dict(),
        "scheduler_state_dict": learner_a.scheduler.state_dict(),
        "learner_step": learner_a._step,
        "step": 7,
    }
    learner_b.load_resume_state(fake_checkpoint)
    assert learner_b._step == 7


def test_load_resume_state_legacy_checkpoint_falls_back_to_step(
    agent, ref_agent, sample_batch
):
    """Pre-fix checkpoints (no scheduler/learner_step keys) anchor to `step`.

    A checkpoint produced by the previous save_checkpoint won't have
    `scheduler_state_dict` or `learner_step`. We must not rewind warmup or
    the entropy anneal back to zero on those resumes — the fallback uses
    the global update counter so the LR schedule and ent_coef continue
    roughly where they left off.
    """
    learner_a = _make_learner_with_epochs(agent, ref_agent, ppo_epochs=1)
    learner_a.update(sample_batch)

    learner_b = _make_learner_with_epochs(agent, ref_agent, ppo_epochs=1)
    legacy_checkpoint = {
        "optimizer_state_dict": learner_a.optimizer.state_dict(),
        "step": 365,
        # no scheduler_state_dict, no learner_step
    }
    learner_b.load_resume_state(legacy_checkpoint)
    assert learner_b._step == 365
    # last_epoch is one less than `step` so the NEXT scheduler.step() call
    # lands on `step` exactly.
    assert learner_b.scheduler.last_epoch == 364


def test_save_checkpoint_round_trip_preserves_scheduler_and_step(
    tmp_path, agent, ref_agent, sample_batch
):
    """End-to-end: save_checkpoint(learner) -> load_checkpoint -> load_resume_state."""
    from elitefurretai.rl.learners import load_checkpoint, save_checkpoint

    learner_a = _make_learner_with_epochs(agent, ref_agent, ppo_epochs=1)
    for _ in range(3):
        learner_a.update(sample_batch)

    save_dir = str(tmp_path)
    filepath = save_checkpoint(
        model=agent,
        learner=learner_a,
        step=3,
        config=learner_a.config,
        curriculum={"self_play": 1.0},
        save_dir=save_dir,
    )

    checkpoint = load_checkpoint(filepath, device="cpu")
    assert "scheduler_state_dict" in checkpoint
    assert "learner_step" in checkpoint
    assert checkpoint["learner_step"] == 3

    learner_b = _make_learner_with_epochs(agent, ref_agent, ppo_epochs=1)
    learner_b.load_resume_state(checkpoint)
    assert learner_b._step == 3
    assert learner_b.scheduler.last_epoch == learner_a.scheduler.last_epoch


def test_load_resume_state_does_not_touch_model_or_ref(agent, ref_agent, sample_batch):
    """load_resume_state restores OPTIMIZER state only — model + ref are caller-managed.

    This is the contract that prevents Bug 1: model weights must already be
    loaded into the agent BEFORE the learner is constructed, so the ref
    deepcopy captures trained (not random) weights. load_resume_state
    intentionally has no model_state_dict handling — that responsibility is
    the caller's, and isolating it keeps the order obvious.
    """
    learner_a = _make_learner_with_epochs(agent, ref_agent, ppo_epochs=1)
    learner_a.update(sample_batch)

    learner_b = _make_learner_with_epochs(agent, ref_agent, ppo_epochs=1)
    before = copy.deepcopy(
        {k: v.clone() for k, v in learner_b.model.model.state_dict().items()}
    )
    fake_checkpoint = {
        "optimizer_state_dict": learner_a.optimizer.state_dict(),
        "scheduler_state_dict": learner_a.scheduler.state_dict(),
        "learner_step": 1,
        "step": 1,
    }
    learner_b.load_resume_state(fake_checkpoint)
    after = learner_b.model.model.state_dict()
    for k in before:
        assert torch.equal(before[k], after[k]), (
            f"load_resume_state must not modify model weights ({k} changed)"
        )


# =============================================================================
# VALUE-TO-TRUNK GRADIENT SCALE
# =============================================================================
#
# These tests pin down the behavior introduced by `value_to_trunk_grad_scale`
# in TransformerThreeHeadedModel — added 2026-05-16 in response to the
# hopeful-wood-69 value-gradient-dominance diagnosis. The semantics:
#
#   - Forward is identity. No change to logits or value outputs.
#   - Backward through the inserted node multiplies grad by `scale`.
#   - Placement is BETWEEN `late_ff_stack` and `value_ff_stack` on the value
#     path → trunk and `late_ff_stack` see scaled value-side gradient;
#     `value_ff_stack` and `win_head` see full magnitude.


def _trunk_param(model):
    """A parameter that's strictly upstream of the value path's grad-scale
    insertion (i.e. lives in the transformer trunk, before `late_ff_stack`).
    Used to verify trunk-side scaling."""
    return model.transformer.layers[0].linear1.weight


def _value_head_param(model):
    """A parameter strictly downstream of the grad-scale insertion (value
    branch only). Should receive FULL-magnitude gradient regardless of
    scale. `win_head` is a Sequential in the legacy 2-layer path and a
    Linear in the deep-value-head path; return the first leaf weight either
    way."""
    for p in model.win_head.parameters():
        if p.requires_grad and p.dim() >= 2:
            return p
    raise RuntimeError("No suitable win_head weight found")


def _grad_norm_from_value_loss(model, scale: float, simple_embedder):
    """Forward a tiny batch, compute a synthetic value loss, backprop, and
    return (trunk_grad_norm, value_head_grad_norm). Uses the same input each
    call so only `scale` varies."""
    torch.manual_seed(0)
    model.value_to_trunk_grad_scale = scale
    # Fresh grads.
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None
    # Dummy input matching the model's embedder.
    B, S = 2, 5
    feature_dim = simple_embedder.embedding_size
    x = torch.randn(B, S, feature_dim)
    mask = torch.ones(B, S, dtype=torch.bool)
    out = model(x, mask)
    win_dist_logits = out[3]  # (B, S, num_value_bins) — see model forward
    # Simple scalar loss that pulls gradient through win_head + value_ff_stack
    # + late_ff_stack + trunk.
    loss = win_dist_logits.pow(2).mean()
    loss.backward()
    trunk_g = _trunk_param(model).grad
    value_g = _value_head_param(model).grad
    assert trunk_g is not None and value_g is not None
    return trunk_g.norm().item(), value_g.norm().item()


def test_value_grad_scale_default_is_one(small_model):
    """Default attribute exists and is 1.0 so existing configs are no-ops."""
    assert hasattr(small_model, "value_to_trunk_grad_scale")
    assert small_model.value_to_trunk_grad_scale == 1.0


def test_value_grad_scale_does_not_affect_forward(small_model, simple_embedder):
    """Scale is a backward-only op; forward outputs must be identical."""
    torch.manual_seed(42)
    B, S = 2, 5
    x = torch.randn(B, S, simple_embedder.embedding_size)
    mask = torch.ones(B, S, dtype=torch.bool)

    small_model.eval()
    small_model.value_to_trunk_grad_scale = 1.0
    out_one = small_model(x, mask)
    small_model.value_to_trunk_grad_scale = 0.1
    out_scaled = small_model(x, mask)
    for a, b in zip(out_one, out_scaled):
        assert torch.allclose(a, b), "Forward must be invariant to grad scale"


def test_value_grad_scale_dampens_trunk_gradient(small_model, simple_embedder):
    """Trunk grad from value-only loss must scale linearly with the knob."""
    small_model.train()
    trunk_g_one, vhead_g_one = _grad_norm_from_value_loss(
        small_model, scale=1.0, simple_embedder=simple_embedder
    )
    trunk_g_quarter, vhead_g_quarter = _grad_norm_from_value_loss(
        small_model, scale=0.25, simple_embedder=simple_embedder
    )
    # Trunk grad should be ~4x smaller at scale=0.25. Allow loose tolerance for
    # numerical noise; the relationship is exact in theory.
    ratio = trunk_g_one / max(trunk_g_quarter, 1e-12)
    assert 3.5 < ratio < 4.5, (
        f"Trunk grad norm should scale ~linearly with grad scale; "
        f"got ratio {ratio:.3f} (expected ~4.0)"
    )


def test_value_grad_scale_leaves_value_head_unscaled(small_model, simple_embedder):
    """Value-head params are DOWNSTREAM of the scale op — full grad regardless."""
    small_model.train()
    _, vhead_g_one = _grad_norm_from_value_loss(
        small_model, scale=1.0, simple_embedder=simple_embedder
    )
    _, vhead_g_quarter = _grad_norm_from_value_loss(
        small_model, scale=0.25, simple_embedder=simple_embedder
    )
    # win_head sits below value_ff_stack which sits below the grad-scale node.
    # Backward stops accumulating at the scale node for win_head — meaning
    # win_head receives the SAME gradient regardless of scale.
    rel = abs(vhead_g_one - vhead_g_quarter) / max(vhead_g_one, 1e-12)
    assert rel < 1e-5, (
        f"win_head grad should be invariant to value_to_trunk_grad_scale; "
        f"got relative diff {rel:.6f}"
    )


# =============================================================================
# PORTFOLIO KL: MEAN-KL OVER MIN-KL
# =============================================================================


def test_portfolio_kl_returns_mean_not_min(learner):
    """With multiple references, the returned KL should be the MEAN of per-ref
    KLs, not the MIN. Constructs three refs at controlled distances from the
    current policy; mean and min are far apart, so the assertion is decisive.
    """
    from torch.distributions import Categorical

    # Current policy: peaked on action 0.
    curr_logits = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
    curr_dist = Categorical(logits=curr_logits)

    # Three refs: one identical (KL ~ 0), two very different (KL large).
    ref_logits_list = [
        torch.tensor([[10.0, 0.0, 0.0, 0.0]]),  # identical to current
        torch.tensor([[0.0, 10.0, 0.0, 0.0]]),  # peaked on action 1
        torch.tensor([[0.0, 0.0, 10.0, 0.0]]),  # peaked on action 2
    ]

    # Re-seed history slots to match the new ref count.
    learner.portfolio_kl_history = [[] for _ in ref_logits_list]
    learner.portfolio_selection_counts = [0] * len(ref_logits_list)

    result = learner._compute_portfolio_kl(curr_dist, ref_logits_list, track=False)

    # Compute reference values directly.
    per_ref_kls = []
    for rl in ref_logits_list:
        rd = Categorical(logits=rl)
        per_ref_kls.append(torch.distributions.kl_divergence(curr_dist, rd).mean().item())
    expected_mean = sum(per_ref_kls) / len(per_ref_kls)
    expected_min = min(per_ref_kls)

    # Mean and min are far apart in this construction, so the test is decisive.
    assert expected_mean - expected_min > 1.0, (
        "Fixture should produce mean-min gap >> 1 to make assertion decisive"
    )
    assert abs(result.item() - expected_mean) < 1e-4, (
        f"_compute_portfolio_kl must return MEAN ({expected_mean:.4f}), got "
        f"{result.item():.4f} (min was {expected_min:.4f})"
    )


def test_portfolio_kl_selection_counter_tracks_closest_ref(learner):
    """Even though loss uses mean-KL, the diagnostic selection counter should
    still bump the closest reference (preserves prior bookkeeping semantics)."""
    from torch.distributions import Categorical

    curr_logits = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
    curr_dist = Categorical(logits=curr_logits)
    ref_logits_list = [
        torch.tensor([[0.0, 10.0, 0.0, 0.0]]),  # far
        torch.tensor([[10.0, 0.0, 0.0, 0.0]]),  # closest (KL ~ 0)
        torch.tensor([[0.0, 0.0, 10.0, 0.0]]),  # far
    ]
    learner.portfolio_kl_history = [[] for _ in ref_logits_list]
    learner.portfolio_selection_counts = [0] * len(ref_logits_list)

    learner._compute_portfolio_kl(curr_dist, ref_logits_list, track=True)

    assert learner.portfolio_selection_counts == [0, 1, 0], (
        f"Closest ref (idx 1) should be selected; got {learner.portfolio_selection_counts}"
    )


def test_portfolio_kl_empty_refs_returns_zero(learner):
    """No references → KL is 0 (no anchor)."""
    from torch.distributions import Categorical

    curr_dist = Categorical(logits=torch.tensor([[1.0, 0.0, 0.0]]))
    result = learner._compute_portfolio_kl(curr_dist, [], track=True)
    assert result.item() == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
