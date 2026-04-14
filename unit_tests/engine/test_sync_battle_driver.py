# -*- coding: utf-8 -*-

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import torch

from elitefurretai.engine.battle_snapshot import BattleSnapshot
from elitefurretai.engine.sync_battle_driver import (
    SyncBaselineController,
    SyncPolicyPlayer,
    SyncRustBattleDriver,
)


class FakeAgent:
    _is_transformer = False

    def get_initial_state(self, batch_size: int, device: str):
        hidden = torch.zeros(1, batch_size, 4, device=device)
        return hidden, hidden.clone()

    def __call__(self, state_tensor, hidden):
        batch_size = state_tensor.shape[0]
        turn_logits = torch.full((batch_size, 1, 2025), -1000.0)
        turn_logits[:, 0, 7] = 1.0
        turn_logits[:, 0, 11] = 3.0
        tp_logits = torch.full((batch_size, 1, 90), -1000.0)
        tp_logits[:, 0, 4] = 2.0
        values = torch.full((batch_size, 1, 1), 0.25, dtype=torch.float32)
        next_hidden = (hidden[0] + 1.0, hidden[1] + 1.0)
        return turn_logits, tp_logits, values, None, next_hidden


class CountingAgent(FakeAgent):
    def __init__(self):
        self.batch_sizes = []

    def __call__(self, state_tensor, hidden):
        self.batch_sizes.append(int(state_tensor.shape[0]))
        return super().__call__(state_tensor, hidden)


class FakeTransformerAgent:
    _is_transformer = True

    def __init__(self, max_seq_len: int = 4):
        self.model = SimpleNamespace(max_seq_len=max_seq_len)

    def get_initial_state(self, batch_size: int, device: str):
        del batch_size, device
        return None

    def __call__(self, state_tensor, hidden):
        batch_size = state_tensor.shape[0]
        hidden_len = 0 if hidden is None else int(hidden.size(1))
        turn_logits = torch.full((batch_size, 1, 2025), -1000.0)
        turn_logits[:, 0, 11] = 3.0
        tp_logits = torch.full((batch_size, 1, 90), -1000.0)
        tp_logits[:, 0, 4] = 2.0
        values = torch.full((batch_size, 1, 1), 0.25, dtype=torch.float32)
        next_hidden = torch.zeros(batch_size, hidden_len + 1, 8, dtype=torch.float32)
        return turn_logits, tp_logits, values, None, next_hidden


class CountingTransformerAgent(FakeTransformerAgent):
    def __init__(self, max_seq_len: int = 4):
        super().__init__(max_seq_len=max_seq_len)
        self.batch_sizes: list[int] = []
        self.hidden_lengths: list[int] = []

    def __call__(self, state_tensor, hidden):
        self.batch_sizes.append(int(state_tensor.shape[0]))
        self.hidden_lengths.append(0 if hidden is None else int(hidden.size(1)))
        return super().__call__(state_tensor, hidden)


def test_sync_policy_player_emits_learner_trajectory(monkeypatch):
    player = SyncPolicyPlayer(
        agent=cast(Any, FakeAgent()),
        format_id="gen9vgc2023regc",
        device="cpu",
        collect_trajectories=True,
        probabilistic=False,
        max_battle_steps=10,
    )
    monkeypatch.setattr(player.embedder, "embed", lambda battle: {"battle": battle})
    monkeypatch.setattr(
        player.embedder,
        "feature_dict_to_vector",
        lambda _: [0.1, 0.2, 0.3],
    )
    monkeypatch.setattr(player.embedder, "embed_to_array", lambda battle: np.asarray([0.1, 0.2, 0.3], dtype=np.float32))
    monkeypatch.setattr(player.embedder, "embed_to_vector", lambda battle: [0.1, 0.2, 0.3])
    battle = SimpleNamespace(
        teampreview=False,
        battle_tag="sync-battle-1",
        opponent_team={
            "mon1": SimpleNamespace(fainted=False),
            "mon2": SimpleNamespace(fainted=True),
        },
    )

    choice, rollout_step = player.choose_action(
        battle_tag="sync-battle-1",
        battle=battle,
        legal_actions=[(7, "move 1, move 1"), (11, "move 2, move 1")],
        opponent_type="ghosts",
    )

    assert choice == "move 2, move 1"
    assert rollout_step is not None
    assert rollout_step.action == 11
    assert rollout_step.value == 0.25
    assert rollout_step.mask is not None
    assert rollout_step.mask[7] == 1
    assert rollout_step.mask[11] == 1

    player.finish_battle("sync-battle-1", won=True)
    completed = player.consume_completed_trajectories()

    assert len(completed) == 1
    assert completed[0]["opponent_type"] == "ghosts"
    assert completed[0]["won"] is True
    assert len(completed[0]["steps"]) == 1
    assert completed[0]["steps"][0]["action"] == 11
    assert np.allclose(completed[0]["steps"][0]["state"], [0.1, 0.2, 0.3])
    assert completed[0]["steps"][0]["reward"] > 0.0


def test_sync_baseline_controller_normalizes_choice_messages():
    class FakeChoice:
        message = "/choose move 1, move 2"

    class FakeBaseline:
        def choose_move(self, battle):
            del battle
            return FakeChoice()

        def teampreview(self, battle):
            del battle
            return "/team 1234"

    controller = SyncBaselineController(FakeBaseline(), "max_damage")

    teampreview_battle = SimpleNamespace(teampreview=True)
    turn_battle = SimpleNamespace(teampreview=False)

    assert controller.choose(teampreview_battle) == "team 1234"
    assert controller.choose(turn_battle) == "move 1, move 2"


def test_sync_policy_player_keeps_playing_past_trajectory_cap(monkeypatch):
    player = SyncPolicyPlayer(
        agent=cast(Any, FakeAgent()),
        format_id="gen9vgc2023regc",
        device="cpu",
        collect_trajectories=True,
        probabilistic=False,
        max_battle_steps=1,
    )
    monkeypatch.setattr(player.embedder, "embed", lambda battle: {"battle": battle})
    monkeypatch.setattr(
        player.embedder,
        "feature_dict_to_vector",
        lambda _: [0.1, 0.2, 0.3],
    )
    monkeypatch.setattr(player.embedder, "embed_to_array", lambda battle: np.asarray([0.1, 0.2, 0.3], dtype=np.float32))
    monkeypatch.setattr(player.embedder, "embed_to_vector", lambda battle: [0.1, 0.2, 0.3])
    battle = SimpleNamespace(
        teampreview=False,
        battle_tag="sync-battle-cap",
        opponent_team={
            "mon1": SimpleNamespace(fainted=False),
            "mon2": SimpleNamespace(fainted=False),
        },
    )

    first_choice, first_step = player.choose_action(
        battle_tag="sync-battle-cap",
        battle=battle,
        legal_actions=[(7, "move 1, move 1"), (11, "move 2, move 1")],
    )
    second_choice, second_step = player.choose_action(
        battle_tag="sync-battle-cap",
        battle=battle,
        legal_actions=[(7, "move 1, move 1"), (11, "move 2, move 1")],
    )

    assert first_choice == "move 2, move 1"
    assert first_step is not None
    assert second_choice == "move 2, move 1"
    assert second_step is None
    assert len(player.current_trajectories["sync-battle-cap"]) == 1


def test_sync_policy_player_uses_snapshot_state_vector():
    player = SyncPolicyPlayer(
        agent=cast(Any, FakeAgent()),
        format_id="gen9vgc2023regc",
        device="cpu",
        collect_trajectories=False,
        probabilistic=False,
    )
    battle = SimpleNamespace(
        teampreview=False,
        battle_tag="snapshot-battle",
        opponent_team={
            "mon1": SimpleNamespace(fainted=False),
            "mon2": SimpleNamespace(fainted=False),
        },
    )
    mask = np.zeros(2025, dtype=np.int8)
    mask[11] = 1
    snapshot = BattleSnapshot(
        battle_tag="snapshot-battle",
        side="p1",
        battle=battle,
        request={"active": [{}, {}]},
        legal_actions=[(7, "move 1, move 1"), (11, "move 2, move 1")],
        action_mask=mask,
        action_to_choice={11: "move 2, move 1"},
        is_teampreview=False,
        opponent_fainted=0,
        state_vector=[0.1, 0.2, 0.3],
    )

    choice, rollout_step = player.choose_action_from_snapshot(snapshot)

    assert choice == "move 2, move 1"
    assert rollout_step is None


def test_sync_policy_player_builds_mask_from_snapshot_legal_actions():
    player = SyncPolicyPlayer(
        agent=cast(Any, FakeAgent()),
        format_id="gen9vgc2023regc",
        device="cpu",
        collect_trajectories=True,
        probabilistic=False,
    )
    battle = SimpleNamespace(
        teampreview=False,
        battle_tag="snapshot-mask",
        opponent_team={
            "mon1": SimpleNamespace(fainted=False),
            "mon2": SimpleNamespace(fainted=False),
        },
    )
    snapshot = BattleSnapshot(
        battle_tag="snapshot-mask",
        side="p1",
        battle=battle,
        request={"active": [{}, {}]},
        legal_actions=[(7, "move 1, move 1"), (11, "move 2, move 1")],
        action_mask=None,
        action_to_choice={},
        is_teampreview=False,
        opponent_fainted=0,
        state_vector=[0.1, 0.2, 0.3],
    )

    choice, rollout_step = player.choose_action_from_snapshot(snapshot)

    assert choice == "move 2, move 1"
    assert rollout_step is not None
    assert rollout_step.mask is not None
    assert rollout_step.mask[7] == 1
    assert rollout_step.mask[11] == 1


def test_sync_policy_player_batches_snapshot_inference():
    counting_agent = CountingAgent()
    player = SyncPolicyPlayer(
        agent=cast(Any, counting_agent),
        format_id="gen9vgc2023regc",
        device="cpu",
        collect_trajectories=True,
        probabilistic=False,
    )
    battle_a = SimpleNamespace(
        teampreview=False,
        battle_tag="batch-a",
        opponent_team={"mon1": SimpleNamespace(fainted=False)},
    )
    battle_b = SimpleNamespace(
        teampreview=False,
        battle_tag="batch-b",
        opponent_team={"mon1": SimpleNamespace(fainted=True)},
    )
    snapshots = [
        BattleSnapshot(
            battle_tag="batch-a",
            side="p1",
            battle=battle_a,
            request={"active": [{}, {}]},
            legal_actions=[(7, "move 1, move 1"), (11, "move 2, move 1")],
            action_mask=None,
            action_to_choice={},
            is_teampreview=False,
            opponent_fainted=0,
            state_vector=[0.1, 0.2, 0.3],
        ),
        BattleSnapshot(
            battle_tag="batch-b",
            side="p1",
            battle=battle_b,
            request={"active": [{}, {}]},
            legal_actions=[(7, "move 1, move 1"), (11, "move 2, move 1")],
            action_mask=None,
            action_to_choice={},
            is_teampreview=False,
            opponent_fainted=1,
            state_vector=[0.4, 0.5, 0.6],
        ),
    ]

    results = player.choose_actions_from_snapshots(snapshots, ["ghosts", "self_play"])

    assert [choice for choice, _ in results] == ["move 2, move 1", "move 2, move 1"]
    assert all(step is not None for _, step in results)
    assert set(player.hidden_states) == {"batch-a", "batch-b"}
    assert counting_agent.batch_sizes == [2]


def test_sync_policy_player_trims_transformer_context_to_model_limit():
    player = SyncPolicyPlayer(
        agent=cast(Any, FakeTransformerAgent(max_seq_len=4)),
        format_id="gen9vgc2023regc",
        device="cpu",
        collect_trajectories=False,
        probabilistic=False,
    )
    snapshot = BattleSnapshot(
        battle_tag="transformer-context",
        side="p1",
        battle=SimpleNamespace(teampreview=False, battle_tag="transformer-context", opponent_team={}),
        request={"active": [{}, {}]},
        legal_actions=[(11, "move 2, move 1")],
        action_mask=None,
        action_to_choice={},
        is_teampreview=False,
        opponent_fainted=0,
        state_vector=[0.1, 0.2, 0.3],
    )

    for _ in range(8):
        choice, rollout_step = player.choose_action_from_snapshot(snapshot)
        assert choice == "move 2, move 1"
        assert rollout_step is None

    hidden_state = player.hidden_states["transformer-context"]
    assert hidden_state is not None
    assert hidden_state.size(1) == 4


def test_sync_policy_player_processes_transformer_snapshots_sequentially():
    agent = CountingTransformerAgent(max_seq_len=4)
    player = SyncPolicyPlayer(
        agent=cast(Any, agent),
        format_id="gen9vgc2023regc",
        device="cpu",
        collect_trajectories=False,
        probabilistic=False,
    )
    snapshots = [
        BattleSnapshot(
            battle_tag="transformer-batch-a",
            side="p1",
            battle=SimpleNamespace(teampreview=False, battle_tag="transformer-batch-a", opponent_team={}),
            request={"active": [{}, {}]},
            legal_actions=[(11, "move 2, move 1")],
            action_mask=None,
            action_to_choice={},
            is_teampreview=False,
            opponent_fainted=0,
            state_vector=[0.1, 0.2, 0.3],
        ),
        BattleSnapshot(
            battle_tag="transformer-batch-b",
            side="p1",
            battle=SimpleNamespace(teampreview=False, battle_tag="transformer-batch-b", opponent_team={}),
            request={"active": [{}, {}]},
            legal_actions=[(11, "move 2, move 1")],
            action_mask=None,
            action_to_choice={},
            is_teampreview=False,
            opponent_fainted=0,
            state_vector=[0.4, 0.5, 0.6],
        ),
    ]

    results = player.choose_actions_from_snapshots(snapshots)

    assert [choice for choice, _ in results] == ["move 2, move 1", "move 2, move 1"]
    assert agent.batch_sizes == [1, 1]
    assert agent.hidden_lengths == [0, 0]
    assert player.hidden_states["transformer-batch-a"].size(1) == 1
    assert player.hidden_states["transformer-batch-b"].size(1) == 1


def test_sync_policy_player_separates_transformer_batches_with_different_context_lengths():
    agent = CountingTransformerAgent(max_seq_len=4)
    player = SyncPolicyPlayer(
        agent=cast(Any, agent),
        format_id="gen9vgc2023regc",
        device="cpu",
        collect_trajectories=False,
        probabilistic=False,
    )
    player.hidden_states["transformer-mixed-a"] = torch.zeros(1, 1, 8)
    player.hidden_states["transformer-mixed-b"] = torch.zeros(1, 2, 8)
    snapshots = [
        BattleSnapshot(
            battle_tag="transformer-mixed-a",
            side="p1",
            battle=SimpleNamespace(teampreview=False, battle_tag="transformer-mixed-a", opponent_team={}),
            request={"active": [{}, {}]},
            legal_actions=[(11, "move 2, move 1")],
            action_mask=None,
            action_to_choice={},
            is_teampreview=False,
            opponent_fainted=0,
            state_vector=[0.1, 0.2, 0.3],
        ),
        BattleSnapshot(
            battle_tag="transformer-mixed-b",
            side="p1",
            battle=SimpleNamespace(teampreview=False, battle_tag="transformer-mixed-b", opponent_team={}),
            request={"active": [{}, {}]},
            legal_actions=[(11, "move 2, move 1")],
            action_mask=None,
            action_to_choice={},
            is_teampreview=False,
            opponent_fainted=0,
            state_vector=[0.4, 0.5, 0.6],
        ),
    ]

    results = player.choose_actions_from_snapshots(snapshots)

    assert [choice for choice, _ in results] == ["move 2, move 1", "move 2, move 1"]
    assert agent.batch_sizes == [1, 1]
    assert agent.hidden_lengths == [1, 2]
    assert player.hidden_states["transformer-mixed-a"].size(1) == 2
    assert player.hidden_states["transformer-mixed-b"].size(1) == 3


def test_sync_rust_battle_driver_retries_rejected_choice_with_fallback():
    class FakeFallbackEngine:
        def __init__(self):
            self.battle_tag = "fallback-battle"
            self.turn = 0
            self.ended = False
            self.p1_battle = SimpleNamespace(won=True)

        def needs_action(self, side: str) -> bool:
            return side == "p1" and not self.ended

        def step(self, p1_choice=None, p2_choice=None):
            del p2_choice
            if p1_choice == "bad":
                return False, False, True
            if p1_choice == "good":
                self.turn = 1
                self.ended = True
                return True, True, True
            return False, True, True

    class FakeDriver(SyncRustBattleDriver):
        def __init__(self):
            self._max_turns_per_battle = 10
            self._max_stalled_steps_per_battle = 2
            self._collect_rollouts = False
            self._p1_policy = None
            self._p2_policy = None

        def _new_active_battle(self, battle_index: int):
            del battle_index
            return {
                "engine": FakeFallbackEngine(),
                "trajectory": [],
                "last_turn": 0,
                "stalled_steps": 0,
                "p1_policy": None,
                "p2_policy": None,
                "opponent_type": "self_play",
            }

        def _build_choice_context(self, engine, side: str, policy=None, opponent_type: str = "self_play"):
            del engine, side, policy, opponent_type
            return {"choice": "bad", "fallback_choice": "good", "rollout_step": None}

    driver = FakeDriver()

    stats = driver.run(total_battles=1, max_concurrent=1)

    assert stats.completed_battles == 1
    assert stats.truncated_battles == 0
    assert stats.p1_wins == 1
    assert stats.p1_rejected_choices == 1
    assert stats.p1_fallback_recoveries == 1
    assert stats.p1_unrecovered_rejections == 0


def test_sync_rust_battle_driver_writes_rejection_diagnostics(tmp_path):
    class FakeFallbackEngine:
        def __init__(self):
            self.battle_tag = "fallback-battle"
            self.turn = 0
            self.ended = False
            self.p1_battle = SimpleNamespace(won=True)

        def needs_action(self, side: str) -> bool:
            return side == "p1" and not self.ended

        def request_type(self, side: str) -> str:
            del side
            return "switch"

        def request_json(self, side: str):
            del side
            return {"forceSwitch": [True, False]}

        def step(self, p1_choice=None, p2_choice=None):
            del p2_choice
            if p1_choice == "bad":
                return False, False, True
            if p1_choice == "good":
                self.turn = 1
                self.ended = True
                return True, True, True
            return False, True, True

    class FakeDriver(SyncRustBattleDriver):
        def __init__(self, diagnostic_log_path: str):
            self._max_turns_per_battle = 10
            self._max_stalled_steps_per_battle = 2
            self._collect_rollouts = False
            self._p1_policy = None
            self._p2_policy = None
            self._diagnostic_log_path = diagnostic_log_path

        def _new_active_battle(self, battle_index: int):
            del battle_index
            return {
                "engine": FakeFallbackEngine(),
                "trajectory": [],
                "last_turn": 0,
                "stalled_steps": 0,
                "p1_policy": None,
                "p2_policy": None,
                "opponent_type": "self_play",
            }

        def _build_choice_context(self, engine, side: str, policy=None, opponent_type: str = "self_play"):
            del engine, side, policy, opponent_type
            snapshot = BattleSnapshot(
                battle_tag="fallback-battle",
                side="p1",
                battle=SimpleNamespace(teampreview=False, opponent_team={}),
                request={"forceSwitch": [True, False], "active": [{"trapped": False, "canTerastallize": None}]},
                legal_actions=[(40, "good")],
                action_mask=None,
                action_to_choice={},
                is_teampreview=False,
                opponent_fainted=0,
                state_vector=[0.1],
                binding_snapshot=None,
            )
            return {
                "choice": "bad",
                "fallback_choice": "good",
                "rollout_step": None,
                "snapshot": snapshot,
            }

    log_path = tmp_path / "driver_diagnostics.jsonl"
    driver = FakeDriver(str(log_path))

    stats = driver.run(total_battles=1, max_concurrent=1)

    assert stats.p1_rejected_choices == 1
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1
    event = __import__("json").loads(lines[0])
    assert event["event_type"] == "rejected_choice"
    assert event["request_type"] == "switch"
    assert event["choice"] == "bad"
    assert event["fallback_choice"] == "good"
    assert event["fallback_recovered"] is True
    assert event["protocol_history"] == []
    assert event["protocol_log_before_step"] == []
    assert event["protocol_log_after_step"] == []
    assert event["legal_choices"] == ["good"]
    assert "battle_state_poke_env" in event
    assert "battle_state_request" in event


def test_sync_rust_battle_driver_records_first_error_battle(tmp_path):
    class FakeErrorEngine:
        def __init__(self):
            self.battle_tag = "recorded-battle"
            self.turn = 0
            self.ended = False
            self.winner = None
            self.p1_battle = SimpleNamespace(won=True)

        def needs_action(self, side: str) -> bool:
            return side == "p1" and not self.ended

        def request_type(self, side: str) -> str:
            del side
            return "move"

        def request_json(self, side: str):
            return {
                "active": [
                    {"moves": [{"target": "normal", "pp": 8, "disabled": False}]},
                    {"moves": [{"target": "normal", "pp": 8, "disabled": False}]},
                ],
                "side": {"id": side, "pokemon": [{"active": True, "condition": "100/100"}]},
            }

        def protocol_history(self, side: str, limit: int = 5):
            del limit
            return [{"raw": f"|turn|{self.turn}", "normalized": f"|turn|{self.turn}"}]

        def protocol_log(self, side: str):
            return [{"raw": f"|turn|{self.turn}", "normalized": f"|turn|{self.turn}"}, {"raw": f"|side|{side}", "normalized": f"|side|{side}"}]

        def step(self, p1_choice=None, p2_choice=None):
            del p2_choice
            if p1_choice == "bad":
                return False, False, True
            self.turn = 1
            self.ended = True
            self.winner = "EliteFurretAI-p1"
            return True, True, True

    class FakeDriver(SyncRustBattleDriver):
        def __init__(self, diagnostic_log_path: str, error_battle_record_path: str):
            self._max_turns_per_battle = 10
            self._max_stalled_steps_per_battle = 2
            self._collect_rollouts = False
            self._p1_policy = None
            self._p2_policy = None
            self._diagnostic_log_path = diagnostic_log_path
            self._error_battle_record_path = error_battle_record_path
            self._selected_error_battle_tag = None
            self._format_id = "gen9vgc2024regg"

        def _new_active_battle(self, battle_index: int):
            del battle_index
            return {
                "engine": FakeErrorEngine(),
                "trajectory": [],
                "last_turn": 0,
                "stalled_steps": 0,
                "p1_policy": None,
                "p2_policy": None,
                "opponent_type": "self_play",
                "battle_trace": [],
            }

        def _build_choice_context(self, engine, side: str, policy=None, opponent_type: str = "self_play"):
            del engine, side, policy, opponent_type
            snapshot = BattleSnapshot(
                battle_tag="recorded-battle",
                side="p1",
                battle=SimpleNamespace(teampreview=False, opponent_team={}),
                request={
                    "active": [
                        {"moves": [{"target": "normal", "pp": 8, "disabled": False}]},
                        {"moves": [{"target": "normal", "pp": 8, "disabled": False}]},
                    ],
                    "side": {"id": "p1", "pokemon": [{"active": True, "condition": "100/100"}]},
                },
                legal_actions=[(0, "bad"), (1, "good")],
                action_mask=None,
                action_to_choice={},
                is_teampreview=False,
                opponent_fainted=0,
                state_vector=[0.1],
                binding_snapshot=cast(Any, SimpleNamespace(raw_request='{"raw": true}', pending_messages=("|turn|0",))),
            )
            return {
                "choice": "bad",
                "fallback_choice": "good",
                "rollout_step": None,
                "snapshot": snapshot,
            }

    log_path = tmp_path / "driver_diagnostics.jsonl"
    record_path = tmp_path / "recorded_error_battle.json"
    driver = FakeDriver(str(log_path), str(record_path))

    stats = driver.run(total_battles=1, max_concurrent=1)

    assert stats.p1_rejected_choices == 1
    assert record_path.exists()
    record = __import__("json").loads(record_path.read_text())
    assert record["battle_tag"] == "recorded-battle"
    assert record["protocol_log"]["p1"]
    assert record["trace"]
    assert any(event["event_type"] == "decision_window" for event in record["trace"])
    assert any(event["event_type"] == "decision_result" for event in record["trace"])


def test_force_switch_non_forced_slot_only_returns_pass():
    driver = SyncRustBattleDriver.__new__(SyncRustBattleDriver)
    battle = SimpleNamespace(active_pokemon=[object(), object()])
    request = {
        "forceSwitch": [True, False],
        "active": [
            {"moves": [{"target": "normal", "pp": 8}], "trapped": False},
            {"moves": [{"target": "normal", "pp": 8}], "trapped": False},
        ],
        "side": {
            "pokemon": [
                {"active": True, "condition": "100/100"},
                {"active": True, "condition": "100/100"},
                {"active": False, "condition": "100/100"},
            ]
        },
    }

    actions = driver._get_slot_actions(battle, request, slot=1)

    assert actions == [
        {
            "encoded": 44,
            "choice": "pass",
            "tera": False,
            "switch_target": None,
        }
    ]


def test_force_switch_slot_with_available_switches_does_not_offer_pass_for_healthy_slot():
    driver = SyncRustBattleDriver.__new__(SyncRustBattleDriver)
    battle = SimpleNamespace(active_pokemon=[object(), object()])
    request = {
        "forceSwitch": [True, False],
        "active": [{"moves": [], "trapped": False}, {"moves": [], "trapped": False}],
        "side": {
            "pokemon": [
                {"active": True, "condition": "100/100"},
                {"active": True, "condition": "100/100"},
                {"active": False, "condition": "100/100"},
                {"active": False, "condition": "100/100"},
            ]
        },
    }

    actions = driver._get_slot_actions(battle, request, slot=0)

    assert {action["choice"] for action in actions} == {"switch 3", "switch 4"}


def test_double_force_switch_does_not_offer_pass_when_enough_replacements_exist():
    driver = SyncRustBattleDriver.__new__(SyncRustBattleDriver)
    battle = SimpleNamespace(active_pokemon=[object(), object()])
    request = {
        "forceSwitch": [True, True],
        "side": {
            "pokemon": [
                {"active": False, "condition": "100/100", "_request_index": 0},
                {"active": False, "condition": "100/100", "_request_index": 1},
                {"active": False, "condition": "0 fnt", "_request_index": 2},
                {"active": False, "condition": "100/100", "_request_index": 3},
                {"active": False, "condition": "100/100", "_request_index": 4},
            ]
        },
    }

    slot0_actions = driver._get_slot_actions(battle, request, slot=0)
    slot1_actions = driver._get_slot_actions(battle, request, slot=1)

    assert "pass" not in {action["choice"] for action in slot0_actions}
    assert "pass" not in {action["choice"] for action in slot1_actions}
    assert {action["choice"] for action in slot0_actions} == {"switch 1", "switch 2", "switch 4", "switch 5"}


def test_switch_actions_preserve_original_request_indices_after_sanitization():
    driver = SyncRustBattleDriver.__new__(SyncRustBattleDriver)
    request = {
        "side": {
            "pokemon": [
                {"active": True, "condition": "100/100", "_request_index": 2},
                {"active": True, "condition": "100/100", "_request_index": 5},
                {"active": False, "condition": "0 fnt", "_request_index": 0},
                {"active": False, "condition": "0 fnt", "_request_index": 1},
                {"active": False, "condition": "130/167", "_request_index": 3},
                {"active": False, "condition": "32/181", "_request_index": 4},
            ]
        }
    }

    actions = driver._get_switch_actions(request)

    assert [action["choice"] for action in actions] == ["switch 4", "switch 5"]
    assert [action["switch_target"] for action in actions] == [3, 4]


def test_turn_slot_actions_emit_protocol_faithful_target_strings():
    driver = SyncRustBattleDriver.__new__(SyncRustBattleDriver)
    battle = SimpleNamespace(
        active_pokemon=[
            SimpleNamespace(fainted=False),
            SimpleNamespace(fainted=False),
        ],
        opponent_active_pokemon=[
            SimpleNamespace(fainted=False),
            SimpleNamespace(fainted=False),
        ],
    )
    request = {
        "active": [
            {
                "moves": [{"target": "normal", "pp": 8, "disabled": False}],
                "trapped": False,
            },
            {"moves": [], "trapped": False},
        ],
        "side": {
            "pokemon": [
                {"active": True, "condition": "100/100"},
                {"active": True, "condition": "100/100"},
                {"active": False, "condition": "100/100"},
            ]
        },
    }

    actions = driver._get_slot_actions(battle, request, slot=0)

    assert {action["choice"] for action in actions} == {
        "move 1 -2",
        "move 1 1",
        "move 1 2",
        "switch 3",
    }


def test_commanding_slot_only_returns_pass():
    driver = SyncRustBattleDriver.__new__(SyncRustBattleDriver)
    battle = SimpleNamespace(active_pokemon=[object(), object()])
    request = {
        "active": [
            {
                "moves": [{"target": "normal", "pp": 8, "disabled": False}],
                "trapped": True,
                "canTerastallize": "Grass",
            },
            {
                "moves": [
                    {"target": "normal", "pp": 8, "disabled": False},
                    {"target": "normal", "pp": 8, "disabled": False},
                ],
                "trapped": True,
                "canTerastallize": "Dragon",
            },
        ],
        "side": {
            "pokemon": [
                {"active": True, "condition": "225/225", "commanding": False},
                {"active": True, "condition": "175/175", "commanding": True},
                {"active": False, "condition": "100/100", "commanding": False, "_request_index": 2},
            ]
        },
    }

    actions = driver._get_slot_actions(battle, request, slot=1)

    assert actions == [
        {
            "encoded": 44,
            "choice": "pass",
            "tera": False,
            "switch_target": None,
        }
    ]




def test_sync_rust_battle_driver_reports_stall_truncation_cause():
    class FakeStallEngine:
        def __init__(self):
            self.battle_tag = "stall-battle"
            self.turn = 0
            self.ended = False
            self.p1_battle = SimpleNamespace(won=False)

        def needs_action(self, side: str) -> bool:
            return side == "p1"

        def step(self, p1_choice=None, p2_choice=None):
            del p1_choice, p2_choice
            return False, True, True

    class FakeDriver(SyncRustBattleDriver):
        def __init__(self):
            self._max_turns_per_battle = 10
            self._max_stalled_steps_per_battle = 1
            self._collect_rollouts = False
            self._p1_policy = None
            self._p2_policy = None

        def _new_active_battle(self, battle_index: int):
            del battle_index
            return {
                "engine": FakeStallEngine(),
                "trajectory": [],
                "last_turn": 0,
                "stalled_steps": 0,
                "p1_policy": None,
                "p2_policy": None,
                "opponent_type": "self_play",
            }

        def _build_choice_context(self, engine, side: str, policy=None, opponent_type: str = "self_play"):
            del engine, side, policy, opponent_type
            return {"choice": "good", "fallback_choice": "good", "rollout_step": None}

    driver = FakeDriver()

    stats = driver.run(total_battles=1, max_concurrent=1)

    assert stats.completed_battles == 1
    assert stats.truncated_battles == 1
    assert stats.stalled_limit_truncations == 1
    assert stats.turn_limit_truncations == 0
    assert stats.dual_limit_truncations == 0


def test_sync_rust_battle_driver_reports_turn_truncation_cause():
    class FakeTurnEngine:
        def __init__(self):
            self.battle_tag = "turn-battle"
            self.turn = 11
            self.ended = False
            self.p1_battle = SimpleNamespace(won=False)

        def needs_action(self, side: str) -> bool:
            del side
            return False

    class FakeDriver(SyncRustBattleDriver):
        def __init__(self):
            self._max_turns_per_battle = 10
            self._max_stalled_steps_per_battle = 10
            self._collect_rollouts = False
            self._p1_policy = None
            self._p2_policy = None

        def _new_active_battle(self, battle_index: int):
            del battle_index
            return {
                "engine": FakeTurnEngine(),
                "trajectory": [],
                "last_turn": 11,
                "stalled_steps": 0,
                "p1_policy": None,
                "p2_policy": None,
                "opponent_type": "self_play",
            }

    driver = FakeDriver()

    stats = driver.run(total_battles=1, max_concurrent=1)

    assert stats.completed_battles == 1
    assert stats.truncated_battles == 1
    assert stats.turn_limit_truncations == 1
    assert stats.stalled_limit_truncations == 0
    assert stats.dual_limit_truncations == 0
