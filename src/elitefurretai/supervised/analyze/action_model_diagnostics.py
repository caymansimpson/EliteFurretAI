#!/usr/bin/env python3
"""
Comprehensive diagnostics for action prediction models.

Analyzes:
1. Action type distribution (predicted vs actual)
2. Top-k accuracy breakdown by action type
3. Loss contribution analysis
4. Action confidence scores
5. Move selection patterns (which moves are predicted vs actual)
6. Invalid action predictions
"""

import json
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from elitefurretai.etl import (
    MDBO,
    Embedder,
    OptimizedBattleDataLoader,
)
from elitefurretai.etl.embedder import MOVE_TO_ID
from elitefurretai.supervised.model_archs import TransformerThreeHeadedModel

# Inverse map for decoding move_id features to human-readable names.
_ID_TO_MOVE: Dict[int, str] = {v: k for k, v in MOVE_TO_ID.items()}
_ID_TO_MOVE[0] = "<unknown>"

# Offensive move categories we read from MON:i:MOVE:j:OFF_CAT:<cat>.
_CATEGORIES = ("PHYSICAL", "SPECIAL", "STATUS")


class ActionDiagnostics:
    """Diagnostic analyzer for action prediction models."""

    def __init__(self, model, device="cuda", state_input_dim: int = 0):
        self.model = model
        self.device = device
        self.state_input_dim = state_input_dim  # 0 means no slicing

        # Storage for analysis
        self.action_type_pred = defaultdict(int)
        self.action_type_actual = defaultdict(int)
        self.action_confidences = defaultdict(list)
        self.topk_hits: defaultdict = defaultdict(lambda: defaultdict(int))
        self.loss_contributions = defaultdict(list)
        self.move_predictions: Counter = Counter()
        self.move_actuals: Counter = Counter()
        self.invalid_predictions = 0
        self.total_predictions = 0

        # Per-active-slot decomposition (one entry per slot per timestep, so
        # ~2x total_predictions). Tracks how the model fares on the (move-slot,
        # target) substructure of each TURN action.
        self.slot_buckets: Counter = Counter()
        # Per-single-order favor/disfavor: counts of canonical single-slot
        # orders (e.g. "move 2 1", "switch 3") in actual vs predicted.
        self.per_order_actual: Counter = Counter()
        self.per_order_predicted: Counter = Counter()

        # Per-move-NAME favor/disfavor. Resolved from MON:i:MOVE:j:move_id at
        # the chosen slot. Ambiguity (when both active mons have different
        # moves at the same slot) is handled by uniform-weight splitting, so
        # totals are float-valued. See planning doc for caveats.
        self.per_move_name_actual: defaultdict = defaultdict(float)
        self.per_move_name_predicted: defaultdict = defaultdict(float)
        self.move_name_total_pairs = 0  # # of (actual, pred) move-slot pairs scored
        self.move_name_ambiguous_pairs = 0  # pairs where at least one side split

        # Move-category confusion: keys are (actual_cat, predicted_cat) tuples.
        # Same ambiguity handling — splits weight across all (a, p) combos when
        # either side disagrees across the two active candidates.
        self.category_confusion: defaultdict = defaultdict(float)
        self.category_total_pairs = 0
        self.category_ambiguous_pairs = 0

    def classify_action_type(self, action_idx: int, order_type: str) -> str:
        """
        Classify action type from action index by decoding it to MDBO.

        Args:
            action_idx: MDBO action index
            order_type: One of MDBO.TURN, MDBO.FORCE_SWITCH, MDBO.TEAMPREVIEW

        Returns:
            One of: MOVE, SWITCH, BOTH, OTHER, INVALID
        """
        try:
            # Use MDBO.from_int() static method to decode
            mdbo_order = MDBO.from_int(int(action_idx), order_type)
            if mdbo_order is None:
                return "INVALID"

            # Get the message string (e.g., "/choose move 1, switch 2")
            message = mdbo_order.message.lower()

            # Parse the message to determine action type
            # Remove the "/choose " prefix
            if message.startswith("/choose "):
                orders = message[8:].split(", ")
            else:
                return "OTHER"

            # Count moves and switches
            has_move = any("move" in order for order in orders)
            has_switch = any("switch" in order for order in orders)

            if has_move and has_switch:
                return "BOTH"
            elif has_switch:
                return "SWITCH"
            elif has_move:
                return "MOVE"
            else:
                return "OTHER"
        except Exception:
            return "INVALID"

    def _parse_order(self, order_str: str) -> Optional[Dict[str, Any]]:
        """Parse one per-active-slot order string (e.g. "move 2 1 terastallize")."""
        order_str = order_str.strip()
        if order_str == "pass":
            return {"kind": "pass"}
        if order_str == "default":
            return {"kind": "default"}
        if order_str.startswith("switch "):
            try:
                return {"kind": "switch", "slot": int(order_str.split(" ", 1)[1])}
            except (ValueError, IndexError):
                return None
        if order_str.startswith("move "):
            tera = "terastallize" in order_str
            body = order_str[5:].replace(" terastallize", "").strip()
            parts = body.split()
            if not parts:
                return None
            try:
                slot = int(parts[0])
            except ValueError:
                return None
            target: Optional[int] = None
            if len(parts) >= 2:
                try:
                    target = int(parts[1])
                except ValueError:
                    target = None
            return {"kind": "move", "slot": slot, "target": target, "tera": tera}
        return None

    def _decompose_action(
        self, action_idx: int, order_type: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Decode an MDBO action int into a list of two per-active-slot orders."""
        try:
            mdbo = MDBO.from_int(int(action_idx), order_type)
        except Exception:
            return None
        if mdbo is None:
            return None
        msg = mdbo.message
        if not msg.startswith("/choose "):
            return None
        orders = msg[8:].split(", ")
        if len(orders) != 2:
            return None
        parsed = [self._parse_order(o) for o in orders]
        if any(p is None for p in parsed):
            return None
        return parsed  # type: ignore[return-value]

    def _order_to_canonical(self, parsed: Dict[str, Any]) -> str:
        """Canonical string form of a parsed order for counting (tera kept distinct)."""
        kind = parsed["kind"]
        if kind in ("pass", "default"):
            return kind
        if kind == "switch":
            return f"switch {parsed['slot']}"
        if kind == "move":
            s = f"move {parsed['slot']}"
            if parsed.get("target") is not None:
                s += f" {parsed['target']}"
            if parsed.get("tera"):
                s += " tera"
            return s
        return "other"

    def _classify_slot_pair(
        self, actual: Dict[str, Any], predicted: Dict[str, Any]
    ) -> str:
        """Bucket a single active-slot's (actual, predicted) order pair.

        Buckets answer: did we get the kind right (move/switch/pass)? If both
        moves, did we get the slot and target right separately?
        """
        a_kind, p_kind = actual["kind"], predicted["kind"]
        if a_kind != p_kind:
            return f"kind_mismatch_actual_{a_kind}_pred_{p_kind}"
        if a_kind == "move":
            slot_match = actual["slot"] == predicted["slot"]
            target_match = actual.get("target") == predicted.get("target")
            if slot_match and target_match:
                return "move_correct"
            if slot_match:
                return "move_slot_correct_target_wrong"
            if target_match:
                return "move_target_correct_slot_wrong"
            return "move_both_wrong"
        if a_kind == "switch":
            return (
                "switch_correct"
                if actual["slot"] == predicted["slot"]
                else "switch_wrong_slot"
            )
        if a_kind == "pass":
            return "pass_match"
        return "other_match"

    def _active_mon_indices(
        self, state_t: torch.Tensor, mon_active_indices: List[int]
    ) -> List[int]:
        """Return team-mon indices (MON:0..5) currently flagged active."""
        return [i for i, idx in enumerate(mon_active_indices) if state_t[idx].item() > 0.5]

    def _mon_at_active_slot(
        self, state_t: torch.Tensor, mon_active_slot_indices: List[int]
    ) -> Optional[int]:
        """Return the unique MON:i index occupying a given active slot (0 or 1).

        Reads MON:i:active_slot_<k> features (added 2026-05-24 in the
        embedder). Returns None if no mon is flagged at that slot — happens
        for slots without an active mon (force-switch holes, etc.).
        """
        for i, idx in enumerate(mon_active_slot_indices):
            if state_t[idx].item() > 0.5:
                return i
        return None

    def _lookup_category(
        self,
        state_t: torch.Tensor,
        mon_idx: int,
        move_slot_0idx: int,
        off_cat_indices: Dict[tuple, int],
    ) -> str:
        """Return PHYSICAL / SPECIAL / STATUS for (mon, move-slot). UNKNOWN if no flag set."""
        for cat in _CATEGORIES:
            key = (mon_idx, move_slot_0idx, cat)
            if key in off_cat_indices and state_t[off_cat_indices[key]].item() > 0.5:
                return cat
        return "UNKNOWN"

    def _lookup_move_name(
        self,
        state_t: torch.Tensor,
        mon_idx: int,
        move_slot_0idx: int,
        move_id_indices: Dict[tuple, int],
    ) -> str:
        """Return the move's human-readable id (e.g. 'protect') for (mon, move-slot)."""
        key = (mon_idx, move_slot_0idx)
        if key not in move_id_indices:
            return "<unknown>"
        mid = int(state_t[move_id_indices[key]].item())
        return _ID_TO_MOVE.get(mid, "<unknown>")

    def analyze_batch(self, batch: Dict[str, torch.Tensor], feature_idx: Dict[str, int]):
        """
        Analyze a single batch of predictions.

        Args:
            batch: Dictionary with keys 'states', 'actions', 'action_masks', 'wins', 'masks'
            feature_idx: Dictionary mapping feature names to indices
        """
        self.model.eval()

        # Extract from batch dictionary
        states = batch["states"].to(torch.float32).to(self.device)
        if self.state_input_dim > 0 and self.state_input_dim < states.shape[-1]:
            states = states[..., : self.state_input_dim]
        actions = batch["actions"].to(self.device)
        masks = batch["action_masks"].to(self.device)
        padding_mask = batch["masks"].to(self.device)

        with torch.no_grad():
            # Get predictions (handle both 3-return and 4-return models)
            outputs = self.model(states, padding_mask)
            if len(outputs) == 4:
                turn_logits, teampreview_logits, win_logits, _ = outputs
            else:
                turn_logits, teampreview_logits, win_logits = outputs

            # Apply action masking
            turn_logits_masked = turn_logits.clone()
            turn_logits_masked[~masks.bool()] = float("-inf")

            # Get top-k predictions
            probs = torch.softmax(turn_logits_masked, dim=-1)
            topk_values, topk_indices = torch.topk(probs, k=10, dim=-1)

            # Analyze each timestep
            batch_size, seq_len = actions.shape

            for b in range(batch_size):
                for t in range(seq_len):
                    # Skip padding
                    if not padding_mask[b, t]:
                        continue

                    actual_action = int(actions[b, t].item())
                    predicted_action = int(topk_indices[b, t, 0].item())
                    confidence = topk_values[b, t, 0].item()

                    # Check if prediction is valid
                    if not masks[b, t, predicted_action]:
                        self.invalid_predictions += 1

                    self.total_predictions += 1

                    # Classify action types
                    # Exclude teampreview steps from all calculations
                    is_teampreview = states[b, t, feature_idx["teampreview"]].item() > 0.5
                    if is_teampreview:
                        continue

                    is_force_switch = False
                    for idx in feature_idx["force_switch_indices"]:  # type: ignore
                        if states[b, t, idx].item() > 0.5:
                            is_force_switch = True
                            break

                    if is_force_switch:
                        action_type = "FORCE_SWITCH"
                        order_type = MDBO.FORCE_SWITCH
                    else:
                        # Only analyze regular turn actions (MOVE/SWITCH/BOTH/OTHER/INVALID)
                        order_type = MDBO.TURN
                        action_type = self.classify_action_type(actual_action, order_type)

                    # Record action type distributions
                    self.action_type_actual[action_type] += 1
                    pred_type = self.classify_action_type(predicted_action, order_type)
                    self.action_type_pred[pred_type] += 1

                    # Record confidence
                    self.action_confidences[action_type].append(confidence)

                    # Check top-k accuracy
                    topk_actions = topk_indices[b, t, :].cpu().numpy()
                    for k in [1, 3, 5, 10]:
                        if actual_action in topk_actions[:k]:
                            self.topk_hits[action_type][f"top{k}"] += 1

                    # Calculate loss contribution
                    ce_loss = -torch.log(probs[b, t, actual_action] + 1e-10).item()
                    self.loss_contributions[action_type].append(ce_loss)

                    # Track move predictions (for MOVE action types)
                    if action_type == "MOVE":
                        self.move_actuals[actual_action] += 1
                        self.move_predictions[predicted_action] += 1

                    # Per-active-slot decomposition: split each MDBO into its
                    # two per-mon orders so we can separate move-slot vs target
                    # vs kind mistakes, and track per-single-order favor/bias.
                    actual_decomp = self._decompose_action(actual_action, order_type)
                    pred_decomp = self._decompose_action(predicted_action, order_type)
                    if actual_decomp is not None and pred_decomp is not None:
                        mon_active_indices = feature_idx.get("mon_active_indices")
                        move_id_indices = feature_idx.get("move_id_indices")
                        move_offcat_indices = feature_idx.get("move_offcat_indices")
                        slot_0_indices = feature_idx.get("mon_active_slot_0_indices")
                        slot_1_indices = feature_idx.get("mon_active_slot_1_indices")
                        state_t = states[b, t]
                        active_mons = (
                            self._active_mon_indices(state_t, mon_active_indices)  # type: ignore[arg-type]
                            if mon_active_indices is not None
                            else []
                        )
                        # Resolve which team-mon occupies each MDBO active slot
                        # (deterministic when active_slot_0/1 features are
                        # present). On checkpoints predating the 2026-05-24
                        # embedder, these are None and we fall back to the old
                        # uniform-split logic.
                        mon_at_mdbo_slot: List[Optional[int]] = [None, None]
                        positional_available = (
                            slot_0_indices is not None and slot_1_indices is not None
                        )
                        if positional_available:
                            mon_at_mdbo_slot[0] = self._mon_at_active_slot(
                                state_t,
                                slot_0_indices,  # type: ignore[arg-type]
                            )
                            mon_at_mdbo_slot[1] = self._mon_at_active_slot(
                                state_t,
                                slot_1_indices,  # type: ignore[arg-type]
                            )

                        for slot_i in range(2):
                            a_slot = actual_decomp[slot_i]
                            p_slot = pred_decomp[slot_i]
                            self.slot_buckets[
                                self._classify_slot_pair(a_slot, p_slot)
                            ] += 1
                            self.per_order_actual[self._order_to_canonical(a_slot)] += 1
                            self.per_order_predicted[self._order_to_canonical(p_slot)] += 1

                            # Category + move-name only apply when both sides
                            # picked a move (else "category" of a switch is
                            # undefined). Skip out-of-range slots (struggle).
                            if (
                                a_slot["kind"] != "move"
                                or p_slot["kind"] != "move"
                                or move_id_indices is None
                                or move_offcat_indices is None
                            ):
                                continue
                            a_slot_0idx = a_slot["slot"] - 1
                            p_slot_0idx = p_slot["slot"] - 1
                            if not (0 <= a_slot_0idx < 4) or not (0 <= p_slot_0idx < 4):
                                continue

                            if positional_available:
                                # Deterministic per-MDBO-slot attribution.
                                # Same mon owns the actual and predicted order
                                # at MDBO slot slot_i — it's determined by
                                # battle state, not by the chosen action.
                                owner = mon_at_mdbo_slot[slot_i]
                                if owner is None:
                                    continue  # no mon at this slot; skip
                                a_cats = {
                                    self._lookup_category(
                                        state_t, owner, a_slot_0idx, move_offcat_indices
                                    )
                                }
                                p_cats = {
                                    self._lookup_category(
                                        state_t, owner, p_slot_0idx, move_offcat_indices
                                    )
                                }
                                a_names = {
                                    self._lookup_move_name(
                                        state_t, owner, a_slot_0idx, move_id_indices
                                    )
                                }
                                p_names = {
                                    self._lookup_move_name(
                                        state_t, owner, p_slot_0idx, move_id_indices
                                    )
                                }
                            else:
                                # Legacy fallback for checkpoints without the
                                # positional features: uniform split over both
                                # active candidates.
                                if not active_mons:
                                    continue
                                a_cats = {
                                    self._lookup_category(
                                        state_t, mi, a_slot_0idx, move_offcat_indices
                                    )
                                    for mi in active_mons
                                }
                                p_cats = {
                                    self._lookup_category(
                                        state_t, mi, p_slot_0idx, move_offcat_indices
                                    )
                                    for mi in active_mons
                                }
                                a_names = {
                                    self._lookup_move_name(
                                        state_t, mi, a_slot_0idx, move_id_indices
                                    )
                                    for mi in active_mons
                                }
                                p_names = {
                                    self._lookup_move_name(
                                        state_t, mi, p_slot_0idx, move_id_indices
                                    )
                                    for mi in active_mons
                                }

                            ambiguous_cat = len(a_cats) > 1 or len(p_cats) > 1
                            self.category_total_pairs += 1
                            if ambiguous_cat:
                                self.category_ambiguous_pairs += 1
                            w_cat = 1.0 / (len(a_cats) * len(p_cats))
                            for ac in a_cats:
                                for pc in p_cats:
                                    self.category_confusion[(ac, pc)] += w_cat

                            ambiguous_name = len(a_names) > 1 or len(p_names) > 1
                            self.move_name_total_pairs += 1
                            if ambiguous_name:
                                self.move_name_ambiguous_pairs += 1
                            wa = 1.0 / len(a_names)
                            wp = 1.0 / len(p_names)
                            for n in a_names:
                                self.per_move_name_actual[n] += wa
                            for n in p_names:
                                self.per_move_name_predicted[n] += wp

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive diagnostic report."""

        report: Dict[str, Dict] = {
            "action_type_distribution": {},
            "topk_accuracy_by_type": {},
            "confidence_stats": {},
            "loss_contribution": {},
            "move_analysis": {},
            "invalid_predictions": {
                "count": self.invalid_predictions,
                "rate": self.invalid_predictions / max(self.total_predictions, 1),
            },
        }

        # Action type distribution
        for action_type in set(
            list(self.action_type_actual.keys()) + list(self.action_type_pred.keys())
        ):
            report["action_type_distribution"][action_type] = {
                "actual_count": self.action_type_actual[action_type],
                "actual_pct": self.action_type_actual[action_type]
                / max(sum(self.action_type_actual.values()), 1),
                "predicted_count": self.action_type_pred[action_type],
                "predicted_pct": self.action_type_pred[action_type]
                / max(sum(self.action_type_pred.values()), 1),
                "prediction_bias": self.action_type_pred[action_type]
                / max(sum(self.action_type_pred.values()), 1)
                - self.action_type_actual[action_type]
                / max(sum(self.action_type_actual.values()), 1),
            }

        # Top-k accuracy by type
        for action_type, hits in self.topk_hits.items():
            total = self.action_type_actual[action_type]
            report["topk_accuracy_by_type"][action_type] = {
                "total": total,
                **{k: v / max(total, 1) for k, v in hits.items()},
            }

        # Overall top-k accuracy (aggregated across all action types)
        total_actions = sum(self.action_type_actual.values())
        overall_topk = {}
        for k in [1, 3, 5, 10]:
            total_hits = sum(hits.get(f"top{k}", 0) for hits in self.topk_hits.values())
            overall_topk[f"top{k}"] = total_hits / max(total_actions, 1)
        report["overall_accuracy"] = {"total": total_actions, **overall_topk}

        # Confidence stats
        for action_type, confidences in self.action_confidences.items():
            if confidences:
                report["confidence_stats"][action_type] = {
                    "mean": float(np.mean(confidences)),
                    "std": float(np.std(confidences)),
                    "median": float(np.median(confidences)),
                    "min": float(np.min(confidences)),
                    "max": float(np.max(confidences)),
                }

        # Loss contribution
        total_loss = sum(sum(losses) for losses in self.loss_contributions.values())
        for action_type, losses in self.loss_contributions.items():
            if losses:
                report["loss_contribution"][action_type] = {
                    "total_loss": float(np.sum(losses)),
                    "mean_loss": float(np.mean(losses)),
                    "pct_of_total": float(np.sum(losses) / max(total_loss, 1)),
                    "count": len(losses),
                }

        # Per-mon kind confusion (question #2): for each active mon per turn,
        # was the actual action a move / switch / pass, and what did the model
        # predict? Each MDBO slot maps to exactly one active mon, so the
        # per-slot buckets are equivalent to per-mon datapoints (2 per turn).
        # The matrix uses bucket prefixes to recover (actual_kind, pred_kind).
        kind_keys = ("MOVE", "SWITCH", "PASS", "OTHER")
        kind_matrix: Dict[str, Dict[str, int]] = {
            a: {p: 0 for p in kind_keys} for a in kind_keys
        }
        for bucket, count in self.slot_buckets.items():
            if bucket.startswith("move_"):
                kind_matrix["MOVE"]["MOVE"] += count
            elif bucket.startswith("switch_"):
                kind_matrix["SWITCH"]["SWITCH"] += count
            elif bucket == "pass_match":
                kind_matrix["PASS"]["PASS"] += count
            elif bucket.startswith("kind_mismatch_actual_"):
                # form: kind_mismatch_actual_<a>_pred_<p>
                parts = bucket[len("kind_mismatch_actual_") :].split("_pred_")
                if len(parts) == 2:
                    a, p = parts[0].upper(), parts[1].upper()
                    if a in kind_matrix and p in kind_matrix[a]:
                        kind_matrix[a][p] += count
            else:
                kind_matrix["OTHER"]["OTHER"] += count
        total_kind = sum(sum(row.values()) for row in kind_matrix.values())
        report["per_mon_kind_confusion"] = {
            "total_datapoints": total_kind,
            "note": (
                "Datapoint = (active mon, turn). Two per turn except when a "
                "slot has no order (FORCE_SWITCH edge cases). Off-diagonal "
                "cells are the model swapping a move for a switch or vice "
                "versa for that mon."
            ),
            "matrix": kind_matrix,
        }

        # Per-slot decomposition (move-slot vs target vs kind mistakes)
        total_slot_pairs = sum(self.slot_buckets.values())
        report["per_slot_decomposition"] = {
            "total_slot_pairs": total_slot_pairs,
            "buckets": {
                bucket: {
                    "count": count,
                    "pct": count / max(total_slot_pairs, 1),
                }
                for bucket, count in sorted(
                    self.slot_buckets.items(), key=lambda kv: -kv[1]
                )
            },
        }

        # Move-vs-move decomposition trees (question #3): two views of the
        # same 4-cell 2x2, each showing conditional probabilities at the
        # inner nodes. Tree 1 splits on move-id-correctness first; tree 2
        # on target-correctness first. Comparing P(target | move id ✓) vs
        # P(target | move id ✗) tells whether move-id and target errors
        # are independent or correlated. "move id" here = the move slot
        # 1-4 within the mon's moveset (not the MDBO active slot).
        b = self.slot_buckets
        n_mc = b.get("move_correct", 0)
        n_sct_w = b.get("move_slot_correct_target_wrong", 0)
        n_tct_w = b.get("move_target_correct_slot_wrong", 0)
        n_bw = b.get("move_both_wrong", 0)
        n_mvm = n_mc + n_sct_w + n_tct_w + n_bw  # total move-vs-move pairs

        def _safe_div(x: float, y: float) -> float:
            return x / y if y > 0 else 0.0

        tree_move_id_first = {
            "total": n_mvm,
            "move_id_correct": {
                "count": n_mc + n_sct_w,
                "share_of_total": _safe_div(n_mc + n_sct_w, n_mvm),
                "p_target_correct_given_move_id_correct": _safe_div(n_mc, n_mc + n_sct_w),
                "target_correct": n_mc,
                "target_wrong": n_sct_w,
            },
            "move_id_wrong": {
                "count": n_tct_w + n_bw,
                "share_of_total": _safe_div(n_tct_w + n_bw, n_mvm),
                "p_target_correct_given_move_id_wrong": _safe_div(n_tct_w, n_tct_w + n_bw),
                "target_correct": n_tct_w,
                "target_wrong": n_bw,
            },
        }
        tree_target_first = {
            "total": n_mvm,
            "target_correct": {
                "count": n_mc + n_tct_w,
                "share_of_total": _safe_div(n_mc + n_tct_w, n_mvm),
                "p_move_id_correct_given_target_correct": _safe_div(n_mc, n_mc + n_tct_w),
                "move_id_correct": n_mc,
                "move_id_wrong": n_tct_w,
            },
            "target_wrong": {
                "count": n_sct_w + n_bw,
                "share_of_total": _safe_div(n_sct_w + n_bw, n_mvm),
                "p_move_id_correct_given_target_wrong": _safe_div(n_sct_w, n_sct_w + n_bw),
                "move_id_correct": n_sct_w,
                "move_id_wrong": n_bw,
            },
        }
        report["move_vs_move_trees"] = {
            "total_move_vs_move_pairs": n_mvm,
            "tree_move_id_first": tree_move_id_first,
            "tree_target_first": tree_target_first,
        }

        # Per-move-NAME favor/disfavor. Compares actual vs predicted
        # distribution over move *names* (e.g. "protect", "fakeout"), not
        # slot ints. Ambiguity (two active mons with different moves at the
        # same slot) is split uniformly — see analyze_batch.
        total_a_names = sum(self.per_move_name_actual.values())
        total_p_names = sum(self.per_move_name_predicted.values())
        all_names = set(self.per_move_name_actual) | set(self.per_move_name_predicted)
        name_rows = []
        for name in all_names:
            a_count = self.per_move_name_actual[name]
            p_count = self.per_move_name_predicted[name]
            if a_count + p_count < 10:  # noise floor (counts are floats due to weights)
                continue
            a_pct = a_count / max(total_a_names, 1.0)
            p_pct = p_count / max(total_p_names, 1.0)
            name_rows.append(
                {
                    "move": name,
                    "actual_count": a_count,
                    "predicted_count": p_count,
                    "actual_pct": a_pct,
                    "predicted_pct": p_pct,
                    "bias": p_pct - a_pct,
                }
            )
        name_rows.sort(key=lambda r: r["bias"])
        amb_pct = self.move_name_ambiguous_pairs / max(self.move_name_total_pairs, 1)
        report["per_move_name_favor"] = {
            "total_actual_weighted": total_a_names,
            "total_predicted_weighted": total_p_names,
            "total_pairs": self.move_name_total_pairs,
            "ambiguous_pairs": self.move_name_ambiguous_pairs,
            "ambiguous_pct": amb_pct,
            "min_joint_count_threshold": 10,
            "top_under_predicted": name_rows[:15],
            "top_over_predicted": list(reversed(name_rows[-15:])),
        }

        # Move-category confusion (question #4): how often the model swaps
        # status moves for damaging ones (and vice versa). Cells are floats
        # because ambiguous timesteps split weight uniformly.
        cat_total = sum(self.category_confusion.values())
        cat_keys = list(_CATEGORIES) + ["UNKNOWN"]
        confusion: Dict[str, Dict[str, float]] = {
            a: {p: float(self.category_confusion.get((a, p), 0.0)) for p in cat_keys}
            for a in cat_keys
        }
        # Off-diagonal share (any actual_cat → different predicted_cat)
        off_diag = sum(v for (a, p), v in self.category_confusion.items() if a != p)

        # Damaging (PHYSICAL+SPECIAL) vs STATUS — collapsed 2x2 view
        def _is_dmg(c: str) -> bool:
            return c in ("PHYSICAL", "SPECIAL")

        collapsed = {
            "actual_DAMAGING_pred_DAMAGING": 0.0,
            "actual_DAMAGING_pred_STATUS": 0.0,
            "actual_STATUS_pred_DAMAGING": 0.0,
            "actual_STATUS_pred_STATUS": 0.0,
        }
        for (a, p), v in self.category_confusion.items():
            if a == "UNKNOWN" or p == "UNKNOWN":
                continue
            ak = "DAMAGING" if _is_dmg(a) else "STATUS"
            pk = "DAMAGING" if _is_dmg(p) else "STATUS"
            collapsed[f"actual_{ak}_pred_{pk}"] += v
        report["category_confusion"] = {
            "total_weighted_pairs": cat_total,
            "total_pairs": self.category_total_pairs,
            "ambiguous_pairs": self.category_ambiguous_pairs,
            "ambiguous_pct": (
                self.category_ambiguous_pairs / max(self.category_total_pairs, 1)
            ),
            "off_diagonal_pct": off_diag / max(cat_total, 1.0),
            "matrix": confusion,
            "collapsed_damaging_vs_status": collapsed,
        }

        # Move analysis
        if self.move_actuals:
            most_common_actual = self.move_actuals.most_common(20)
            most_common_pred = self.move_predictions.most_common(20)

            report["move_analysis"] = {
                "unique_moves_actual": len(self.move_actuals),
                "unique_moves_predicted": len(self.move_predictions),
                "most_common_actual": [
                    {"action": action, "count": count}
                    for action, count in most_common_actual
                ],
                "most_common_predicted": [
                    {"action": action, "count": count}
                    for action, count in most_common_pred
                ],
                "overlap": len(
                    set(self.move_actuals.keys()) & set(self.move_predictions.keys())
                ),
            }

        return report

    def print_report(self, report: Dict[str, Any]):
        """Pretty print the diagnostic report."""

        print("\n" + "=" * 60)
        print("ACTION MODEL DIAGNOSTIC REPORT")
        print("=" * 60)

        # Print overall accuracy first
        print("\n### OVERALL ACCURACY ###")
        oa = report["overall_accuracy"]
        print(f"Total predictions: {oa['total']}")
        print(f"Top-1:  {oa.get('top1', 0) * 100:>6.2f}%")
        print(f"Top-3:  {oa.get('top3', 0) * 100:>6.2f}%")
        print(f"Top-5:  {oa.get('top5', 0) * 100:>6.2f}%")
        print(f"Top-10: {oa.get('top10', 0) * 100:>6.2f}%")

        print("\n### ACTION TYPE DISTRIBUTION ###")
        print(f"{'Type':<15} {'Actual %':<12} {'Pred %':<12} {'Bias':<12}")
        print("-" * 60)
        for action_type, stats in sorted(report["action_type_distribution"].items()):
            print(
                f"{action_type:<15} {stats['actual_pct'] * 100:>10.2f}% {stats['predicted_pct'] * 100:>10.2f}% {stats['prediction_bias'] * 100:>+10.2f}%"
            )

        print("\n### TOP-K ACCURACY BY ACTION TYPE ###")
        print(
            f"{'Type':<15} {'Count':<10} {'Top-1':<10} {'Top-3':<10} {'Top-5':<10} {'Top-10':<10}"
        )
        print("-" * 75)
        for action_type, stats in sorted(report["topk_accuracy_by_type"].items()):
            print(
                f"{action_type:<15} {stats['total']:<10} {stats.get('top1', 0) * 100:>8.2f}% {stats.get('top3', 0) * 100:>8.2f}% {stats.get('top5', 0) * 100:>8.2f}% {stats.get('top10', 0) * 100:>8.2f}%"
            )

        print("\n### PREDICTION CONFIDENCE BY TYPE ###")
        print(
            f"{'Type':<15} {'Mean':<10} {'Std':<10} {'Median':<10} {'Min':<10} {'Max':<10}"
        )
        print("-" * 75)
        for action_type, stats in sorted(report["confidence_stats"].items()):
            print(
                f"{action_type:<15} {stats['mean']:>8.4f}  {stats['std']:>8.4f}  {stats['median']:>8.4f}  {stats['min']:>8.4f}  {stats['max']:>8.4f}"
            )

        print("\n### LOSS CONTRIBUTION BY TYPE ###")
        print(
            f"{'Type':<15} {'Total Loss':<15} {'Mean Loss':<15} {'% of Total':<15} {'Count':<10}"
        )
        print("-" * 75)
        for action_type, stats in sorted(
            report["loss_contribution"].items(),
            key=lambda x: x[1]["total_loss"],
            reverse=True,
        ):
            print(
                f"{action_type:<15} {stats['total_loss']:>13.2f}  {stats['mean_loss']:>13.4f}  {stats['pct_of_total'] * 100:>13.2f}% {stats['count']:>10}"
            )

        if "move_analysis" in report and report["move_analysis"]:
            print("\n### MOVE PREDICTION ANALYSIS ###")
            ma = report["move_analysis"]
            print(f"Unique moves (actual): {ma['unique_moves_actual']}")
            print(f"Unique moves (predicted): {ma['unique_moves_predicted']}")
            print(f"Overlap: {ma['overlap']}")

            print("\nMost common actual moves:")
            for i, move in enumerate(ma["most_common_actual"][:10], 1):
                print(
                    f"  {i}. Action {move['action']}: {move['count']} times ({MDBO.from_int(move['action'], MDBO.TURN).message if MDBO.from_int(move['action'], MDBO.TURN) else 'N/A'})"
                )

            print("\nMost common predicted moves:")
            for i, move in enumerate(ma["most_common_predicted"][:10], 1):
                print(
                    f"  {i}. Action {move['action']}: {move['count']} times ({MDBO.from_int(move['action'], MDBO.TURN).message if MDBO.from_int(move['action'], MDBO.TURN) else 'N/A'})"
                )

        if (
            "per_mon_kind_confusion" in report
            and report["per_mon_kind_confusion"]["total_datapoints"]
        ):
            kc = report["per_mon_kind_confusion"]
            print("\n### PER-MON KIND CONFUSION (move/switch/pass) ###")
            print(
                f"Total datapoints: {kc['total_datapoints']} "
                f"(one per active mon per turn). Rows = actual kind, cols = predicted."
            )
            kind_keys = ("MOVE", "SWITCH", "PASS", "OTHER")
            hdr = f"{'actual ↓ / pred →':<22}" + "".join(f"{k:>10}" for k in kind_keys)
            print(hdr)
            print("-" * len(hdr))
            for a in kind_keys:
                row_total = sum(kc["matrix"][a].values())
                row_cells = "".join(f"{kc['matrix'][a][p]:>10}" for p in kind_keys)
                print(f"{a + ' (n=' + str(row_total) + ')':<22}{row_cells}")

        if (
            "per_slot_decomposition" in report
            and report["per_slot_decomposition"]["total_slot_pairs"]
        ):
            psd = report["per_slot_decomposition"]
            print("\n### PER-SLOT DECOMPOSITION (per active-mon order) ###")
            print(f"Total slot pairs analyzed: {psd['total_slot_pairs']}")
            print(f"{'Bucket':<48} {'Count':>10} {'Pct':>8}")
            print("-" * 70)
            for bucket, stats in psd["buckets"].items():
                print(f"{bucket:<48} {stats['count']:>10} {stats['pct'] * 100:>7.2f}%")

        if (
            "move_vs_move_trees" in report
            and report["move_vs_move_trees"]["total_move_vs_move_pairs"]
        ):
            tt = report["move_vs_move_trees"]
            n_total = tt["total_move_vs_move_pairs"]
            t1 = tt["tree_move_id_first"]
            t2 = tt["tree_target_first"]
            print("\n### MOVE-vs-MOVE DECOMPOSITION TREES (both sides chose a move) ###")
            print(
                '("move id" = the move slot 1-4 within the mon\'s moveset, '
                "not the MDBO active slot)"
            )
            print(f"Total move-vs-move pairs: {n_total}\n")

            print("Tree 1 — move id first → target:")
            mc = t1["move_id_correct"]
            mw = t1["move_id_wrong"]
            print(
                f"  └─ move id CORRECT   n={mc['count']:>7} "
                f"({mc['share_of_total'] * 100:>5.2f}%)   "
                f"P(target ✓ | move id ✓) = {mc['p_target_correct_given_move_id_correct'] * 100:>5.2f}%"
            )
            print(f"      ├─ target ✓        n={mc['target_correct']:>7}")
            print(f"      └─ target ✗        n={mc['target_wrong']:>7}")
            print(
                f"  └─ move id WRONG     n={mw['count']:>7} "
                f"({mw['share_of_total'] * 100:>5.2f}%)   "
                f"P(target ✓ | move id ✗) = {mw['p_target_correct_given_move_id_wrong'] * 100:>5.2f}%"
            )
            print(f"      ├─ target ✓        n={mw['target_correct']:>7}")
            print(f"      └─ target ✗        n={mw['target_wrong']:>7}")

            print("\nTree 2 — target first → move id:")
            tc = t2["target_correct"]
            tw = t2["target_wrong"]
            print(
                f"  └─ target CORRECT    n={tc['count']:>7} "
                f"({tc['share_of_total'] * 100:>5.2f}%)   "
                f"P(move id ✓ | target ✓) = {tc['p_move_id_correct_given_target_correct'] * 100:>5.2f}%"
            )
            print(f"      ├─ move id ✓       n={tc['move_id_correct']:>7}")
            print(f"      └─ move id ✗       n={tc['move_id_wrong']:>7}")
            print(
                f"  └─ target WRONG      n={tw['count']:>7} "
                f"({tw['share_of_total'] * 100:>5.2f}%)   "
                f"P(move id ✓ | target ✗) = {tw['p_move_id_correct_given_target_wrong'] * 100:>5.2f}%"
            )
            print(f"      ├─ move id ✓       n={tw['move_id_correct']:>7}")
            print(f"      └─ move id ✗       n={tw['move_id_wrong']:>7}")

        if "category_confusion" in report and report["category_confusion"]["total_pairs"]:
            cc = report["category_confusion"]
            print("\n### MOVE CATEGORY CONFUSION (actual rows × predicted cols) ###")
            print(
                "  ambiguous = the two active mons have different categories "
                "at the chosen move slot."
            )
            print(
                "  We can't tell from the state tensor which active mon owns "
                "MDBO slot 0 vs slot 1,"
            )
            print(
                "  so on ambiguous timesteps we split unit weight uniformly "
                "across all (actual_cat,"
            )
            print("  predicted_cat) combinations consistent with the two candidates.")
            print(
                f"\nTotal move-vs-move slot pairs: {cc['total_pairs']} "
                f"(ambiguous {cc['ambiguous_pairs']} = {cc['ambiguous_pct'] * 100:.1f}%)"
            )
            cats = list(_CATEGORIES) + ["UNKNOWN"]
            hdr = f"{'actual ↓ / pred →':<20}" + "".join(f"{c:>12}" for c in cats)
            print(hdr)
            print("-" * len(hdr))
            for a in cats:
                row = f"{a:<20}" + "".join(f"{cc['matrix'][a][p]:>12.1f}" for p in cats)
                print(row)
            print(f"\nOff-diagonal share: {cc['off_diagonal_pct'] * 100:.2f}%")
            cd = cc["collapsed_damaging_vs_status"]
            tot_cd = sum(cd.values())
            print("\nCollapsed (damaging = PHYSICAL+SPECIAL, status = STATUS):")
            print(
                f"  actual DAMAGING → pred DAMAGING : {cd['actual_DAMAGING_pred_DAMAGING']:>10.1f} "
                f"({cd['actual_DAMAGING_pred_DAMAGING'] / max(tot_cd, 1) * 100:>5.2f}%)"
            )
            print(
                f"  actual DAMAGING → pred STATUS   : {cd['actual_DAMAGING_pred_STATUS']:>10.1f} "
                f"({cd['actual_DAMAGING_pred_STATUS'] / max(tot_cd, 1) * 100:>5.2f}%)"
            )
            print(
                f"  actual STATUS   → pred DAMAGING : {cd['actual_STATUS_pred_DAMAGING']:>10.1f} "
                f"({cd['actual_STATUS_pred_DAMAGING'] / max(tot_cd, 1) * 100:>5.2f}%)"
            )
            print(
                f"  actual STATUS   → pred STATUS   : {cd['actual_STATUS_pred_STATUS']:>10.1f} "
                f"({cd['actual_STATUS_pred_STATUS'] / max(tot_cd, 1) * 100:>5.2f}%)"
            )

        if (
            "per_move_name_favor" in report
            and report["per_move_name_favor"]["total_pairs"]
        ):
            pmn = report["per_move_name_favor"]
            print("\n### PER-MOVE-NAME FAVOR / DISFAVOR (per-mon) ###")
            print(
                "  Unit of analysis: one datapoint per active mon per turn "
                "(2 per turn when both active mons"
            )
            print(
                "  choose a move). We tally each chosen move name by the "
                "mon's MON:i:MOVE:j:move_id."
            )
            print(
                "  ambiguous = the two active mons have different moves at "
                "the chosen move slot, so we can't"
            )
            print(
                "  attribute the chosen-slot move to a single mon — weight "
                "splits uniformly across candidates."
            )
            print(
                f"\nDatapoints: {pmn['total_pairs']} "
                f"(ambiguous {pmn['ambiguous_pairs']} = {pmn['ambiguous_pct'] * 100:.1f}%; "
                f"min joint weight: {pmn['min_joint_count_threshold']})"
            )
            header = (
                f"{'Move':<22} {'Actual%':>10} {'Pred%':>10} {'Bias':>10} "
                f"{'Actual#':>10} {'Pred#':>10}"
            )
            print("\nTop OVER-predicted moves:")
            print(header)
            print("-" * len(header))
            for r in pmn["top_over_predicted"]:
                print(
                    f"{r['move']:<22} {r['actual_pct'] * 100:>9.2f}% "
                    f"{r['predicted_pct'] * 100:>9.2f}% {r['bias'] * 100:>+9.2f}% "
                    f"{r['actual_count']:>10.1f} {r['predicted_count']:>10.1f}"
                )
            print("\nTop UNDER-predicted moves:")
            print(header)
            print("-" * len(header))
            for r in pmn["top_under_predicted"]:
                print(
                    f"{r['move']:<22} {r['actual_pct'] * 100:>9.2f}% "
                    f"{r['predicted_pct'] * 100:>9.2f}% {r['bias'] * 100:>+9.2f}% "
                    f"{r['actual_count']:>10.1f} {r['predicted_count']:>10.1f}"
                )

        print("\n### INVALID PREDICTIONS ###")
        inv = report["invalid_predictions"]
        print(f"Invalid predictions: {inv['count']} ({inv['rate'] * 100:.2f}%)")

        print("\n" + "=" * 60)


def main(
    model_path: str,
    data_path: str,
    max_batches: Optional[int] = 100,
    device: Optional[str] = None,
    batch_size: int = 64,
    num_workers: int = 7,
    prefetch_factor: int = 8,
    files_per_worker: int = 3,
    torch_num_threads: Optional[int] = None,
):
    """
    Run diagnostics on a trained model.

    Args:
        model_path: Path to saved model (.pt file)
        data_path: Path to evaluation data directory
        max_batches: Maximum number of batches to analyze
        device: 'cpu', 'cuda', or None (= auto)
        batch_size, num_workers, prefetch_factor, files_per_worker:
            dataloader knobs. Defaults match the unconstrained-machine config.
            Set lower for "co-existing with a training run" safe runs.
        torch_num_threads: cap BLAS thread fan-out (useful when sharing CPU
            with training; None = leave at torch default).
    """
    if torch_num_threads is not None:
        torch.set_num_threads(torch_num_threads)

    print(f"Loading model from {model_path}")

    # Load model
    checkpoint = torch.load(model_path, map_location="cpu")
    config = checkpoint["config"]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize embedder
    embedder = Embedder(
        gen=9,
        feature_set=config["embedder_feature_set"],
        omniscient=False,
    )

    # Get feature indices
    feature_names = {name: i for i, name in enumerate(embedder.feature_names)}
    feature_idx: Dict[str, Any] = {
        "teampreview": feature_names["teampreview"],
        "force_switch_indices": [feature_names[f"MON:{j}:force_switch"] for j in range(6)],
        "mon_active_indices": [feature_names[f"MON:{j}:active"] for j in range(6)],
        # Per-active-slot positional features (added in the 2026-05-24
        # embedder change). When present, the diagnostic can resolve MDBO
        # slot 0/1 to a specific MON:i deterministically — no more
        # uniform-weight splitting. Skipped on old checkpoints that
        # predate this feature.
        "mon_active_slot_0_indices": (
            [feature_names[f"MON:{j}:active_slot_0"] for j in range(6)]
            if "MON:0:active_slot_0" in feature_names
            else None
        ),
        "mon_active_slot_1_indices": (
            [feature_names[f"MON:{j}:active_slot_1"] for j in range(6)]
            if "MON:0:active_slot_1" in feature_names
            else None
        ),
        # (mon_idx, move_slot_0idx) -> state index. Some teampreview slots
        # may not have move features when the team has <6 mons; we tolerate
        # missing keys at lookup time.
        "move_id_indices": {
            (i, j): feature_names[f"MON:{i}:MOVE:{j}:move_id"]
            for i in range(6)
            for j in range(4)
            if f"MON:{i}:MOVE:{j}:move_id" in feature_names
        },
        "move_offcat_indices": {
            (i, j, cat): feature_names[f"MON:{i}:MOVE:{j}:OFF_CAT:{cat}"]
            for i in range(6)
            for j in range(4)
            for cat in _CATEGORIES
            if f"MON:{i}:MOVE:{j}:OFF_CAT:{cat}" in feature_names
        },
    }

    # Initialize model (detect architecture from config)
    state_dict = checkpoint["model_state_dict"]
    # Strip _orig_mod. prefix from torch.compile'd checkpoints
    stripped_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("_orig_mod.", "") if k.startswith("_orig_mod.") else k
        stripped_state_dict[new_key] = v

    passthrough_keys = [
        "number_bank_hp_bins",
        "number_bank_stat_bins",
        "number_bank_power_bins",
        "number_bank_embedding_dim",
        "number_bank_damage_bins",
        "number_bank_damage_embed_dim",
        "number_bank_turn_bins",
        "number_bank_turn_embed_dim",
        "number_bank_rating_bins",
        "number_bank_rating_embed_dim",
        "ability_embed_dim",
        "item_embed_dim",
        "species_embed_dim",
        "move_embed_dim",
        "value_to_trunk_grad_scale",
    ]
    extra_kwargs = {k: config[k] for k in passthrough_keys if k in config}

    model = TransformerThreeHeadedModel(
        embedder=embedder,
        early_layers=config["early_layers"],
        late_layers=config["late_layers"],
        dropout=0.0,  # No dropout for evaluation
        grouped_encoder_hidden_dim=config["grouped_encoder_hidden_dim"],
        grouped_encoder_aggregated_dim=config["grouped_encoder_aggregated_dim"],
        pokemon_attention_heads=config["pokemon_attention_heads"],
        num_actions=MDBO.action_space(),
        num_teampreview_actions=MDBO.teampreview_space(),
        teampreview_head_layers=config["teampreview_head_layers"],
        teampreview_head_dropout=0.0,
        teampreview_attention_heads=config["teampreview_attention_heads"],
        turn_head_layers=config["turn_head_layers"],
        max_seq_len=config["max_seq_len"],
        num_value_bins=config.get("num_value_bins", 51),
        value_min=config.get("value_min", -1.0),
        value_max=config.get("value_max", 1.0),
        value_head_layers=config.get("value_head_layers"),
        transformer_layers=config.get("transformer_layers", 6),
        transformer_heads=config.get("transformer_heads", 16),
        transformer_ff_dim=config.get("transformer_ff_dim", 2048),
        transformer_dropout=0.0,
        use_decision_tokens=config.get("use_decision_tokens", True),
        use_causal_mask=config.get("use_causal_mask", True),
        **extra_kwargs,
    ).to(device)

    model.load_state_dict(stripped_state_dict)
    model.eval()

    print("Model loaded successfully")

    # Load data
    print(
        f"Loading data from {data_path} "
        f"(batch_size={batch_size}, num_workers={num_workers}, "
        f"prefetch_factor={prefetch_factor}, files_per_worker={files_per_worker})"
    )
    dataloader = OptimizedBattleDataLoader(
        data_path,
        embedder=embedder,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        files_per_worker=files_per_worker,
    )

    # Run diagnostics — pass embedding size so states are sliced to featureset size
    print("Running diagnostics...")
    diagnostics = ActionDiagnostics(model, device, state_input_dim=embedder.embedding_size)

    # Use tqdm for progress bar
    dataloader_iter = enumerate(dataloader)
    if max_batches is not None:
        dataloader_iter = enumerate(
            tqdm(dataloader, total=max_batches, desc="Processing batches")
        )
    else:
        dataloader_iter = enumerate(tqdm(dataloader, desc="Processing batches"))

    for i, batch in dataloader_iter:
        if max_batches is not None and i >= max_batches:
            break
        diagnostics.analyze_batch(batch, feature_idx)

    # Generate and print report
    report = diagnostics.generate_report()
    diagnostics.print_report(report)

    # Save report to JSON
    output_path = model_path.replace(".pt", "_action_diagnostics.json")
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Action prediction diagnostics — see SUPERVISED.md interpretation guide.",
    )
    parser.add_argument("model_path", help="Path to checkpoint .pt file")
    parser.add_argument("data_path", help="Directory of .pt.zst trajectory files")
    parser.add_argument(
        "--max-batches", type=int, default=None, help="Cap on batches (default: all)"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="auto = pick cuda if available; cpu = forced CPU (safe to run "
        "alongside a training job that has the GPU)",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=7)
    parser.add_argument("--prefetch-factor", type=int, default=8)
    parser.add_argument("--files-per-worker", type=int, default=3)
    parser.add_argument(
        "--torch-num-threads",
        type=int,
        default=None,
        help="Cap BLAS thread fan-out (omit to leave at torch default)",
    )
    parser.add_argument(
        "--safe",
        action="store_true",
        help="Shortcut for co-existing with a training job: forces device=cpu, "
        "batch_size=32, num_workers=2, prefetch_factor=1, files_per_worker=1, "
        "torch_num_threads=2.",
    )
    args = parser.parse_args()

    if args.safe:
        device_arg = "cpu"
        bs, nw, pf, fpw, tnt = 32, 2, 1, 1, 2
    else:
        device_arg = None if args.device == "auto" else args.device
        bs, nw, pf, fpw, tnt = (
            args.batch_size,
            args.num_workers,
            args.prefetch_factor,
            args.files_per_worker,
            args.torch_num_threads,
        )

    main(
        args.model_path,
        args.data_path,
        max_batches=args.max_batches,
        device=device_arg,
        batch_size=bs,
        num_workers=nw,
        prefetch_factor=pf,
        files_per_worker=fpw,
        torch_num_threads=tnt,
    )
