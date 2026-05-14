# -*- coding: utf-8 -*-
"""Real model `BatchHandler` implementations for `InferenceService`.

`RealModelBatchHandler` is a callable that takes a list of `InferenceRequest`
and returns a list of `InferenceResponse`, doing the same work that
`BatchInferencePlayer._gpu_inference_sync` + the `_run_batch` per-request
loop do today. M2 ships the real-model variant; M1 has an echo handler
for plumbing tests.

D3-alt: hidden state lives here
-------------------------------
After the M4e measurement showed shipping per-turn hidden tensors over
mp.Queue dominated wire cost, the handler now owns a `hidden_states`
dict keyed by (worker_id, battle_tag). The wire payload only carries
the lightweight battle_tag; the handler looks up the prior context,
runs the forward, and stores the updated context back in its own dict.
The wire response no longer carries `next_hidden`.

Memory hygiene: callers MUST send an `EvictRequest` when a battle ends
or the dict grows monotonically over the run. The handler exposes
`evict(worker_id, battle_tag)` for the service to call.

What this handler owns
----------------------
- The model (an `RNaDAgent` wrapping a transformer or LSTM model).
- The device (typically "cpu" today; "cuda" once we move worker inference
  to GPU).
- The `hidden_states` dict (D3-alt).
- Per-request sampling math: temperature scaling, mask + renormalize,
  top-p nucleus filter, multinomial draw OR argmax, masked T=1 log-prob.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, cast

import numpy as np
import torch

from elitefurretai.rl.inference_ipc import InferenceRequest, InferenceResponse
from elitefurretai.rl.players import RNaDAgent
from elitefurretai.supervised.model_archs import TransformerThreeHeadedModel

logger = logging.getLogger(__name__)


class RealModelBatchHandler:
    """Callable that runs one batched forward + sampling pass.

    Use as the `batch_handler` argument to `InferenceService`.
    """

    def __init__(
        self,
        agent: RNaDAgent,
        device: str = "cpu",
        probabilistic: bool = True,
    ):
        self.agent = agent
        self.device = device
        self.probabilistic = probabilistic
        # Determine architecture once; affects hidden-state batching.
        self._is_transformer = isinstance(
            getattr(agent, "model", agent), TransformerThreeHeadedModel
        )
        # D3-alt: hidden state lives on the trainer side, keyed by
        # (worker_id, battle_tag). Service calls __call__ on a single
        # thread, so no lock needed for normal request processing.
        # `evict` is called from the same thread (service drains evict
        # requests off the same queue), so still no lock needed.
        # Key: (worker_id, player_id, battle_tag). player_id is required
        # because both sides of a self-play battle share the same
        # battle_tag — keying without it would collide and double-grow
        # the hidden tensor.
        self.hidden_states: Dict[Tuple[int, str, str], torch.Tensor] = {}

    def __call__(self, batch: List[InferenceRequest]) -> List[InferenceResponse]:
        if not batch:
            return []

        states_np = np.stack([r.state for r in batch], axis=0)
        states = torch.from_numpy(states_np).to(self.device).unsqueeze(1).float()

        # Look up each request's prior hidden from our local dict.
        prior_hiddens: List[Optional[torch.Tensor]] = [
            self.hidden_states.get((r.worker_id, r.player_id, r.battle_tag))
            for r in batch
        ]
        # Diagnostic: catch overgrown contexts before they crash inside
        # the model. Useful while D3-alt eviction policy is still being
        # validated.
        for ph, req in zip(prior_hiddens, batch):
            if ph is not None and ph.shape[1] > 35:
                logger.warning(
                    "Long context: worker=%d player=%s battle=%s shape=%s",
                    req.worker_id,
                    req.player_id,
                    req.battle_tag,
                    tuple(ph.shape),
                )

        if self._is_transformer:
            hidden_batch, hidden_mask = self._pad_transformer_context(prior_hiddens)
            with torch.no_grad():
                turn_logits, tp_logits, values, _, next_hidden = self.agent(
                    states, hidden_batch, mask=None, hidden_mask=hidden_mask
                )
        else:
            h_batch, c_batch = self._pad_lstm_state(prior_hiddens)
            with torch.no_grad():
                turn_logits, tp_logits, values, _, next_hidden = self.agent(
                    states, (h_batch, c_batch), mask=None
                )

        turn_logits_cpu = turn_logits.cpu()
        tp_logits_cpu = tp_logits.cpu()
        values_cpu = values.cpu().numpy()

        responses: List[InferenceResponse] = []
        for i, req in enumerate(batch):
            if req.is_teampreview:
                logits = tp_logits_cpu[i, 0]
            else:
                logits = turn_logits_cpu[i, 0]

            mask_np = req.mask
            action_idx, log_prob = self._sample_action(
                logits=logits,
                mask=mask_np,
                temperature=req.temperature,
                top_p=req.top_p,
                is_teampreview=req.is_teampreview,
            )

            # Update the trainer-side hidden_states dict in place. The
            # response no longer carries this — workers don't need it.
            self.hidden_states[
                (req.worker_id, req.player_id, req.battle_tag)
            ] = self._slice_next_hidden(next_hidden, i, prior_hiddens[i])
            responses.append(
                InferenceResponse(
                    request_id=req.request_id,
                    action_idx=action_idx,
                    log_prob=log_prob,
                    value=float(values_cpu[i, 0]),
                )
            )
        return responses

    def evict(self, worker_id: int, player_id: str, battle_tag: str) -> None:
        """Free the hidden state for one side of a finished battle.

        Called by the service when an EvictRequest arrives. No-op if
        missing. Each side of a self-play battle evicts independently."""
        self.hidden_states.pop((worker_id, player_id, battle_tag), None)

    # ── hidden-state batching ─────────────────────────────────────────

    def _pad_transformer_context(
        self, prior_hiddens: List[Optional[torch.Tensor]]
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Pad each request's growing context to the max in this batch.

        Mirrors the legacy `_run_batch` transformer padding logic, but
        reads from the trainer-side `hidden_states` dict instead of
        per-request `hidden` fields.
        """
        context_lengths: List[int] = []
        hidden_size: Optional[int] = None
        for ctx in prior_hiddens:
            if ctx is None:
                context_lengths.append(0)
                continue
            if ctx.ndim != 3 or ctx.shape[0] != 1:
                raise ValueError(
                    f"Expected transformer context shape (1, T, H), got {tuple(ctx.shape)}"
                )
            context_lengths.append(int(ctx.shape[1]))
            if hidden_size is None:
                hidden_size = int(ctx.shape[2])

        max_context_len = max(context_lengths, default=0)
        if max_context_len == 0:
            return None, None
        assert hidden_size is not None

        B = len(prior_hiddens)
        hidden = torch.zeros(B, max_context_len, hidden_size, dtype=torch.float32)
        hmask = torch.zeros(B, max_context_len, dtype=torch.bool)
        for i, ctx in enumerate(prior_hiddens):
            length = context_lengths[i]
            if ctx is None or length == 0:
                continue
            hidden[i, :length, :] = ctx[0, :length, :]
            hmask[i, :length] = True
        return hidden.to(self.device), hmask.to(self.device)

    def _pad_lstm_state(
        self, prior_hiddens: List[Optional[torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """LSTM hidden states: stack (h, c) along batch dim. The dict
        stores them packed as (2, num_layers*dirs, H) per the same
        convention used by `_slice_next_hidden`.
        """
        h_list: List[torch.Tensor] = []
        c_list: List[torch.Tensor] = []
        for ctx in prior_hiddens:
            if ctx is None:
                init = self.agent.get_initial_state(1, self.device)
                assert init is not None
                h, c = init
            else:
                if ctx.ndim != 3 or ctx.shape[0] != 2:
                    raise ValueError(
                        f"Expected LSTM hidden shape (2, L, H), got {tuple(ctx.shape)}"
                    )
                h = ctx[0:1].to(self.device).transpose(0, 1)
                c = ctx[1:2].to(self.device).transpose(0, 1)
            h_list.append(h)
            c_list.append(c)
        h_batch = torch.cat(h_list, dim=1)
        c_batch = torch.cat(c_list, dim=1)
        return h_batch, c_batch

    def _slice_next_hidden(
        self,
        next_hidden_batch: object,
        index: int,
        prev_hidden: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Return the new hidden state for request[index] in a form the
        handler stores back into `hidden_states[(worker_id, battle_tag)]`.

        Transformer notes (correctness fix vs legacy):
            The model's forward_with_hidden concatenates `hidden_state`
            (padded to max_T) + `encoded` (length 1), so the new
            encoded state sits at position max_T regardless of this
            request's actual prior length L_i. The legacy
            BatchInferencePlayer._run_batch sliced [:L_i+1], which
            included a padding-derived position at L_i instead of the
            real new state at position max_T. With mixed-length batches
            this corrupted the next-turn hidden state and propagated
            indefinitely. We fix by explicitly concatenating the prior
            slice with the new-state slice at max_T.
        """
        if self._is_transformer:
            ctx_batch = cast(torch.Tensor, next_hidden_batch).cpu()
            prev_len = 0 if prev_hidden is None else prev_hidden.shape[1]
            max_T = ctx_batch.shape[1] - 1  # output length is max_T + 1
            prior = ctx_batch[index : index + 1, :prev_len, :]
            new_state = ctx_batch[index : index + 1, max_T : max_T + 1, :]
            return torch.cat([prior, new_state], dim=1).clone()
        # LSTM: pack (h, c) back into the (2, layers*dirs, H) wire shape.
        h_batch, c_batch = cast(tuple, next_hidden_batch)
        h_i = h_batch[:, index : index + 1, :].cpu()  # (L, 1, H)
        c_i = c_batch[:, index : index + 1, :].cpu()
        # Reshape to (2, L, H) for IPC.
        return torch.stack([h_i.squeeze(1), c_i.squeeze(1)], dim=0)

    # ── per-request sampling ──────────────────────────────────────────

    def _sample_action(
        self,
        logits: torch.Tensor,
        mask: Optional[np.ndarray],
        temperature: float,
        top_p: float,
        is_teampreview: bool,
    ) -> tuple[int, float]:
        """Mirror BatchInferencePlayer's per-request sampling math:
        temperature softmax → mask + renormalize → top-p filter →
        multinomial; PPO old_log_prob from the masked T=1 distribution.
        """
        temp = max(temperature, 1e-6)
        probs = torch.softmax(logits / temp, dim=-1).numpy()
        unscaled_log_probs = torch.log_softmax(logits, dim=-1).numpy()

        if not is_teampreview and mask is not None:
            probs = probs * mask
            total = probs.sum()
            if total == 0.0:
                probs = mask / mask.sum()
            else:
                probs = probs / total
            if top_p < 1.0:
                sorted_idx = np.argsort(-probs)
                cum = np.cumsum(probs[sorted_idx])
                cutoff = int(np.searchsorted(cum, top_p)) + 1
                keep = sorted_idx[:cutoff]
                filtered = np.zeros_like(probs)
                filtered[keep] = probs[keep]
                probs = filtered / filtered.sum()

        n = probs.shape[0]
        if self.probabilistic:
            action = int(np.random.choice(n, p=probs))
        else:
            action = int(np.argmax(probs))

        if mask is not None and not is_teampreview:
            valid_mask = mask.astype(bool)
            log_valid_mass = float(np.log(np.exp(unscaled_log_probs[valid_mask]).sum()))
            log_prob = float(unscaled_log_probs[action] - log_valid_mass)
        else:
            log_prob = float(unscaled_log_probs[action])
        return action, log_prob
