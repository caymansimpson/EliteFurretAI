"""Centralized GPU inference transport for Rust workers.

The worker-side policy API stays the same. When enabled, model forward passes
are shipped to a dedicated inference process over a Pipe so the GPU can batch
requests across workers.
"""

from __future__ import annotations

import time
import traceback
from collections import defaultdict
from multiprocessing import Process, get_context
from multiprocessing.connection import Connection, wait
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
import torch

from elitefurretai.etl import Embedder


def _cpu_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    cpu_state_dict: Dict[str, Any] = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            cpu_state_dict[key] = value.detach().cpu()
        else:
            cpu_state_dict[key] = value
    return cpu_state_dict


def _serialize_hidden(hidden_state: Any) -> Any:
    if hidden_state is None:
        return None
    if isinstance(hidden_state, tuple):
        return tuple(_serialize_hidden(value) for value in hidden_state)
    if isinstance(hidden_state, torch.Tensor):
        return hidden_state.detach().cpu().numpy()
    return np.asarray(hidden_state)


def _deserialize_hidden(hidden_state: Any) -> Any:
    if hidden_state is None:
        return None
    if isinstance(hidden_state, tuple):
        return tuple(_deserialize_hidden(value) for value in hidden_state)
    return torch.as_tensor(hidden_state)


def _move_hidden_to_device(hidden_state: Any, device: str) -> Any:
    if hidden_state is None:
        return None
    if isinstance(hidden_state, tuple):
        return tuple(_move_hidden_to_device(value, device) for value in hidden_state)
    return hidden_state.to(device)


def _state_to_tensor(state: Any, device: str) -> torch.Tensor:
    state_array = np.asarray(state, dtype=np.float32)
    if state_array.ndim == 1:
        state_array = state_array.reshape(1, 1, -1)
    elif state_array.ndim == 2:
        state_array = state_array[:, None, :]
    return torch.as_tensor(state_array, dtype=torch.float32, device=device)


def _is_lstm_hidden(hidden_state: Any) -> bool:
    return isinstance(hidden_state, tuple) and len(hidden_state) == 2


def _concat_lstm_hidden(hidden_states: Sequence[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    return (
        np.concatenate([hidden[0] for hidden in hidden_states], axis=1),
        np.concatenate([hidden[1] for hidden in hidden_states], axis=1),
    )


def _split_lstm_hidden(next_hidden: Tuple[np.ndarray, np.ndarray], batch_sizes: Sequence[int]) -> List[Tuple[np.ndarray, np.ndarray]]:
    split_hidden: List[Tuple[np.ndarray, np.ndarray]] = []
    start = 0
    for batch_size in batch_sizes:
        end = start + batch_size
        split_hidden.append(
            (
                next_hidden[0][:, start:end, :],
                next_hidden[1][:, start:end, :],
            )
        )
        start = end
    return split_hidden


def _transformer_hidden_length(hidden_state: Any) -> int:
    if hidden_state is None:
        return 0
    return int(np.asarray(hidden_state).shape[1])


def _concat_transformer_hidden(hidden_states: Sequence[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(hidden_state) for hidden_state in hidden_states], axis=0)


def _split_transformer_hidden(next_hidden: np.ndarray, batch_sizes: Sequence[int]) -> List[np.ndarray]:
    split_hidden: List[np.ndarray] = []
    start = 0
    for batch_size in batch_sizes:
        end = start + batch_size
        split_hidden.append(next_hidden[start:end])
        start = end
    return split_hidden


class CentralGpuInferenceClient:
    """Blocking client that mirrors the local RNaDAgent forward API."""

    def __init__(self, connection: Connection):
        self.connection = connection
        self._registered_models: set[str] = set()

    def register_model(
        self,
        model_key: str,
        model_config: Dict[str, Any],
        state_dict: Dict[str, Any],
        *,
        force: bool = False,
    ) -> None:
        if not force and model_key in self._registered_models:
            return

        self.connection.send(
            {
                "kind": "register_model",
                "model_key": model_key,
                "model_config": model_config,
                "state_dict": _cpu_state_dict(state_dict),
            }
        )
        response = self.connection.recv()
        if not isinstance(response, dict) or response.get("kind") != "register_ok":
            raise RuntimeError(f"Unexpected inference registration response: {response!r}")
        self._registered_models.add(model_key)

    def forward(
        self,
        model_key: str,
        state_tensor: torch.Tensor,
        hidden_state: Any = None,
        mask: Any = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        del mask

        self.connection.send(
            {
                "kind": "infer",
                "model_key": model_key,
                "state": state_tensor.detach().cpu().numpy(),
                "hidden": _serialize_hidden(hidden_state),
            }
        )
        response = self.connection.recv()
        if not isinstance(response, dict):
            raise RuntimeError(f"Unexpected inference response: {response!r}")
        if response.get("kind") == "error":
            raise RuntimeError(
                "Central inference server failed:\n"
                f"{response.get('error', 'unknown error')}\n"
                f"{response.get('traceback', '')}"
            )
        if response.get("kind") != "infer_result":
            raise RuntimeError(f"Unexpected inference response: {response!r}")

        turn_logits = torch.as_tensor(response["turn_logits"])
        tp_logits = torch.as_tensor(response["tp_logits"])
        values = torch.as_tensor(response["values"])
        win_dist_logits = torch.as_tensor(response["win_dist_logits"])
        next_hidden = _deserialize_hidden(response["next_hidden"])
        return turn_logits, tp_logits, values, win_dist_logits, next_hidden


def _build_model(model_config: Dict[str, Any], state_dict: Dict[str, Any], device: str) -> torch.nn.Module:
    from elitefurretai.etl import Embedder as ModelEmbedder
    from elitefurretai.rl.model_io import build_model_from_config

    embedder = ModelEmbedder(
        format=model_config.get("battle_format", "gen9vgc2023regc"),
        feature_set=model_config.get("embedder_feature_set", Embedder.FULL),
        omniscient=False,
    )
    model = build_model_from_config(model_config, embedder, device, state_dict)
    model.eval()
    return model


def run_central_gpu_inference_server(
    connections: Sequence[Connection],
    *,
    device: str,
    batch_timeout: float = 0.005,
    max_batch_size: int = 64,
) -> None:
    """Serve batched model inference over worker pipes."""

    torch.set_num_threads(1)
    resolved_device = device if device != "cuda" or torch.cuda.is_available() else "cpu"

    active_connections = list(connections)
    models: Dict[str, Any] = {}
    model_is_transformer: Dict[str, bool] = {}

    def remove_connection(connection: Connection) -> None:
        if connection in active_connections:
            active_connections.remove(connection)

    def send_error(connection: Connection, message: str) -> None:
        connection.send({"kind": "error", "error": message, "traceback": traceback.format_exc()})

    def process_infer_requests(requests: Sequence[Tuple[Connection, Dict[str, Any]]]) -> None:
        grouped: Dict[str, List[Tuple[Connection, Dict[str, Any]]]] = defaultdict(list)
        for connection, request in requests:
            grouped[str(request["model_key"])].append((connection, request))

        for model_key, group in grouped.items():
            model = models.get(model_key)
            if model is None:
                for connection, _ in group:
                    connection.send(
                        {
                            "kind": "error",
                            "error": f"Model {model_key!r} has not been registered.",
                            "traceback": "",
                        }
                    )
                continue

            if model_is_transformer.get(model_key, False):
                grouped_by_context: Dict[int, List[Tuple[Connection, Dict[str, Any]]]] = defaultdict(list)
                for connection, request in group:
                    grouped_by_context[_transformer_hidden_length(request.get("hidden"))].append((connection, request))

                for context_len, transformer_group in grouped_by_context.items():
                    transformer_states: List[np.ndarray] = []
                    transformer_hidden: List[np.ndarray] = []
                    transformer_batch_sizes: List[int] = []
                    fallback = False
                    for _, request in transformer_group:
                        state_np = np.asarray(request["state"], dtype=np.float32)
                        if state_np.ndim == 3 and state_np.shape[1] == 1:
                            state_np = state_np[:, 0, :]
                        elif state_np.ndim == 1:
                            state_np = state_np.reshape(1, -1)
                        transformer_states.append(state_np)
                        transformer_batch_sizes.append(state_np.shape[0])

                        hidden_np = request.get("hidden")
                        if context_len > 0:
                            if hidden_np is None or _is_lstm_hidden(hidden_np):
                                fallback = True
                                break
                            transformer_hidden.append(np.asarray(hidden_np, dtype=np.float32))

                    if fallback or not transformer_states:
                        for connection, request in transformer_group:
                            try:
                                state_tensor = _state_to_tensor(request["state"], resolved_device)
                                hidden_tensor = _move_hidden_to_device(_deserialize_hidden(request.get("hidden")), resolved_device)
                                with torch.inference_mode():
                                    turn_logits, tp_logits, values, win_dist_logits, next_hidden = model.forward_with_hidden(
                                        state_tensor,
                                        hidden_tensor,
                                    )
                                connection.send(
                                    {
                                        "kind": "infer_result",
                                        "turn_logits": turn_logits.detach().cpu().numpy(),
                                        "tp_logits": tp_logits.detach().cpu().numpy(),
                                        "values": values.detach().cpu().numpy(),
                                        "win_dist_logits": win_dist_logits.detach().cpu().numpy(),
                                        "next_hidden": _serialize_hidden(next_hidden),
                                    }
                                )
                            except Exception:
                                send_error(connection, f"Inference failed for model {model_key!r}")
                        continue

                    combined_states = np.concatenate(transformer_states, axis=0)
                    state_tensor = torch.as_tensor(combined_states, dtype=torch.float32, device=resolved_device).unsqueeze(1)
                    hidden_tensor = None
                    if context_len > 0:
                        hidden_tensor = torch.as_tensor(
                            _concat_transformer_hidden(transformer_hidden),
                            dtype=torch.float32,
                            device=resolved_device,
                        )

                    try:
                        with torch.inference_mode():
                            turn_logits, tp_logits, values, win_dist_logits, next_hidden = model.forward_with_hidden(
                                state_tensor,
                                hidden_tensor,
                            )

                        next_hidden_np = next_hidden.detach().cpu().numpy() if next_hidden is not None else None
                        if next_hidden_np is not None:
                            transformer_split_hidden: List[Optional[np.ndarray]] = list(
                                _split_transformer_hidden(next_hidden_np, transformer_batch_sizes)
                            )
                        else:
                            transformer_split_hidden = [None for _ in transformer_batch_sizes]

                        start = 0
                        for (connection, _), batch_size, next_batch_hidden in zip(
                            transformer_group,
                            transformer_batch_sizes,
                            transformer_split_hidden,
                        ):
                            end = start + batch_size
                            connection.send(
                                {
                                    "kind": "infer_result",
                                    "turn_logits": turn_logits[start:end].detach().cpu().numpy(),
                                    "tp_logits": tp_logits[start:end].detach().cpu().numpy(),
                                    "values": values[start:end].detach().cpu().numpy(),
                                    "win_dist_logits": win_dist_logits[start:end].detach().cpu().numpy(),
                                    "next_hidden": next_batch_hidden,
                                }
                            )
                            start = end
                    except Exception:
                        for connection, _ in transformer_group:
                            send_error(connection, f"Inference failed for model {model_key!r}")
                continue

            lstm_states: List[np.ndarray] = []
            lstm_hidden: List[Tuple[np.ndarray, np.ndarray]] = []
            lstm_batch_sizes: List[int] = []
            fallback = False
            for _, request in group:
                state_np = np.asarray(request["state"], dtype=np.float32)
                if state_np.ndim == 3 and state_np.shape[1] == 1:
                    state_np = state_np[:, 0, :]
                elif state_np.ndim == 1:
                    state_np = state_np.reshape(1, -1)
                hidden_np = request.get("hidden")
                if not _is_lstm_hidden(hidden_np):
                    fallback = True
                    break
                lstm_states.append(state_np)
                hidden_tuple = cast(Tuple[np.ndarray, np.ndarray], hidden_np)
                lstm_hidden.append((np.asarray(hidden_tuple[0]), np.asarray(hidden_tuple[1])))
                lstm_batch_sizes.append(state_np.shape[0])

            if fallback or not lstm_states:
                for connection, request in group:
                    try:
                        state_tensor = _state_to_tensor(request["state"], resolved_device)
                        hidden_tensor = _move_hidden_to_device(_deserialize_hidden(request.get("hidden")), resolved_device)
                        with torch.inference_mode():
                            turn_logits, tp_logits, values, win_dist_logits, next_hidden = model.forward_with_hidden(
                                state_tensor,
                                hidden_tensor,
                            )
                        connection.send(
                            {
                                "kind": "infer_result",
                                "turn_logits": turn_logits.detach().cpu().numpy(),
                                "tp_logits": tp_logits.detach().cpu().numpy(),
                                "values": values.detach().cpu().numpy(),
                                "win_dist_logits": win_dist_logits.detach().cpu().numpy(),
                                "next_hidden": _serialize_hidden(next_hidden),
                            }
                        )
                    except Exception:
                        send_error(connection, f"Inference failed for model {model_key!r}")
                continue

            combined_states = np.concatenate(lstm_states, axis=0)
            hidden_tuple = _concat_lstm_hidden(lstm_hidden)
            state_tensor = torch.as_tensor(combined_states, dtype=torch.float32, device=resolved_device).unsqueeze(1)
            hidden_tensor = (
                torch.as_tensor(hidden_tuple[0], device=resolved_device),
                torch.as_tensor(hidden_tuple[1], device=resolved_device),
            )

            try:
                with torch.inference_mode():
                    turn_logits, tp_logits, values, win_dist_logits, next_hidden = model.forward_with_hidden(
                        state_tensor,
                        hidden_tensor,
                    )

                lstm_split_hidden = _split_lstm_hidden(
                    (
                        next_hidden[0].detach().cpu().numpy(),
                        next_hidden[1].detach().cpu().numpy(),
                    ),
                    lstm_batch_sizes,
                )

                start = 0
                for (connection, _), batch_size, next_lstm_hidden in zip(group, lstm_batch_sizes, lstm_split_hidden):
                    end = start + batch_size
                    connection.send(
                        {
                            "kind": "infer_result",
                            "turn_logits": turn_logits[start:end].detach().cpu().numpy(),
                            "tp_logits": tp_logits[start:end].detach().cpu().numpy(),
                            "values": values[start:end].detach().cpu().numpy(),
                            "win_dist_logits": win_dist_logits[start:end].detach().cpu().numpy(),
                            "next_hidden": next_lstm_hidden,
                        }
                    )
                    start = end
            except Exception:
                for connection, _ in group:
                    send_error(connection, f"Inference failed for model {model_key!r}")

    while active_connections:
        try:
            ready_connections = cast(List[Connection], wait(active_connections, timeout=0.05))
        except (OSError, ValueError):
            break

        if not ready_connections:
            continue

        batch_requests: List[Tuple[Connection, Dict[str, Any]]] = []
        deadline = time.perf_counter() + batch_timeout

        while True:
            for connection in list(ready_connections):
                while connection.poll():
                    try:
                        message = connection.recv()
                    except EOFError:
                        remove_connection(connection)
                        break

                    if not isinstance(message, dict):
                        continue

                    kind = message.get("kind")
                    if kind == "stop":
                        remove_connection(connection)
                        continue
                    if kind == "register_model":
                        model_key = str(message["model_key"])
                        try:
                            model_config = dict(message["model_config"])
                            state_dict = message["state_dict"]
                            models[model_key] = _build_model(model_config, state_dict, resolved_device)
                            model_is_transformer[model_key] = bool(model_config.get("use_transformer", False))
                            connection.send({"kind": "register_ok", "model_key": model_key})
                        except Exception:
                            connection.send(
                                {
                                    "kind": "error",
                                    "error": f"Failed to register model {model_key!r}",
                                    "traceback": traceback.format_exc(),
                                }
                            )
                        continue
                    if kind == "infer":
                        batch_requests.append((connection, message))

            if len(batch_requests) >= max_batch_size:
                break

            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break

            additional_ready = cast(List[Connection], wait(active_connections, timeout=remaining))
            if not additional_ready:
                break
            ready_connections = additional_ready

        if batch_requests:
            process_infer_requests(batch_requests)


def start_central_gpu_inference_server(
    connections: Sequence[Connection],
    *,
    device: str,
    batch_timeout: float = 0.005,
    max_batch_size: int = 64,
    name: str = "CentralGpuInferenceServer",
) -> Process:
    spawn_context = get_context("spawn")
    process = spawn_context.Process(
        target=run_central_gpu_inference_server,
        args=(list(connections),),
        kwargs={
            "device": device,
            "batch_timeout": batch_timeout,
            "max_batch_size": max_batch_size,
        },
        daemon=True,
        name=name,
    )
    process.start()
    return cast(Process, process)


__all__ = [
    "CentralGpuInferenceClient",
    "run_central_gpu_inference_server",
    "start_central_gpu_inference_server",
]
