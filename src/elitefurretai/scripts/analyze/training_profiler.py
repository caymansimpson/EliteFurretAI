import sys
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import time

from elitefurretai.model_utils import MDBO, Embedder, PreprocessedTrajectoryDataset, flatten_and_filter, topk_cross_entropy_loss


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_features, out_features, dropout=0.3):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
        self.ln = torch.nn.LayerNorm(out_features)
        self.dropout = torch.nn.Dropout(dropout)
        self.relu = torch.nn.ReLU()
        self.shortcut = torch.nn.Sequential()
        if in_features != out_features:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Linear(in_features, out_features),
                torch.nn.LayerNorm(out_features),
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.linear(x)
        x = self.ln(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.relu(x + residual)  # Add ReLU after addition


class TwoHeadedHybridModel(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_layers=[1024, 512, 256],
        num_heads=4,
        num_lstm_layers=1,
        num_actions=MDBO.action_space(),
        max_seq_len=40,
        dropout=0.1,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_layers[-1]
        self.num_actions = num_actions

        # Feedforward stack with residual blocks
        layers = []
        prev_size = input_size
        for h in hidden_layers:
            layers.append(ResidualBlock(prev_size, h, dropout=dropout))
            prev_size = h
        self.ff_stack = torch.nn.Sequential(*layers)

        # Positional encoding (learned) for the final hidden size
        self.pos_embedding = torch.nn.Embedding(max_seq_len, self.hidden_size)

        # Bidirectional LSTM
        self.lstm = torch.nn.LSTM(
            self.hidden_size,
            self.hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm_proj = torch.nn.Linear(self.hidden_size * 2, self.hidden_size)

        # Multihead Self-Attention block
        self.self_attn = torch.nn.MultiheadAttention(
            self.hidden_size, num_heads, batch_first=True
        )

        # Normalize outputs
        self.norm = torch.nn.LayerNorm(self.hidden_size)

        # Output heads
        self.action_head = torch.nn.Linear(self.hidden_size, num_actions)
        self.win_head = torch.nn.Linear(self.hidden_size, 1)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # Feedforward stack with residuals
        x = self.ff_stack(x)

        # Add positional encoding
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = (
            x + self.pos_embedding(positions) * mask.unsqueeze(-1)
            if mask is not None
            else x + self.pos_embedding(positions)
        )

        if mask is None:
            mask = torch.ones(batch_size, seq_len, device=x.device)

        # LSTM (packed)
        lengths = mask.sum(dim=1).long().cpu()
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True, total_length=seq_len
        )
        lstm_out = self.lstm_proj(lstm_out)

        # Multihead Self-Attention
        attn_mask = ~mask.bool()
        attn_out, _ = self.self_attn(
            lstm_out, lstm_out, lstm_out, key_padding_mask=attn_mask
        )
        out = self.norm(attn_out + lstm_out)

        # *** REMOVE POOLING ***
        # Output heads: now per-step
        action_logits = self.action_head(out)  # (batch, seq_len, num_actions)
        win_logits = self.win_head(out).squeeze(-1)  # (batch, seq_len)

        return action_logits, win_logits

    def predict(self, x, mask=None):
        with torch.no_grad():
            action_logits, win_logits = self.forward(x, mask)
            action_probs = torch.softmax(action_logits, dim=-1)
            win_prob = torch.sigmoid(win_logits)
        return action_probs, win_prob


def profile_training_step(model, dataloader, optimizer):
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True
    ) as prof:
        model.train()
        scaler = torch.amp.GradScaler("cuda")

        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 10:  # Profile first 10 batches
                break

            # Get data from dictionary
            with record_function("data_loading"):
                states = batch["states"].to("cuda")
                actions = batch["actions"].to("cuda")
                action_masks = batch["action_masks"].to("cuda")
                wins = batch["wins"].to("cuda")
                masks = batch["masks"].to("cuda")

            # Mixed precision context
            with torch.amp.autocast("cuda"):

                # Forward pass
                with record_function("forward pass"):
                    action_logits, win_logits = model(states, masks)

                # Processing data and logits
                with record_function("data processing"):
                    masked_action_logits = action_logits.masked_fill(
                        ~action_masks.bool(), float("-inf")
                    )

                    # Use helper for flattening and filtering
                    flat_data = flatten_and_filter(
                        states=states,
                        action_logits=masked_action_logits,
                        actions=actions,
                        win_logits=win_logits,
                        wins=wins,
                        action_masks=action_masks,
                        masks=masks,
                    )
                    if flat_data is None:
                        continue

                    valid_states, valid_action_logits, valid_actions, valid_win_logits, valid_wins = (
                        flat_data
                    )

                # Losses
                with record_function("loss calculation"):
                    action_loss = topk_cross_entropy_loss(
                        valid_action_logits, valid_actions, weights=None, k=3
                    )
                    win_loss = torch.nn.functional.mse_loss(valid_win_logits, valid_wins.float())
                    loss = action_loss + win_loss

            with record_function("backwards pass"):
                optimizer.zero_grad()

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

    # Print summary
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    print("\n" + "=" * 80 + "\n")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

    # Export for tensorboard visualization
    prof.export_chrome_trace("trace.json")
    print("\nTrace saved to trace.json - view at chrome://tracing/")


# Profiles dataloader performance
def profile_dataloader(dataloader, num_batches=50):
    """
    Profile your DataLoader to find bottlenecks
    """
    print(f"Profiling DataLoader with {dataloader.num_workers} workers...")
    print(f"Batch size: {dataloader.batch_size}")
    print(f"Dataset size: {len(dataloader.dataset)}")

    # Warmup
    iterator = iter(dataloader)
    for _ in range(5):
        try:
            next(iterator)
        except StopIteration:
            break

    # Actual timing
    times = []
    data_transfer_times = []

    iterator = iter(dataloader)
    for i in range(num_batches):
        try:
            start_time = time.time()
            batch = next(iterator)
            load_time = time.time() - start_time

            # Load different data structures
            states = batch["states"]
            actions = batch["actions"]
            action_masks = batch["action_masks"]
            wins = batch["wins"]
            masks = batch["masks"]

            start_transfer = time.time()

            # Transfer to GPU
            states = states.cuda(non_blocking=True)
            actions = actions.cuda(non_blocking=True)
            action_masks = action_masks.cuda(non_blocking=True)
            wins = wins.cuda(non_blocking=True)
            masks = masks.cuda(non_blocking=True)
            torch.cuda.synchronize()  # Wait for transfer to complete

            # Record time
            transfer_time = time.time() - start_transfer
            data_transfer_times.append(transfer_time)
            times.append(load_time)

            if i % 10 == 0:
                print(f"Batch {i}: Load={load_time:.3f}s, Transfer={transfer_time:.3f}s")

        except StopIteration:
            print(f"DataLoader exhausted at batch {i}")
            break

    # Analysis
    avg_load_time = sum(times) / len(times)
    avg_transfer_time = sum(data_transfer_times) / len(data_transfer_times)

    print(f"\n{'=' * 50}")
    print("DATALOADER ANALYSIS")
    print(f"{'=' * 50}")
    print(f"Average batch load time: {avg_load_time:.3f}s")
    print(f"Average GPU transfer time: {avg_transfer_time:.3f}s")
    print(f"Total data loading overhead: {avg_load_time + avg_transfer_time:.3f}s per batch")
    print(f"Estimated throughput: {1 / (avg_load_time + avg_transfer_time):.1f} batches/sec")

    if avg_load_time > 0.1:
        print("🚨 SLOW DATA LOADING - Try:")
        print("  - Increase num_workers")
        print("  - Use pin_memory=True")
        print("  - Optimize your Dataset.__getitem__ method")

    if avg_transfer_time > 0.05:
        print("⚠️  SLOW GPU TRANSFER - Try:")
        print("  - Use pin_memory=True")
        print("  - Use non_blocking=True in .cuda() calls")

    return {
        'avg_load_time': avg_load_time,
        'avg_transfer_time': avg_transfer_time,
        'total_overhead': avg_load_time + avg_transfer_time
    }


# Test different worker configurations
def test_worker_configurations(dataset, batch_size, device='cuda'):
    """
    Test different num_workers to find optimal setting
    """
    worker_counts = [0, 1, 2, 4, 8, 12, 16]
    results = {}

    for num_workers in worker_counts:
        print(f"\nTesting {num_workers} workers...")

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0
        )

        try:
            result = profile_dataloader(dataloader, num_batches=20)
            results[num_workers] = result['total_overhead']
        except Exception as e:
            print(f"Error with {num_workers} workers: {e}")
            results[num_workers] = float('inf')

    # Find optimal
    best_workers = min(results.keys(), key=lambda k: results[k])

    print(f"\n{'=' * 60}")
    print("OPTIMAL CONFIGURATION")
    print(f"{'=' * 60}")
    print(f"Best num_workers: {best_workers}")
    print(f"Best time per batch: {results[best_workers]:.3f}s")

    for workers, time_per_batch in sorted(results.items()):
        if time_per_batch != float('inf'):
            print(f"{workers:2d} workers: {time_per_batch:.3f}s per batch")


def main(data_path):
    print("Starting!")
    embedder = Embedder(format="gen9vgc2023regulationc", feature_set=Embedder.FULL, omniscient=False)
    print(f"Embedder initialized. Size: {embedder.embedding_size}! Loading datasets...")
    start = time.time()
    train_dataset = PreprocessedTrajectoryDataset(data_path, embedder=embedder)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    print(f"Finished loading dataset in {time.time() - start:.2f}s! Size: {len(train_dataset)} trajectories. Now profiling...")

    # Profile dataloader
    profile_dataloader(train_loader)

    # Initialize model
    print("Now onto training profilingInitializing model...")
    model = TwoHeadedHybridModel(
        input_size=embedder.embedding_size,
        hidden_layers=[1024, 512, 256],
        num_heads=4,
        num_lstm_layers=2,
        num_actions=MDBO.action_space(),
    ).to("cuda")
    print(f"Finished loading data and model! for a total of {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")

    # Initialize optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.002,
        weight_decay=0.0001,
        betas=(0.9, 0.999),
    )

    print("Initialized model! Starting training profiling...")
    profile_training_step(model, train_loader, optimizer)


if __name__ == "__main__":
    main(sys.argv[1])
