# -*- coding: utf-8 -*-
"""This script trains a supervised model for move prediction; if you have collected replays, you can build a model
to try to play like humans.
"""
import os.path
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

import wandb
from elitefurretai.model_utils import Embedder, ModelBattleOrder


class HybridModel(nn.Module):
    def __init__(
        self, input_size, hidden_size, num_heads, num_layers, num_actions, max_seq_len=17
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size

        # Positional embeddings
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_size)

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Multihead attention
        self.attention = nn.MultiheadAttention(hidden_size, num_heads)

        # Transformer layer
        self.transformer = nn.Transformer(
            d_model=hidden_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
        )

        # Output heads
        self.action_head = nn.Linear(hidden_size, num_actions)
        self.win_head = nn.Linear(hidden_size, 1)

        # LayerNorm
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x, mask=None):
        """
        Args:
            x: Padded input tensor of shape (batch_size, seq_len, input_size)
            mask: Binary mask of shape (batch_size, seq_len) where 1 = valid, 0 = padding
        """
        batch_size, seq_len, _ = x.shape

        # Generate mask if not provided (for inference)
        if mask is None:
            mask = torch.ones(batch_size, seq_len).to(x.device)

        # Positional encoding
        positions = torch.arange(seq_len).unsqueeze(0).to(x.device)
        x = x + self.pos_embedding(positions)

        # LSTM with packed sequences
        lengths = mask.sum(dim=1).cpu()  # Convert to CPU for packing
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)

        # Transformer with masking
        transformer_out = self.transformer(
            lstm_out.transpose(0, 1),  # Transformer expects (seq_len, batch, dim)
            src_key_padding_mask=~mask.bool(),
        ).transpose(0, 1)

        # Multihead attention
        attn_out, _ = self.attention(
            transformer_out.transpose(0, 1),
            transformer_out.transpose(0, 1),
            transformer_out.transpose(0, 1),
            key_padding_mask=~mask.bool(),
        )
        attn_out = attn_out.transpose(0, 1)

        # Final output with LayerNorm
        combined = self.norm(attn_out + transformer_out)

        # Pooling: Weighted average using mask
        mask_unsqueezed = mask.unsqueeze(-1)  # (batch, seq_len, 1)
        pooled = (combined * mask_unsqueezed).sum(dim=1) / mask_unsqueezed.sum(dim=1)

        # Output heads
        action_logits = self.action_head(pooled)
        win_prob = torch.sigmoid(self.win_head(pooled))

        return action_logits, win_prob

    def predict(self, x):
        """
        Predict actions and win probability for a single battle state.
        Args:
            battle_state: Tensor of shape (1, seq_len, input_size)
        Returns:
            action_probs: Probability distribution over actions
            win_prob: Predicted win probability (0-1)
        """
        with torch.no_grad():
            action_logits, win_prob = self(x)
            action_probs = torch.softmax(action_logits, dim=-1)
        return action_probs.squeeze(), win_prob.squeeze()


def train_model(model, dataloader, criterion_action, criterion_win, optimizer):
    model.train()
    total_loss = torch.tensor(0.0, device=model.device)

    for batch in dataloader:
        optimizer.zero_grad()

        # Unpack batch
        padded_sequences, masks, action_targets, win_targets = batch

        # Forward pass
        action_logits, win_probs = model(padded_sequences, masks)

        # Calculate losses
        loss_action = criterion_action(action_logits, action_targets)
        loss_win = criterion_win(win_probs.squeeze(), win_targets.float())

        # Combine losses with mask consideration
        mask_sum = masks.sum()
        loss_action = (
            loss_action * masks[:, -1]
        ).sum() / mask_sum  # Use last timestep mask for actions
        loss_win = (loss_win * masks[:, -1]).sum() / mask_sum

        # TODO: need to correct for loss_action using different loss function; consider temperature scaling
        total_loss = 0.7 * loss_action + 0.3 * loss_win

        # Backward pass
        total_loss.backward()
        optimizer.step()

        # Accumulate total loss
        total_loss += total_loss.detach().item()

    # Return average loss over batches (if accumulated)
    return total_loss.item() / len(dataloader)


def main():
    # Hyperparameters
    input_size = 8996  # Your feature dimension
    hidden_size = 128
    num_layers = 2
    num_heads = 4
    num_actions = 12  # e.g., 4 moves × 2 Pokémon + switches

    # TODO: add evaluation; create a dataset opject for sequences that handles padding


if __name__ == "__main__":
    main()
