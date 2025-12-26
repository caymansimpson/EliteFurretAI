import torch
from elitefurretai.agents.behavior_clone_player import FlexibleThreeHeadedModel


class RNaDAgent(torch.nn.Module):
    """
    RL Agent wrapper around FlexibleThreeHeadedModel.
    Handles hidden states and value function transformation.
    """
    def __init__(self, model: FlexibleThreeHeadedModel):
        super().__init__()
        self.model = model

    def get_initial_state(self, batch_size, device):
        # LSTM state is (h_0, c_0)
        # shape: (num_layers * num_directions, batch, hidden_size)
        num_directions = 2  # bidirectional
        h = torch.zeros(self.model.lstm.num_layers * num_directions, batch_size, self.model.lstm_hidden_size, device=device)
        c = torch.zeros(self.model.lstm.num_layers * num_directions, batch_size, self.model.lstm_hidden_size, device=device)
        return (h, c)

    def forward(self, x, hidden_state=None, mask=None):
        """
        Returns:
            turn_logits: (batch, seq, num_actions)
            tp_logits: (batch, seq, num_tp_actions)
            value: (batch, seq) - Value in [-1, 1]
            next_hidden: (h, c)
        """
        turn_logits, tp_logits, value, next_hidden = self.model.forward_with_hidden(x, hidden_state, mask)
        return turn_logits, tp_logits, value, next_hidden
