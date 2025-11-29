import torch
from torch.distributions import Categorical
from typing import Optional


# TODO: implement more complicated architecture
class ActorCritic(torch.nn.Module):
    """
    The Actor-Critic model that shares a common body and has two heads:
    1. Actor (Policy Head): Outputs a probability distribution over actions.
    2. Critic (Value Head): Outputs an estimate of the value of the current state.

    This class defines the neural network architecture. It will be instantiated once
    in the central Learner (on GPU) and copied to each ActorWorker (on CPU).

    Supports loading pretrained BC model weights for warm-start initialization.
    """
    def __init__(self, state_dim, action_dim, hidden_sizes=[512, 256]):
        """
        Initializes the neural network layers.

        Args:
            state_dim (int): The dimensionality of the state observation.
            action_dim (int): The number of possible discrete actions (e.g., 2025 for MDBO).
            hidden_sizes (list): List of hidden layer sizes for the shared body.
        """
        super(ActorCritic, self).__init__()

        # A shared body for feature extraction from the state.
        # This can be a series of linear layers, CNNs, or a Transformer,
        # depending on your state representation.
        layers = []
        prev_size = state_dim
        for hidden_size in hidden_sizes:
            layers.extend([
                torch.nn.Linear(prev_size, hidden_size),
                torch.nn.ReLU(),
            ])
            prev_size = hidden_size

        self.shared_body = torch.nn.Sequential(*layers)

        # The Actor head, which outputs action probabilities.
        self.actor_head = torch.nn.Linear(prev_size, action_dim)

        # The Critic head, which estimates the state value.
        self.critic_head = torch.nn.Linear(prev_size, 1)

    def forward(self, state, action_mask=None):
        """
        Performs a forward pass through the network.

        Args:
            state (torch.Tensor): The input state tensor.
            action_mask (torch.Tensor, optional): Binary mask for valid actions.

        Returns:
            (torch.Tensor, torch.Tensor): A tuple containing:
                - Action logits from the actor head.
                - The state value from the critic head.
        """
        features = self.shared_body(state)
        action_logits = self.actor_head(features)

        # Apply action mask if provided
        if action_mask is not None:
            action_logits = action_logits.masked_fill(~action_mask.bool(), float('-inf'))

        state_value = self.critic_head(features)
        return action_logits, state_value

    def get_action_and_value(self, state, action_mask=None):
        """
        Used by workers during rollouts to select an action and get its log probability
        and the current state's value. This method includes sampling from the
        action distribution.

        Args:
            state (torch.Tensor): The current state observation.
            action_mask (torch.Tensor, optional): Binary mask for valid actions.

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor): A tuple containing:
                - The chosen action.
                - The log probability of the chosen action.
                - The estimated value of the state.
        """
        action_logits, state_value = self.forward(state, action_mask)

        # Create a categorical distribution and sample an action
        action_distribution = Categorical(logits=action_logits)
        action = action_distribution.sample()
        log_prob = action_distribution.log_prob(action)

        # Return the action, its log probability, and the state value
        return action, log_prob, state_value

    def get_value_and_log_prob(self, state, action, action_mask=None):
        """
        Used by the learner during PPO updates. It computes the value of a given
        state and the log probability of an action that was previously taken.

        Args:
            state (torch.Tensor): A batch of states from the experience buffer.
            action (torch.Tensor): A batch of actions taken in those states.
            action_mask (torch.Tensor, optional): Binary mask for valid actions.

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor): A tuple containing:
                - The estimated values of the states.
                - The log probabilities of the given actions in the given states.
                - The entropy of the action distribution.
        """
        action_logits, state_value = self.forward(state, action_mask)
        action_distribution = Categorical(logits=action_logits)
        log_prob = action_distribution.log_prob(action)
        entropy = action_distribution.entropy()

        return state_value, log_prob, entropy

    def load_bc_model_weights(self, bc_model_path: str, freeze_shared_body: bool = False):
        """
        Load pretrained behavioral cloning model weights for warm-start initialization.

        Args:
            bc_model_path (str): Path to the saved BC model checkpoint.
            freeze_shared_body (bool): If True, freeze the shared body weights during RL training.
        """
        # Load the BC model checkpoint
        bc_checkpoint = torch.load(bc_model_path, map_location='cpu')

        # The BC model has a different architecture (TwoHeadedHybridModel)
        # We need to extract compatible weights for the shared body and actor head

        # Map BC model weights to RL model
        # BC has: ff_stack, pos_embedding, lstm, self_attn, action_head, win_head
        # RL has: shared_body, actor_head, critic_head

        # For now, we'll load the action_head weights directly
        # You may want to customize this mapping based on your architecture
        current_state_dict = self.state_dict()

        # Try to load actor head from BC model's action head
        if 'action_head.weight' in bc_checkpoint:
            current_state_dict['actor_head.weight'] = bc_checkpoint['action_head.weight']
            current_state_dict['actor_head.bias'] = bc_checkpoint['action_head.bias']
            print("Loaded BC action_head weights into RL actor_head")

        # Optionally load ff_stack weights into shared_body
        # This requires matching dimensions - skipping for now due to architecture differences

        self.load_state_dict(current_state_dict)

        if freeze_shared_body:
            for param in self.shared_body.parameters():
                param.requires_grad = False
            print("Froze shared_body parameters")

    @staticmethod
    def create_from_bc_model(bc_model_path: str, state_dim: int, action_dim: int,
                             hidden_sizes: Optional[list] = None) -> 'ActorCritic':
        """
        Factory method to create an ActorCritic model initialized with BC weights.

        Args:
            bc_model_path (str): Path to the saved BC model checkpoint.
            state_dim (int): Dimensionality of the state observation.
            action_dim (int): Number of possible discrete actions.
            hidden_sizes (list, optional): Hidden layer sizes. If None, uses default.

        Returns:
            ActorCritic: Initialized model with BC weights.
        """
        if hidden_sizes is None:
            hidden_sizes = [512, 256]

        model = ActorCritic(state_dim, action_dim, hidden_sizes)
        model.load_bc_model_weights(bc_model_path, freeze_shared_body=False)
        return model
