"""Recurrent controller of AdaImprint (step 5) and streaming feature normalisation.

Independent of FIO and of any embedding or storage backend: the caller passes
normalised features and receives the importance correction delta_t and the
policy logits.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class AdaImprintConfig:
    """Dimensions and bounds for the recurrent retention controller."""

    input_dim: int
    hidden_dim: int = 128
    max_log_importance_correction: float = 2.0
    num_policy_actions: int = 3  # write, merge, update (discard is the gate)


class RunningFeatureNorm(nn.Module):
    """Running per-feature normalization for streaming controller inputs."""

    def __init__(self, feature_dim: int, epsilon: float = 1e-5):
        super().__init__()
        self.epsilon = float(epsilon)
        self.register_buffer("count", torch.zeros((), dtype=torch.long))
        self.register_buffer("mean", torch.zeros(feature_dim))
        self.register_buffer("m2", torch.zeros(feature_dim))

    @torch.no_grad()
    def update(self, features: Tensor) -> None:
        """Update statistics from a [batch, feature_dim] tensor."""
        if features.ndim == 1:
            features = features.unsqueeze(0)
        if features.ndim != 2 or features.shape[-1] != self.mean.numel():
            raise ValueError("features must have shape [batch, feature_dim]")
        batch_count = features.shape[0]
        if batch_count == 0:
            return
        batch_mean = features.mean(dim=0)
        batch_m2 = ((features - batch_mean) ** 2).sum(dim=0)
        old_count = self.count.to(features.dtype)
        new_count = old_count + batch_count
        delta = batch_mean - self.mean
        self.mean.add_(delta * (batch_count / new_count))
        self.m2.add_(batch_m2 + delta.square() * old_count * batch_count / new_count)
        self.count.add_(batch_count)

    def forward(self, features: Tensor) -> Tensor:
        if self.count.item() < 2:
            return features
        variance = self.m2 / (self.count.to(self.m2.dtype) - 1).clamp_min(1)
        return (features - self.mean) / torch.sqrt(variance + self.epsilon)


class AdaImprintController(nn.Module):
    """LSTM controller producing adaptive importance and soft write decisions.

    Input features should include prototype history, novelty, redundancy,
    predictive surprise, and current budget occupancy. Policy outputs are
    preferences; hard memory-budget enforcement remains the caller's job.
    """

    def __init__(self, config: AdaImprintConfig):
        super().__init__()
        if config.input_dim <= 0 or config.hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        self.config = config
        self.lstm = nn.LSTMCell(config.input_dim, config.hidden_dim)
        self.correction_head = nn.Sequential(
            nn.Linear(config.hidden_dim + config.input_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, 1),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(config.hidden_dim + config.input_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, config.num_policy_actions),
        )

    def initial_state(self, batch_size: int, *, device=None, dtype=None):
        parameter = next(self.parameters())
        device = device or parameter.device
        dtype = dtype or parameter.dtype
        shape = (batch_size, self.config.hidden_dim)
        return torch.zeros(shape, device=device, dtype=dtype), torch.zeros(
            shape, device=device, dtype=dtype
        )

    def forward(
        self,
        features: Tensor,
        state: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """Process one timestep; features have shape [batch, input_dim]."""
        if features.ndim == 1:
            features = features.unsqueeze(0)
        if features.ndim != 2 or features.shape[-1] != self.config.input_dim:
            raise ValueError("features must have shape [batch, input_dim]")
        if state is None:
            state = self.initial_state(features.shape[0], device=features.device,
                                       dtype=features.dtype)
        h, c = self.lstm(features, state)
        joint = torch.cat((h, features), dim=-1)
        correction = self.config.max_log_importance_correction * torch.tanh(
            self.correction_head(joint).squeeze(-1)
        )
        return {
            "log_importance_correction": correction,
            "policy_logits": self.policy_head(joint),
            "state": (h, c),
        }
