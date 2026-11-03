"""Small, isolated building blocks for the AdaImprint experiment.

This module deliberately does not modify or import the production FIO build
pipeline. The caller supplies normalized stream features and prediction
logits, which keeps the controller independent of a particular embedding or
storage backend.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class AdaImprintConfig:
    """Dimensions and bounds for the recurrent retention controller."""

    input_dim: int
    hidden_dim: int = 128
    max_log_importance_correction: float = 2.0
    num_policy_actions: int = 4  # write, merge, update, discard


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
        state: Optional[tuple] = None,
        base_importance: Optional[Tensor] = None,
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
        result = {
            "log_importance_correction": correction,
            "policy_logits": self.policy_head(joint),
            "state": (h, c),
        }
        if base_importance is not None:
            base = torch.as_tensor(base_importance, device=features.device,
                                   dtype=features.dtype).reshape(-1)
            result["adaptive_importance"] = torch.exp(
                torch.log(base.clamp_min(1e-8)) + correction
            )
        return result


def self_supervised_memory_loss(
    next_logits: Tensor,
    next_targets: Tensor,
    past_logits: Tensor,
    past_targets: Tensor,
    memory_cost: Tensor,
    redundancy_penalty: Tensor,
    *,
    beta: float = 0.01,
    gamma: float = 0.1,
    past_weight: float = 1.0,
) -> Dict[str, Tensor]:
    """Combine next-event prediction and sampled-past reconstruction losses.

    ``past_logits`` and ``past_targets`` represent randomly sampled past
    interactions that remain represented by compressed memory. Including this
    term reduces the incentive to retain only routine, easy-to-predict events.
    Cost and redundancy are caller-computed scalar tensors, kept explicit so
    the experiment can define its own budget units and merge semantics.
    """
    if next_logits.shape[0] != next_targets.shape[0]:
        raise ValueError("next logits and targets must have matching batches")
    if past_logits.shape[0] != past_targets.shape[0]:
        raise ValueError("past logits and targets must have matching batches")
    next_loss = F.cross_entropy(next_logits, next_targets.long())
    past_loss = F.cross_entropy(past_logits, past_targets.long())
    memory_cost = torch.as_tensor(memory_cost, device=next_loss.device,
                                  dtype=next_loss.dtype).mean()
    redundancy_penalty = torch.as_tensor(
        redundancy_penalty, device=next_loss.device, dtype=next_loss.dtype
    ).mean()
    total = next_loss + past_weight * past_loss + beta * memory_cost + gamma * redundancy_penalty
    return {
        "loss": total,
        "next_prediction": next_loss,
        "past_reconstruction": past_loss,
        "memory_cost": memory_cost,
        "redundancy": redundancy_penalty,
    }
