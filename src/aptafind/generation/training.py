"""Deterministic training and evaluation utilities for the sequence CVAE."""

from __future__ import annotations

import copy
import math
import random
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from aptafind.generation.model import ConditionalSequenceVAE, sequence_cvae_loss


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 40
    batch_size: int = 16
    learning_rate: float = 1e-3
    beta_max: float = 0.05
    beta_warmup_epochs: int = 15
    patience: int = 8
    gradient_clip_norm: float = 5.0
    seed: int = 42
    device: str = "cpu"

    def __post_init__(self) -> None:
        for name in ("epochs", "batch_size", "patience"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive.")
        if self.beta_warmup_epochs < 0:
            raise ValueError("beta_warmup_epochs cannot be negative.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.beta_max < 0:
            raise ValueError("beta_max cannot be negative.")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive.")

    def state_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EpochMetrics:
    loss: float
    reconstruction_loss: float
    kl_divergence: float
    token_accuracy: float
    perplexity: float
    sample_count: int
    token_count: int

    def state_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass
class TrainingResult:
    history: list[dict[str, Any]]
    best_epoch: int
    stopped_epoch: int
    best_validation_loss: float


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch generators."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    """Resolve a requested device and fail instead of silently changing it."""

    normalized = requested.lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if normalized == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS was requested but is unavailable.")
    if normalized not in ("cpu", "cuda", "mps"):
        raise ValueError("device must be one of: auto, cpu, cuda, mps.")
    return torch.device(normalized)


def beta_for_epoch(epoch: int, config: TrainingConfig) -> float:
    """Linearly warm the KL term to reduce posterior collapse pressure."""

    if config.beta_warmup_epochs == 0:
        return config.beta_max
    return config.beta_max * min(1.0, epoch / config.beta_warmup_epochs)


def _run_epoch(
    model: ConditionalSequenceVAE,
    loader: DataLoader[dict[str, torch.Tensor]],
    *,
    device: torch.device,
    beta: float,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip_norm: float,
) -> EpochMetrics:
    training = optimizer is not None
    model.train(training)
    reconstruction_sum = 0.0
    kl_sum = 0.0
    correct_token_sum = 0.0
    token_count_sum = 0
    sample_count_sum = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            token_ids = batch["token_ids"].to(device)
            lengths = batch["length"].to(device)
            conditions = batch["condition"].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)

            output = model(
                token_ids,
                lengths,
                conditions,
                sample_latent=training,
            )
            loss = sequence_cvae_loss(
                output,
                token_ids[:, 1:],
                pad_token_id=model.config.pad_token_id,
                beta=beta,
            )
            if training:
                loss.loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()

            batch_token_count = int(loss.token_count.detach().cpu())
            batch_size = int(token_ids.shape[0])
            reconstruction_sum += (
                float(loss.reconstruction_loss.detach().cpu()) * batch_token_count
            )
            kl_sum += float(loss.kl_divergence.detach().cpu()) * batch_size
            correct_token_sum += (
                float(loss.token_accuracy.detach().cpu()) * batch_token_count
            )
            token_count_sum += batch_token_count
            sample_count_sum += batch_size

    if sample_count_sum == 0 or token_count_sum == 0:
        raise ValueError("Cannot evaluate an empty dataset partition.")
    reconstruction_loss = reconstruction_sum / token_count_sum
    kl_divergence = kl_sum / sample_count_sum
    total_loss = reconstruction_loss + beta * kl_divergence
    return EpochMetrics(
        loss=total_loss,
        reconstruction_loss=reconstruction_loss,
        kl_divergence=kl_divergence,
        token_accuracy=correct_token_sum / token_count_sum,
        perplexity=math.exp(min(reconstruction_loss, 20.0)),
        sample_count=sample_count_sum,
        token_count=token_count_sum,
    )


def evaluate_model(
    model: ConditionalSequenceVAE,
    loader: DataLoader[dict[str, torch.Tensor]],
    *,
    device: torch.device,
    beta: float,
) -> EpochMetrics:
    """Evaluate deterministically by decoding from the posterior mean."""

    return _run_epoch(
        model,
        loader,
        device=device,
        beta=beta,
        optimizer=None,
        gradient_clip_norm=1.0,
    )


def train_model(
    model: ConditionalSequenceVAE,
    train_loader: DataLoader[dict[str, torch.Tensor]],
    validation_loader: DataLoader[dict[str, torch.Tensor]],
    config: TrainingConfig,
) -> TrainingResult:
    """Train with KL warmup and validation-loss early stopping."""

    seed_everything(config.seed)
    device = resolve_device(config.device)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=config.learning_rate)

    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_validation_loss = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_epoch = config.epochs

    for epoch in range(1, config.epochs + 1):
        beta = beta_for_epoch(epoch, config)
        train_metrics = _run_epoch(
            model,
            train_loader,
            device=device,
            beta=beta,
            optimizer=optimizer,
            gradient_clip_norm=config.gradient_clip_norm,
        )
        validation_metrics = evaluate_model(
            model,
            validation_loader,
            device=device,
            beta=beta,
        )
        history.append(
            {
                "epoch": epoch,
                "beta": beta,
                "train": train_metrics.state_dict(),
                "validation": validation_metrics.state_dict(),
            }
        )

        if validation_metrics.loss < best_validation_loss - 1e-8:
            best_validation_loss = validation_metrics.loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                stopped_epoch = epoch
                break

    if best_state is None:
        raise RuntimeError("Training completed without a valid model state.")
    model.load_state_dict(best_state)
    return TrainingResult(
        history=history,
        best_epoch=best_epoch,
        stopped_epoch=stopped_epoch,
        best_validation_loss=best_validation_loss,
    )
