"""Simple token baselines for interpreting CVAE reconstruction metrics."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from aptafind.generation.data import AptamerSequenceDataset
from aptafind.generation.tokenizer import DNATokenizer


@dataclass(frozen=True)
class TokenBaselineMetrics:
    negative_log_likelihood: float
    perplexity: float
    token_accuracy: float
    token_count: int

    def state_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _next_token_arrays(
    dataset: AptamerSequenceDataset, tokenizer: DNATokenizer
) -> tuple[np.ndarray, np.ndarray]:
    token_rows = dataset.token_ids.numpy()
    previous = token_rows[:, :-1].reshape(-1)
    following = token_rows[:, 1:].reshape(-1)
    mask = following != tokenizer.pad_id
    return previous[mask], following[mask]


def evaluate_unigram_baseline(
    train_dataset: AptamerSequenceDataset,
    evaluation_dataset: AptamerSequenceDataset,
    tokenizer: DNATokenizer,
    *,
    smoothing: float = 1.0,
) -> TokenBaselineMetrics:
    """Evaluate a smoothed training-set next-token frequency baseline."""

    if smoothing <= 0:
        raise ValueError("smoothing must be positive.")
    allowed_ids = [tokenizer.eos_id, *range(3, tokenizer.vocabulary_size)]
    counts = np.zeros(tokenizer.vocabulary_size, dtype=np.float64)
    counts[allowed_ids] = smoothing
    _, train_following = _next_token_arrays(train_dataset, tokenizer)
    np.add.at(counts, train_following, 1.0)
    probabilities = counts / counts.sum()

    _, evaluation_following = _next_token_arrays(evaluation_dataset, tokenizer)
    if len(evaluation_following) == 0:
        raise ValueError("Cannot evaluate a token baseline on an empty dataset.")
    losses = -np.log(probabilities[evaluation_following])
    prediction = int(probabilities.argmax())
    negative_log_likelihood = float(losses.mean())
    return TokenBaselineMetrics(
        negative_log_likelihood=negative_log_likelihood,
        perplexity=math.exp(negative_log_likelihood),
        token_accuracy=float((evaluation_following == prediction).mean()),
        token_count=int(len(evaluation_following)),
    )


def evaluate_bigram_baseline(
    train_dataset: AptamerSequenceDataset,
    evaluation_dataset: AptamerSequenceDataset,
    tokenizer: DNATokenizer,
    *,
    smoothing: float = 1.0,
) -> TokenBaselineMetrics:
    """Evaluate a smoothed first-order DNA transition baseline."""

    if smoothing <= 0:
        raise ValueError("smoothing must be positive.")
    allowed_ids = [tokenizer.eos_id, *range(3, tokenizer.vocabulary_size)]
    transition_counts = np.zeros(
        (tokenizer.vocabulary_size, tokenizer.vocabulary_size), dtype=np.float64
    )
    transition_counts[:, allowed_ids] = smoothing
    train_previous, train_following = _next_token_arrays(train_dataset, tokenizer)
    np.add.at(transition_counts, (train_previous, train_following), 1.0)
    transition_probabilities = transition_counts / transition_counts.sum(
        axis=1, keepdims=True
    )

    evaluation_previous, evaluation_following = _next_token_arrays(
        evaluation_dataset, tokenizer
    )
    if len(evaluation_following) == 0:
        raise ValueError("Cannot evaluate a token baseline on an empty dataset.")
    probabilities = transition_probabilities[
        evaluation_previous, evaluation_following
    ]
    predictions = transition_probabilities[evaluation_previous].argmax(axis=1)
    negative_log_likelihood = float((-np.log(probabilities)).mean())
    return TokenBaselineMetrics(
        negative_log_likelihood=negative_log_likelihood,
        perplexity=math.exp(negative_log_likelihood),
        token_accuracy=float((predictions == evaluation_following).mean()),
        token_count=int(len(evaluation_following)),
    )
