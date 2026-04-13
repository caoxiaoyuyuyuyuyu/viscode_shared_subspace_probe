#!/usr/bin/env python3
"""E2E float32 overflow test for run_residualized_a2_perm_aggregate.

Constructs synthetic hidden-state tensors with magnitudes characteristic of
deep-layer coder models (norm ~10-100), passes them through the full aggregate
A2 permutation pipeline with n=252, d=6144 (codestral-size), and asserts:
  1. No FloatingPointError despite seterr(over='raise', invalid='raise')
  2. Observed CKA is finite and nonzero (not the overflow-zeroed 0.0 bug)
  3. Null distribution is finite

This reproduces the reviewer-reported bug where float32 HSIC denom
sqrt(hsic_xx * hsic_yy) overflowed (>3.4e38) and produced CKA=0.0.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from stage_b_residualized_cka import (
    run_residualized_a2_perm_aggregate,
    fit_format_classifier,
    FORMATS,
)


def _make_data(n=252, n_layers=3, hidden_dim=6144, scale=30.0, seed=0):
    """Synthetic hidden states with large norms and some format-shared structure."""
    rng = np.random.RandomState(seed)
    shared = rng.randn(n, n_layers, hidden_dim).astype(np.float32) * scale
    data = {}
    for i, fmt in enumerate(FORMATS):
        fmt_noise = rng.randn(n, n_layers, hidden_dim).astype(np.float32) * (scale * 0.3)
        data[fmt] = shared + fmt_noise
    return data


def test_aggregate_no_overflow():
    """Run the full aggregate A2 pipeline on large-norm inputs."""
    np.seterr(over="raise", invalid="raise")
    n = 252
    hidden_dim = 6144
    layers = [0, 1, 2]

    data = _make_data(n=n, n_layers=len(layers), hidden_dim=hidden_dim, scale=30.0)

    format_weights = []
    for li in range(len(layers)):
        W, _ = fit_format_classifier(data, li, n)
        format_weights.append(W)

    result = run_residualized_a2_perm_aggregate(
        data, format_weights, layers, n_triples=n, n_perm=20
    )

    print(f"[e2e] aggregate result = {result}")
    obs = result["observed"]
    null_mean = result["null_mean"]
    assert np.isfinite(obs), f"observed CKA not finite: {obs}"
    assert obs != 0.0, f"observed CKA is exactly 0.0 — overflow bug! {obs}"
    assert np.isfinite(null_mean), f"null_mean not finite: {null_mean}"
    print(f"[e2e] OK: observed={obs:.4f}, null_mean={null_mean:.4f}, p={result['p_value']}")


def test_extreme_scale():
    """Extreme scale (100x) — pre-fix this would overflow float32 in HSIC denom."""
    np.seterr(over="raise", invalid="raise")
    n = 100
    hidden_dim = 4096
    layers = [0]
    data = _make_data(n=n, n_layers=1, hidden_dim=hidden_dim, scale=100.0)
    format_weights = []
    for li in range(len(layers)):
        W, _ = fit_format_classifier(data, li, n)
        format_weights.append(W)
    result = run_residualized_a2_perm_aggregate(
        data, format_weights, layers, n_triples=n, n_perm=5
    )
    print(f"[e2e extreme] result = {result}")
    assert np.isfinite(result["observed"])
    assert result["observed"] != 0.0
    print("[e2e extreme] OK")


if __name__ == "__main__":
    test_aggregate_no_overflow()
    test_extreme_scale()
    print("\n=== ALL E2E OVERFLOW TESTS PASSED ===")
