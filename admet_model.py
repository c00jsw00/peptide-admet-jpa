#!/usr/bin/env python3
"""
admet_model.py
==============

Shared model definitions for the peptide ADMET pipeline (v4.0 real-data
edition), so the trainer and predictor always agree on architecture and save
format.

Model class
-----------
* ``MixedADMETMLP`` — the v4.0 model: one shared trunk feeding task heads.
  The class is **generic** (parameterized by ``input_dim`` + ``endpoints`` +
  per-endpoint kinds), and in v4.0 it is instantiated **once per endpoint** as
  a single-head model, because the four real-data endpoints (Hemolysis,
  Half-life, Caco-2, PAMPA/MDCK) are disjoint molecule sets spanning two input
  modalities:

    - binary     head  — Hemolysis (sigmoid)
    - regression head  — Half-life (log10 s), Caco-2 (logPapp),
                         PAMPA/MDCK (logPapp) (identity)

  Each head has its own Linear layer, so tasks are coupled only through the
  shared trunk (a standard multi-task design). ``input_dim`` is 428 for the
  sequence endpoints and 2,265 for the molecular endpoints.

* ``ADMETMLP`` — the original single-trunk, 5 binary sigmoid heads, kept only
  for backward-compatible loading of pre-v3.0 checkpoints (none ship with v4.0).

Save format (a single torch file)
---------------------------------
``MixedADMETMLP`` checkpoints store ``state_dict`` plus the metadata needed to
rebuild the module deterministically: ``input_dim``, ``endpoints`` (list of
names), ``kinds`` (per-endpoint kind), ``num_classes`` (per-endpoint, for
multiclass), and per-binary-endpoint ``pos_weights``.  The trainer and
predictor both derive the module from this metadata, so they cannot drift.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

# v3.0 endpoint set (imported once — single source of truth)
from endpoint_config import (
    ENDPOINTS, ENDPOINT_NAMES, ENDPOINT_BY_NAME,
    KIND_BINARY, KIND_MULTICLASS, KIND_REGRESSION,
)


# =========================================================================== #
# v2.0 model (kept for loading old checkpoints)
# =========================================================================== #
class ADMETMLP(nn.Module):
    """MLP with per-endpoint binary heads (5 heads, v2.0)."""

    def __init__(self, input_dim: int = 428, hidden=(256, 128),
                 num_endpoints: int = 5, dropout: float = 0.25):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(),
                       nn.Dropout(dropout)]
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(prev, num_endpoints)

    def forward(self, x):
        return self.head(self.trunk(x))


# =========================================================================== #
# v3.0 mixed multi-task model
# =========================================================================== #
class MixedADMETMLP(nn.Module):
    """
    Shared trunk + per-task heads for a mixed endpoint set.

    ``endpoints``: ordered list of endpoint names (must match the column order
    the trainer/predictor use — i.e. ``endpoint_config.ENDPOINT_NAMES``).
    """

    def __init__(self, input_dim: int = 428, hidden=(256, 128),
                 endpoints: Optional[Sequence[str]] = None,
                 dropout: float = 0.25):
        super().__init__()
        self.endpoints: List[str] = list(endpoints) if endpoints is not None \
            else list(ENDPOINT_NAMES)
        self.hidden = tuple(hidden)
        self.dropout = dropout
        self.kinds: List[str] = [ENDPOINT_BY_NAME[e].kind for e in self.endpoints]
        self.num_classes: List[int] = [ENDPOINT_BY_NAME[e].num_classes
                                       for e in self.endpoints]

        layers, prev = [], input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(),
                       nn.Dropout(dropout)]
            prev = h
        self.trunk = nn.Sequential(*layers)

        # per-task heads
        self.binary_head = nn.ModuleDict({
            e: nn.Linear(prev, 1) for e in self.endpoints
            if ENDPOINT_BY_NAME[e].kind == KIND_BINARY})
        self.multiclass_head = nn.ModuleDict({
            e: nn.Linear(prev, ENDPOINT_BY_NAME[e].num_classes)
            for e in self.endpoints
            if ENDPOINT_BY_NAME[e].kind == KIND_MULTICLASS})
        self.regression_head = nn.ModuleDict({
            e: nn.Linear(prev, 1) for e in self.endpoints
            if ENDPOINT_BY_NAME[e].kind == KIND_REGRESSION})

    # ---- indexing helpers ------------------------------------------------- #
    @property
    def binary_names(self) -> List[str]:
        return [e for e in self.endpoints if ENDPOINT_BY_NAME[e].kind == KIND_BINARY]

    @property
    def multiclass_names(self) -> List[str]:
        return [e for e in self.endpoints if ENDPOINT_BY_NAME[e].kind == KIND_MULTICLASS]

    @property
    def regression_names(self) -> List[str]:
        return [e for e in self.endpoints if ENDPOINT_BY_NAME[e].kind == KIND_REGRESSION]

    def forward(self, x) -> Dict[str, torch.Tensor]:
        """
        Returns a dict of raw outputs per endpoint:
            binary      -> (B, 1) logits
            multiclass  -> (B, C) logits
            regression  -> (B, 1) value
        """
        h = self.trunk(x)
        out: Dict[str, torch.Tensor] = {}
        for e in self.binary_names:
            out[e] = self.binary_head[e](h)
        for e in self.multiclass_names:
            out[e] = self.multiclass_head[e](h)
        for e in self.regression_names:
            out[e] = self.regression_head[e](h)
        return out


# =========================================================================== #
# Save / load
# =========================================================================== #
def save_admet_model(model: nn.Module, path: Path, input_dim: int = 428,
                     endpoints=None, pos_weights=None):
    """Save a v2.0 ``ADMETMLP`` (legacy format, 5 binary endpoints)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_class': 'ADMETMLP',
        'state_dict': model.state_dict(),
        'input_dim': input_dim,
        'num_endpoints': len(endpoints or ENDPOINTS),
        'endpoints': list(endpoints or [e.name for e in ENDPOINTS[:5]]),
        'pos_weights': (pos_weights.detach().cpu().tolist()
                        if isinstance(pos_weights, torch.Tensor)
                        else list(pos_weights or [])),
    }, path)
    return path


def save_mixed_model(model: 'MixedADMETMLP', path: Path,
                     pos_weights: Optional[Dict[str, float]] = None):
    """Save a v3.0 ``MixedADMETMLP`` with full metadata for rebuild."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        'model_class': 'MixedADMETMLP',
        'state_dict': model.state_dict(),
        'input_dim': _input_dim_from_model(model),
        'endpoints': model.endpoints,
        'kinds': model.kinds,
        'num_classes': model.num_classes,
        'pos_weights': pos_weights or {},
        # v4.2: persist architecture so a widened/changed trunk still rebuilds
        # exactly (older checkpoints lack these keys -> defaults on load).
        'hidden': tuple(getattr(model, 'hidden', (256, 128))),
        'dropout': float(getattr(model, 'dropout', 0.25)),
    }
    torch.save(blob, path)
    return path


def load_admet_model(path: str) -> tuple:
    """
    Load either a v2.0 ``ADMETMLP`` or a v3.0 ``MixedADMETMLP`` from one file.
    Returns ``(model, meta)``; ``meta['model_class']`` tells which.
    """
    path = Path(path)
    blob = torch.load(path, map_location='cpu', weights_only=False)
    cls = blob.get('model_class', 'ADMETMLP')
    if cls == 'MixedADMETMLP':
        model = MixedADMETMLP(input_dim=blob['input_dim'],
                              endpoints=blob['endpoints'],
                              hidden=blob.get('hidden', (256, 128)),
                              dropout=blob.get('dropout', 0.25))
        model.load_state_dict(blob['state_dict'])
    else:
        model = ADMETMLP(input_dim=blob['input_dim'],
                         num_endpoints=blob['num_endpoints'])
        model.load_state_dict(blob['state_dict'])
    model.eval()
    return model, blob


def _input_dim_from_model(model) -> int:
    # first Linear of the trunk
    for m in model.trunk:
        if isinstance(m, nn.Linear):
            return m.in_features
    raise ValueError('cannot infer input_dim from trunk')


# =========================================================================== #
# Inference helpers (v3.0)
# =========================================================================== #
@torch.no_grad()
def predict_mixed(model: 'MixedADMETMLP', X: np.ndarray,
                  batch_size: int = 1024) -> Dict[str, np.ndarray]:
    """
    Run a MixedADMETMLP over ``X`` and return per-endpoint predictions in a
    *readable* form (not raw logits):

        binary      -> (B,)  sigmoid probability of the positive class
        multiclass  -> (B,)  argmax class id (int)
        regression  -> (B,)  predicted value (float)
    """
    model.eval()
    X = np.asarray(X, dtype=np.float32)
    acc: Dict[str, List[np.ndarray]] = {e: [] for e in model.endpoints}
    for i in range(0, len(X), batch_size):
        t = torch.from_numpy(X[i:i + batch_size])
        out = model(t)
        for e in model.endpoints:
            kind = ENDPOINT_BY_NAME[e].kind
            if kind == KIND_BINARY:
                acc[e].append(torch.sigmoid(out[e]).squeeze(1).numpy())
            elif kind == KIND_MULTICLASS:
                acc[e].append(out[e].argmax(dim=1).numpy())
            else:
                acc[e].append(out[e].squeeze(1).numpy())
    return {e: np.concatenate(v, axis=0) for e, v in acc.items()}


def predict_proba_v2(model: nn.Module, X: np.ndarray,
                     batch_size: int = 1024) -> np.ndarray:
    """Legacy v2.0 helper: sigmoid probabilities for 5 binary endpoints."""
    X = np.asarray(X, dtype=np.float32)
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            t = torch.from_numpy(X[i:i + batch_size])
            out.append(torch.sigmoid(model(t)).numpy())
    return np.concatenate(out, axis=0)


if __name__ == '__main__':
    m = MixedADMETMLP()
    n = sum(p.numel() for p in m.parameters())
    print(f'MixedADMETMLP: {len(ENDPOINT_NAMES)} endpoints, {n:,} params')
    print('  binary:      ', m.binary_names)
    print('  multiclass:  ', m.multiclass_names)
    print('  regression:  ', m.regression_names)
    x = torch.randn(8, 428)
    out = m(x)
    for e, t in out.items():
        print(f'  {e:20s} {tuple(t.shape)}')
