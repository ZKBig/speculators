"""``--init-backbone-from``: a DFlash-format drafter's backbone lands in the draft
model by name, everything else stays fresh, and drift fails loudly."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from test_xpress_backbone import _tiny_config

from speculators.models.xpress.core import XPressDraftModel

_spec = importlib.util.spec_from_file_location(
    "train_script", Path(__file__).resolve().parents[3] / "scripts" / "train.py"
)
_train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_train)  # type: ignore[union-attr]

BACKBONE_PREFIXES = ("fc.", "hidden_norm.", "norm.", "layers.")


def _drafter_state(model: XPressDraftModel) -> dict[str, torch.Tensor]:
    """A z-lab style checkpoint: the backbone tensors only, with fresh values."""
    return {
        k: torch.randn_like(v)
        for k, v in model.state_dict().items()
        if k.startswith(BACKBONE_PREFIXES)
    }


def test_backbone_loads_and_head_stays_fresh(tmp_path):
    model = XPressDraftModel(_tiny_config())
    before = {k: v.clone() for k, v in model.state_dict().items()}
    drafter = _drafter_state(model)
    save_file(drafter, str(tmp_path / "model.safetensors"))

    _train.init_backbone_from(model, str(tmp_path))

    after = model.state_dict()
    for k, v in drafter.items():
        assert torch.equal(after[k], v), k
    for k in after:
        if k.startswith("refiner_head."):
            assert torch.equal(after[k], before[k]), f"{k} was overwritten"


def test_shape_drift_is_rejected(tmp_path):
    model = XPressDraftModel(_tiny_config())
    drafter = _drafter_state(model)
    drafter["fc.weight"] = torch.randn(drafter["fc.weight"].shape[0], 7)
    save_file(drafter, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="shape mismatch"):
        _train.init_backbone_from(model, str(tmp_path))


def test_unknown_tensor_is_rejected(tmp_path):
    model = XPressDraftModel(_tiny_config())
    drafter = _drafter_state(model)
    drafter["conv_taps.weight"] = torch.randn(3)
    save_file(drafter, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="no home"):
        _train.init_backbone_from(model, str(tmp_path))


def test_missing_backbone_tensor_is_rejected(tmp_path):
    model = XPressDraftModel(_tiny_config())
    drafter = _drafter_state(model)
    del drafter["norm.weight"]
    save_file(drafter, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="did not provide"):
        _train.init_backbone_from(model, str(tmp_path))
