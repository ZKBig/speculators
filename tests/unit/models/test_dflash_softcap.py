"""Verifier logits used as distillation targets honour final_logit_softcapping."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_xpress_backbone import _tiny_config

from speculators.models.xpress.core import XPressDraftModel


def test_softcap_is_noop_when_unset():
    model = XPressDraftModel(_tiny_config())
    assert model.final_logit_softcapping is None
    x = torch.randn(2, 5)
    assert torch.equal(model._softcap(x), x)


def test_softcap_applies_tanh_squash():
    cfg = _tiny_config()
    cfg.transformer_layer_config.final_logit_softcapping = 30.0
    model = XPressDraftModel(cfg)
    assert model.final_logit_softcapping == 30.0
    x = torch.tensor([[0.0, 30.0, 300.0, -300.0]])
    y = model._softcap(x)
    torch.testing.assert_close(y, 30.0 * torch.tanh(x / 30.0))
    assert y.abs().max() <= 30.0
