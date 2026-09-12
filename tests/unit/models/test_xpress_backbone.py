"""XPress refiner over the DFlash2 (convolution) backbone."""

from typing import Any

import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators import SpeculatorModelConfig, SpeculatorsConfig, VerifierConfig
from speculators.losses import resolve_loss_config
from speculators.models.dflash.model_definitions import Qwen3DFlashDecoderLayer
from speculators.models.dflash2.model_definitions import Qwen3DFlash2DecoderLayer
from speculators.models.xpress import XPressDraftModel, XPressSpeculatorConfig
from speculators.proposals import GreedyTokenProposalConfig


def _tiny_config(**overrides) -> XPressSpeculatorConfig:
    transformer_config = Qwen3Config(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=64,
        _attn_implementation="eager",  # type: ignore[call-arg]
    )
    values: dict[str, Any] = {
        "transformer_layer_config": transformer_config,
        "draft_vocab_size": 64,
        "block_size": 4,
        "aux_hidden_state_layer_ids": [0, 1],
        "mask_token_id": 0,
        "xpress_rank": 8,
        "conv_kernel_size": 2,
        "conv_group_size": 4,
    }
    values.update(overrides)
    return XPressSpeculatorConfig(**values)


def _finite_init(model: torch.nn.Module) -> None:
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.isnan().any():
                torch.nn.init.normal_(parameter, std=0.02)


def _inputs(model: XPressDraftModel, seq_len: int = 32) -> dict[str, torch.Tensor]:
    hidden_size = model.config.transformer_layer_config.hidden_size
    return {
        "hidden_states": torch.randn(1, seq_len, 2 * hidden_size),
        "input_ids": torch.randint(0, model.verifier_vocab_size, (1, seq_len)),
        "loss_mask": torch.ones(1, seq_len),
        "verifier_last_hidden_states": torch.randn(1, seq_len, hidden_size),
        "document_ids": torch.zeros(1, seq_len, dtype=torch.long),
    }


def test_default_backbone_is_plain_dflash():
    """xpress_backbone='dflash' keeps the released-checkpoint layer type and keys."""
    model = XPressDraftModel(_tiny_config())
    assert type(model.layers[0]) is Qwen3DFlashDecoderLayer
    assert not any("attention_conv" in name for name in model.state_dict())


def test_dflash2_backbone_uses_dflash2_layers_with_identity_convs():
    """xpress_backbone='dflash2' builds DFlash2 layers, convs start as identity."""
    model = XPressDraftModel(_tiny_config(xpress_backbone="dflash2"))
    layer = model.layers[0]
    assert isinstance(layer, Qwen3DFlash2DecoderLayer)
    keys = set(model.state_dict())
    assert "layers.0.attention_conv.base_kernel" in keys
    assert "layers.0.mlp_conv.kernel_projection.weight" in keys
    assert "refiner_head.w1.weight" in keys or any(
        k.startswith("refiner_head.") for k in keys
    )
    # DFlash2's reset_convolutions: tap 0 = 1, other taps 0, projection 0.
    base = layer.attention_conv.base_kernel
    assert torch.equal(base[:, 0], torch.ones_like(base[:, 0]))
    assert torch.count_nonzero(base[:, 1:]) == 0
    assert torch.count_nonzero(layer.attention_conv.kernel_projection.weight) == 0


def test_dflash2_backbone_trains_convs_and_refiner():
    """One step must reach the conv, the refiner and the shared backbone."""
    torch.manual_seed(0)
    model = XPressDraftModel(_tiny_config(xpress_backbone="dflash2"))
    _finite_init(model)
    _, loss, _ = model(  # type: ignore[call-arg]
        **_inputs(model),
        max_anchors=4,
        loss_config=resolve_loss_config("kl_div", "eager"),
        consistency_passes=1,
    )
    assert torch.isfinite(loss)
    loss.backward()
    for fragment in ("attention_conv", "mlp_conv", "refiner_head", "self_attn"):
        grads = [
            p.grad
            for n, p in model.named_parameters()
            if fragment in n and p.grad is not None
        ]
        assert grads, f"no gradient reached {fragment}"
        assert all(torch.isfinite(g).all() for g in grads), fragment
        assert any(torch.count_nonzero(g) for g in grads), (
            f"zero gradient for {fragment}"
        )


def test_config_round_trip_keeps_backbone_choice(tmp_path):
    config = _tiny_config(
        xpress_backbone="dflash2",
        sliding_window_non_causal=True,
        speculators_config=SpeculatorsConfig(
            algorithm="xpress",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=3)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path="Qwen/Qwen3-4B", architectures=["Qwen3ForCausalLM"]
            ),
        ),
    )
    config.save_pretrained(tmp_path)
    loaded = SpeculatorModelConfig.from_pretrained(tmp_path)
    assert isinstance(loaded, XPressSpeculatorConfig)
    assert loaded.speculators_model_type == "xpress"
    assert loaded.xpress_backbone == "dflash2"
    assert loaded.conv_group_size == 4
    assert loaded.sliding_window_non_causal is True


def test_selector_is_off_by_default():
    model = XPressDraftModel(_tiny_config())
    assert model.candidate_selector is None
    assert not any(k.startswith("candidate_selector.") for k in model.state_dict())


def test_selector_seed_walks_from_the_anchor():
    """Slot 0 carries the anchor; every later slot is one of its unary top-k."""
    torch.manual_seed(0)
    model = XPressDraftModel(_tiny_config(xpress_selector=True, selector_top_k=4))
    _finite_init(model)
    num_blocks, block, vocab = 3, model.block_size, model.verifier_vocab_size
    hidden = model.config.transformer_layer_config.hidden_size
    logits = torch.randn(num_blocks, block, vocab)
    hidden_blocks = torch.randn(num_blocks, block, hidden)
    anchors = torch.randint(0, vocab, (num_blocks, 1))

    seed = model._selector_seed(logits, hidden_blocks, anchors)  # noqa: SLF001

    assert seed.shape == (num_blocks, block)
    assert torch.equal(seed[:, 0], anchors[:, 0])
    top = logits.topk(4, dim=-1).indices
    assert (seed[:, 1:, None] == top[:, 1:]).any(dim=-1).all()


def test_selector_trains_with_the_refiner_on_the_dflash2_backbone():
    """Full stack: conv backbone + selector + refiner, one step reaches all three."""
    torch.manual_seed(0)
    model = XPressDraftModel(
        _tiny_config(xpress_backbone="dflash2", xpress_selector=True, selector_top_k=4)
    )
    _finite_init(model)
    _, loss, metrics = model(  # type: ignore[call-arg]
        **_inputs(model),
        max_anchors=4,
        loss_config=resolve_loss_config("kl_div", "eager"),
        consistency_passes=1,
    )
    assert torch.isfinite(loss)
    assert "selector_loss" in metrics and torch.isfinite(metrics["selector_loss"])
    loss.backward()
    for fragment in ("attention_conv", "candidate_selector", "refiner_head"):
        grads = [
            p.grad
            for n, p in model.named_parameters()
            if fragment in n and p.grad is not None
        ]
        assert grads, f"no gradient reached {fragment}"
        assert any(torch.count_nonzero(g) for g in grads), (
            f"zero gradient for {fragment}"
        )


def test_selector_loss_alpha_zero_drops_the_selector_term():
    torch.manual_seed(0)
    model = XPressDraftModel(_tiny_config(xpress_selector=True, selector_top_k=4))
    _finite_init(model)
    _, loss, metrics = model(  # type: ignore[call-arg]
        **_inputs(model),
        max_anchors=4,
        loss_config=resolve_loss_config("kl_div", "eager"),
        consistency_passes=0,
        consistency_weight=0.0,
        selector_loss_alpha=0.0,
    )
    assert "selector_loss" not in metrics
    loss.backward()
    codebook = model.candidate_selector.successor_codebook  # type: ignore[union-attr]
    assert codebook.grad is None or torch.count_nonzero(codebook.grad) == 0
