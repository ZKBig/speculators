"""Re-type a converted DFlash checkpoint as an XPress starting point.

`convert_model()` knows nothing about XPress, so a warm start is two steps: convert
the DFlash backbone normally, then run this to switch the config over and stamp the
recipe fields that live in the MODEL CONFIG rather than on the CLI. Keeping it
separate leaves the upstream converter untouched.

    python examples/train/xpress_morph_config.py <converted_dir> [--shift] [--rank R] [--mlp-ratio N]

Idempotent, and safe — meant — to re-run: a checkpoint morphed by an older revision
would otherwise silently miss any field added since.

`--rank` / `--mlp-ratio` size the refiner head. They matter here and nowhere else: a
`--from-pretrained` run reads its architecture from this config, so the CLI flags of
the same name on `scripts/train.py` are ignored on that path. The defaults (256 / 2,
i.e. an MLP hidden of 512) are what the released checkpoints use.

`--shift` selects the DeepSeek/DSpark block convention (`sample_from_anchor=True`,
every slot predicts the next token). The default is fill-in, which is what z-lab
checkpoints are native to: slot 0 carries the known anchor and slots 1..B-1 predict.
"""

import json
import sys
from pathlib import Path

def _opt(name: str, default: int) -> int:
    if name not in sys.argv:
        return default
    return int(sys.argv[sys.argv.index(name) + 1])


argv = sys.argv[1:]
shift = "--shift" in argv
rank = _opt("--rank", 256)
mlp_ratio = _opt("--mlp-ratio", 2)
consumed = {"--shift", "--rank", "--mlp-ratio", str(rank), str(mlp_ratio)}
args = [a for a in argv if a not in consumed and not a.startswith("-")]
if len(args) != 1:
    sys.exit(__doc__)

path = Path(args[0]) / "config.json"
cfg = json.loads(path.read_text())
assert cfg.get("speculators_model_type") in ("dflash", "xpress"), (
    f"expected a converted dflash/xpress checkpoint, "
    f"got {cfg.get('speculators_model_type')}"
)

cfg["speculators_model_type"] = "xpress"
cfg["architectures"] = ["XPressDraftModel"]
cfg["sample_from_anchor"] = shift
cfg["xpress_rank"] = rank
cfg["xpress_mlp_ratio"] = mlp_ratio
cfg.setdefault("num_jacobi_passes", 6)
cfg.setdefault("mask_token_id", 151669)

path.write_text(json.dumps(cfg, indent=2))
print(
    f"morphed {path}: xpress, {'shift' if shift else 'fill-in'}, "
    f"block_size={cfg.get('block_size')}, rank={rank}, "
    f"mlp_hidden={rank * mlp_ratio}"
)
