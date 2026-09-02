#!/usr/bin/env python3
# ruff: noqa: T201, PLC0415, PTH123
"""Normalize a regenerated-response corpus into a uniform conversations dataset.

The published collections carry per-row generation metadata, and the shape of it
differs between files -- some rows record an `error`, some do not. `datasets`
infers its schema from the first file and then refuses the rest:

    TypeError: Couldn't cast array of type struct<..., error: string> to {...}

prepare_data.py only reads `conversations`, so the fix is to keep that (plus an
id) and drop everything else. Rows whose metadata records an error are dropped
too: a failed generation is not target-model output, which is the one property
the training data has to have.

    python examples/train/normalize_regen_corpus.py \
      --src hf:inference-optimization/Qwen3-8B-Regenerated-Collection \
      --out /gpfs/zwang33/dflash2/data/regen_clean
"""

import argparse
import json
import os
import pathlib
import sys
import urllib.request


def _iter_files(
    src: str, include: list[str] | None, exclude: list[str] | None
) -> list[pathlib.Path]:
    if src.startswith("hf:"):
        from huggingface_hub import snapshot_download

        # Filter at download time, not after: these files are hundreds of MB each,
        # and one of them (nemotron_math) is published with an LFS size that
        # disagrees with the bytes served, so every downloader rejects it on the
        # consistency check. Excluding it is the only way past that.
        allow = [f"{stem}.jsonl" for stem in include] if include else ["*.jsonl"]
        ignore = [f"{stem}.jsonl" for stem in exclude] if exclude else None
        if ignore:
            print(f"[src] excluding {', '.join(exclude or [])}")
        local = snapshot_download(
            repo_id=src[3:],
            repo_type="dataset",
            allow_patterns=allow,
            ignore_patterns=ignore,
        )
        print(f"[src] {src} -> {local}")
        root = pathlib.Path(local)
        extra = []
        if exclude and os.environ.get("DF2_NORMALIZE_FETCH_UNVERIFIED") == "1":
            raw_dir = pathlib.Path(
                os.environ.get("DF2_NORMALIZE_RAW_DIR") or (root.parent / "unverified")
            )
            extra = [
                _fetch_unverified(src[3:], f"{stem}.jsonl", raw_dir / f"{stem}.jsonl")
                for stem in exclude
            ]
        files = sorted(p for p in root.rglob("*.jsonl") if p.is_file())
        if not files and not extra:
            raise SystemExit(f"no .jsonl under {root}")
        return files + extra
    else:
        root = pathlib.Path(src)
    files = sorted(p for p in root.rglob("*.jsonl") if p.is_file())
    if not files:
        raise SystemExit(f"no .jsonl under {root}")
    return files


def _fetch_unverified(repo: str, filename: str, dest: pathlib.Path) -> pathlib.Path:
    """Stream one file straight from the CDN, skipping the size check.

    huggingface_hub compares the bytes it received against the size recorded in
    the LFS pointer and refuses the download when they disagree. That guard is
    right in general, but nemotron_math_Qwen3-8B.jsonl is published with a stale
    pointer -- the same 44 KB discrepancy on every attempt and on both download
    backends -- so the only way to read that file is to skip the check.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[raw] {dest} already present ({dest.stat().st_size} bytes)")
        return dest
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{filename}"
    print(f"[raw] fetching {filename} without the size check")
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:  # noqa: S310
        while chunk := r.read(1 << 22):
            f.write(chunk)
    print(f"[raw] {dest} ({dest.stat().st_size} bytes)")
    return dest


def _error_of(row: dict) -> str | None:
    """Find a non-empty `error` anywhere in the row's metadata."""
    for value in row.values():
        if isinstance(value, dict) and value.get("error"):
            return str(value["error"])
    return row.get("error") or None


def main() -> None:
    # Env defaults so this can run as a job's mainProgram, which passes no args.
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src",
        default=os.environ.get("DF2_NORMALIZE_SRC"),
        help='"hf:org/dataset" or a local dir (env DF2_NORMALIZE_SRC)',
    )
    ap.add_argument(
        "--out",
        default=os.environ.get("DF2_NORMALIZE_OUT"),
        help="output directory (env DF2_NORMALIZE_OUT)",
    )
    ap.add_argument(
        "--exclude",
        nargs="*",
        default=(os.environ.get("DF2_NORMALIZE_EXCLUDE") or "").split() or None,
        help="skip these stems (env DF2_NORMALIZE_EXCLUDE, space sep)",
    )
    ap.add_argument(
        "--include",
        nargs="*",
        default=(os.environ.get("DF2_NORMALIZE_INCLUDE") or "").split() or None,
        help="only these stems (env DF2_NORMALIZE_INCLUDE, space sep)",
    )
    args = ap.parse_args()
    if not args.src or not args.out:
        raise SystemExit("need --src/--out (or DF2_NORMALIZE_SRC / DF2_NORMALIZE_OUT)")

    # One writer. As a mainProgram this runs under torchrun, and eight ranks
    # writing the same files would interleave lines into corrupt JSON.
    if int(os.environ.get("RANK") or 0) != 0:
        print(f"[rank {os.environ.get('RANK')}] not the writer; exiting")
        sys.exit(0)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kept_total = dropped_total = 0

    for path in _iter_files(args.src, args.include, args.exclude):
        if args.include and path.stem not in args.include:
            print(f"[skip] {path.name}")
            continue
        if (
            args.exclude
            and path.stem in args.exclude
            and os.environ.get("DF2_NORMALIZE_FETCH_UNVERIFIED") != "1"
        ):
            print(f"[skip] {path.name} (excluded)")
            continue
        kept = dropped = 0
        with (
            open(path, encoding="utf-8") as fin,
            open(out / path.name, "w", encoding="utf-8") as fout,
        ):
            for raw in fin:
                text = raw.strip()
                if not text:
                    continue
                row = json.loads(text)
                convs = row.get("conversations")
                if not convs:
                    dropped += 1
                    continue
                if _error_of(row):
                    dropped += 1
                    continue
                fout.write(
                    json.dumps(
                        {
                            "id": row.get("id", f"{path.stem}_{kept}"),
                            "conversations": convs,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                kept += 1
        print(f"[ok] {path.name:44s} kept {kept:7d}  dropped {dropped}")
        kept_total += kept
        dropped_total += dropped

    print(f"\nwrote {out}  ({kept_total} rows, {dropped_total} dropped)")
    print("point --data / DF2_TRAIN_SRC at that directory")


if __name__ == "__main__":
    main()
