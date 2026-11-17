"""Review reconstructed matches with a V4 policy.

    python -m native_runner.review select  DATASET_ROOT OUTPUT_ROOT [--tag TAG] [--deck-card KEY] [--limit N]
    python -m native_runner.review list    CACHE_INDEX
    python -m native_runner.review analyze CACHE_INDEX --output DIR [--tag TAG] [--dataset-root DIR]

``select`` and ``analyze`` never touch the battle engine.  Between them, build
the cache from the selected dataset with ``native_runner.training.v4.cache_builder``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

from ..paths import REPOSITORY_ROOT
from ..training.v4.policy_session import load_policy_v4
from .report import MatchContext, build_review, fidelity_report, render_summary
from .scoring import score_sequence
from .sources import iter_cached_replays, list_cached_replays, load_replay_payloads, match_context, select_replays


DEFAULT_CHECKPOINT = REPOSITORY_ROOT / "checkpoints" / "General" / "checkpoint-step-00000460.pt"


def _select(args: argparse.Namespace) -> int:
    selected = select_replays(
        args.dataset_root, args.output_root, tags=args.tag, deck_cards=args.deck_card, limit=args.limit
    )
    missing = sorted(set(args.tag) - set(selected))
    print(json.dumps({"output_root": str(args.output_root), "selected": selected, "skipped_tags": missing}, indent=2))
    return 1 if missing else 0


def _list(args: argparse.Namespace) -> int:
    for tag, metadata in list_cached_replays(args.cache_index).items():
        fidelity = fidelity_report(metadata)
        frames = metadata.get("retained_frame_count", 0) if metadata["has_sequences"] else 0
        reason = f"  {fidelity['failure_reason']}" if fidelity["failure_reason"] else ""
        print(f"{tag}  {fidelity['verdict']:<10} frames={frames}{reason}")
    return 0


def _match_context(payload: Mapping[str, Any] | None, replay_tag: str) -> MatchContext | None:
    if payload is None:
        return None
    try:
        return match_context(payload)
    except Exception as error:
        print(f"warning: no source context for {replay_tag}: {type(error).__name__}: {error}", file=sys.stderr)
        return None


def _analyze(args: argparse.Namespace) -> int:
    cached = list_cached_replays(args.cache_index)
    requested = list(args.tag) or [tag for tag, metadata in cached.items() if metadata["has_sequences"]]
    unavailable = []
    for tag in requested:
        if tag not in cached:
            unavailable.append(f"{tag}: not in this cache")
        elif not cached[tag]["has_sequences"]:
            unavailable.append(f"{tag}: reconstruction produced no frames ({cached[tag].get('failure_reason')})")
    reviewable = [tag for tag in requested if tag in cached and cached[tag]["has_sequences"]]
    for message in unavailable:
        print(f"无法复盘 {message}", file=sys.stderr)
    if not reviewable:
        print("缓存里没有可复盘的对局", file=sys.stderr)
        return 1
    payloads = load_replay_payloads(args.dataset_root, reviewable) if args.dataset_root else {}

    loaded = load_policy_v4(args.checkpoint, device=args.device)
    model = {
        "checkpoint_file": loaded.checkpoint_path.name,
        "checkpoint_id": loaded.checkpoint_id,
        "checkpoint_sha256": loaded.checkpoint_sha256,
        "update_step": loaded.update_step,
        "training_stage": loaded.training_stage,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for replay in iter_cached_replays(args.cache_index, reviewable):
        started = time.perf_counter()
        review = build_review(
            replay_tag=replay.replay_tag,
            model=model,
            fidelity=replay.metadata,
            frames_by_owner={
                owner: score_sequence(loaded.model, sequence, device=args.device)
                for owner, sequence in replay.sequences.items()
            },
            match=_match_context(payloads.get(replay.replay_tag), replay.replay_tag),
        )
        destination = args.output / f"{replay.replay_tag}.review.json"
        destination.write_text(json.dumps(review, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        print(render_summary(review))
        print(f"\n已写入 {destination}（用时 {time.perf_counter() - started:.1f} 秒）\n", flush=True)
    return 1 if unavailable else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m native_runner.review", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    select = commands.add_parser("select", help="copy chosen replays into a small dataset for the cache builder")
    select.add_argument("dataset_root", type=Path)
    select.add_argument("output_root", type=Path)
    select.add_argument("--tag", action="append", default=[], help="exact replay tag; repeatable")
    select.add_argument("--deck-card", action="append", default=[], help="card key one deck must hold; repeatable")
    select.add_argument("--limit", type=int, default=None, help="stop after this many replays")
    select.set_defaults(run=_select)

    listing = commands.add_parser("list", help="show every replay in a cache and how faithfully it reconstructed")
    listing.add_argument("cache_index", type=Path)
    listing.set_defaults(run=_list)

    analyze = commands.add_parser("analyze", help="score cached replays and write one review JSON per match")
    analyze.add_argument("cache_index", type=Path)
    analyze.add_argument("--output", type=Path, required=True)
    analyze.add_argument("--tag", action="append", default=[], help="replay tag to review; default is every replay")
    analyze.add_argument("--dataset-root", type=Path, default=None, help="source dataset, for sides and results")
    analyze.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    analyze.add_argument("--device", default="cpu")
    analyze.set_defaults(run=_analyze)

    args = parser.parse_args(argv)
    return int(args.run(args))


if __name__ == "__main__":
    raise SystemExit(main())
