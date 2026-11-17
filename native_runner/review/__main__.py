"""Review reconstructed matches with a V4 policy.

    python -m native_runner.review select  DATASET_ROOT OUTPUT_ROOT [--tag TAG] [--deck-card KEY] [--limit N]
    python -m native_runner.review list    CACHE_INDEX
    python -m native_runner.review analyze CACHE_INDEX --output DIR [--tag TAG] [--dataset-root DIR]
    python -m native_runner.review page    REVIEW_JSON [--output HTML]
    python -m native_runner.review explain REVIEW_JSON [--player OWNER] [--effort LEVEL] [--show-request]
    python -m native_runner.review summary REVIEW_DIR

``select`` and ``analyze`` never touch the battle engine.  Between them, build
the cache from the selected dataset with ``native_runner.training.v4.cache_builder``.
``analyze`` writes a review JSON and its report page per match; ``page`` redraws
the page from an existing JSON; ``explain`` asks Claude for written commentary
and adds it to both; ``summary`` tabulates every review in a directory.
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
from .page import render_page
from .report import MatchContext, build_review, fidelity_report, render_review_table, render_summary
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
        page = destination.with_suffix(".html")
        page.write_text(render_page(review), encoding="utf-8")
        print(render_summary(review))
        print(f"\n已写入 {destination} 和 {page.name}（用时 {time.perf_counter() - started:.1f} 秒）\n", flush=True)
    return 1 if unavailable else 0


def _page(args: argparse.Namespace) -> int:
    review = json.loads(args.review_json.read_text(encoding="utf-8"))
    destination = args.output or args.review_json.with_suffix(".html")
    destination.write_text(render_page(review), encoding="utf-8")
    print(f"已写入 {destination}")
    return 0


def _summary(args: argparse.Namespace) -> int:
    paths = sorted(args.review_dir.glob("*.review.json"))
    if not paths:
        print(f"{args.review_dir} 里没有 *.review.json", file=sys.stderr)
        return 1
    print(render_review_table([json.loads(path.read_text(encoding="utf-8")) for path in paths]))
    return 0


def _explain(args: argparse.Namespace) -> int:
    # The Claude SDK is only needed here, so the other commands work without it.
    import anthropic

    from .commentary import CommentaryError, build_brief, commentary_request, render_commentary, request_commentary

    review = json.loads(args.review_json.read_text(encoding="utf-8"))
    owners = args.player or [int(player["owner"]) for player in review["players"]]
    if args.show_request:
        for owner in owners:
            request = commentary_request(build_brief(review, owner), effort=args.effort)
            print(json.dumps(request, ensure_ascii=False, indent=2))
        return 0

    page = args.review_json.with_suffix(".html")
    try:
        client = anthropic.Anthropic()
        for owner in owners:
            commentary = request_commentary(build_brief(review, owner), client=client, effort=args.effort)
            next(player for player in review["players"] if int(player["owner"]) == owner)["commentary"] = commentary
            # Save after every player so a later failure keeps the commentary already paid for.
            args.review_json.write_text(json.dumps(review, ensure_ascii=False, allow_nan=False), encoding="utf-8")
            page.write_text(render_page(review), encoding="utf-8")
            usage = commentary["usage"]
            print(
                f"玩家 {owner} 的讲解（{commentary['model']}，"
                f"输入 {usage['input_tokens']} / 输出 {usage['output_tokens']} tokens）\n"
            )
            print(render_commentary(commentary) + "\n", flush=True)
    except TypeError as error:
        # The SDK reports missing credentials as a TypeError when the request is built.
        if "authentication" not in str(error):
            raise
        print("没有找到 Claude API 凭据。请先设置环境变量 ANTHROPIC_API_KEY。", file=sys.stderr)
        return 1
    except anthropic.AuthenticationError:
        print("Claude API 拒绝了这个密钥，请检查 ANTHROPIC_API_KEY。", file=sys.stderr)
        return 1
    except anthropic.RateLimitError:
        print("Claude API 限流了，稍后再试。", file=sys.stderr)
        return 1
    except anthropic.APIStatusError as error:
        print(f"Claude API 返回错误 {error.status_code}：{error.message}", file=sys.stderr)
        return 1
    except anthropic.APIConnectionError as error:
        print(f"连接不上 Claude API：{error}", file=sys.stderr)
        return 1
    except CommentaryError as error:
        print(f"没有得到可用的讲解：{error}", file=sys.stderr)
        return 1
    print(f"已写入 {args.review_json} 和 {page.name}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    # A redirected Windows console may not encode every character of the Chinese summary.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")
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

    page = commands.add_parser("page", help="redraw the report page from an existing review JSON")
    page.add_argument("review_json", type=Path)
    page.add_argument("--output", type=Path, default=None, help="default: next to the JSON, with .html")
    page.set_defaults(run=_page)

    explain = commands.add_parser("explain", help="ask Claude for written commentary and add it to the review")
    explain.add_argument("review_json", type=Path)
    explain.add_argument("--player", type=int, action="append", default=[], help="owner seat; default is both")
    explain.add_argument("--effort", default="high", choices=("low", "medium", "high", "xhigh", "max"))
    explain.add_argument("--show-request", action="store_true", help="print what would be sent; call nothing")
    explain.set_defaults(run=_explain)

    summary = commands.add_parser("summary", help="tabulate every review JSON in a directory")
    summary.add_argument("review_dir", type=Path)
    summary.set_defaults(run=_summary)

    args = parser.parse_args(argv)
    return int(args.run(args))


if __name__ == "__main__":
    raise SystemExit(main())
