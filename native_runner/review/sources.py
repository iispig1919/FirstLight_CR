"""Locate reconstructed sequences and the source replays they came from.

A review reads the same model-input cache that imitation learning trains on,
so the battle engine is only needed once, when ``cache_builder`` reconstructs
the matches.  The source Parquet dataset is optional context: it names the
sides, the result, and every action the player really made.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from ..contracts import ActionKind
from ..royaleapi_replay import base_card_key, prepare_collected_replay
from ..training.v4.cache import load_il_cache_index, load_il_cache_manifest, load_il_cache_shard
from ..training.v4.dataset import screen_replay_payload
from ..training.v4.imitation import ILSequenceV4
from .report import MatchContext, PlayerContext


REPLAY_DIRECTORY = "replays"
SELECTED_PART_NAME = "part-000000.parquet"
SIDES = ("team", "opponent")


@dataclass(frozen=True, slots=True)
class CachedReplay:
    """Both owner sequences of one reconstructed match and its fidelity record."""

    replay_tag: str
    metadata: Mapping[str, Any]
    sequences: Mapping[int, ILSequenceV4]


def cache_summary_paths(index_path: str | Path) -> tuple[Path, ...]:
    index_file = Path(index_path)
    shards = load_il_cache_index(index_file).get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError(f"cache index has no shards: {index_file}")
    result = []
    for shard in shards:
        name = shard.get("summary_file") if isinstance(shard, Mapping) else None
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError(f"cache shard has no local summary file: {index_file}")
        result.append(index_file.parent / name)
    return tuple(result)


def list_cached_replays(index_path: str | Path) -> dict[str, dict[str, Any]]:
    """Map every replay the cache builder attempted to its fidelity record.

    Reads shard manifests only.  ``has_sequences`` is false for replays that
    failed before producing a single frame; those cannot be reviewed.
    """

    result: dict[str, dict[str, Any]] = {}
    for summary_path in cache_summary_paths(index_path):
        manifest: Mapping[str, Any] = load_il_cache_manifest(summary_path)
        with_sequences = {str(item["replay_tag"]) for item in manifest["sequences"]}
        for metadata in manifest["results"]:
            tag = str(metadata["replay_tag"])
            if tag in result:
                raise ValueError(f"replay {tag} appears in more than one cache shard")
            result[tag] = {**metadata, "has_sequences": tag in with_sequences}
    return result


def iter_cached_replays(index_path: str | Path, tags: Sequence[str] | None = None) -> Iterator[CachedReplay]:
    """Yield reviewable replays, loading each shard at most once."""

    wanted = None if tags is None else set(tags)
    for summary_path in cache_summary_paths(index_path):
        manifest: Mapping[str, Any] = load_il_cache_manifest(summary_path)
        selected = {str(item["replay_tag"]) for item in manifest["sequences"]}
        if wanted is not None:
            selected &= wanted
        if not selected:
            continue
        shard = load_il_cache_shard(summary_path)
        metadata_by_tag = {str(item["replay_tag"]): item for item in shard.manifest["results"]}
        sequences: dict[str, dict[int, ILSequenceV4]] = {}
        for index, descriptor in enumerate(shard.descriptors):
            if descriptor.replay_tag in selected:
                sequences.setdefault(descriptor.replay_tag, {})[descriptor.owner] = shard.sequence(index)
        for tag, by_owner in sequences.items():
            yield CachedReplay(replay_tag=tag, metadata=metadata_by_tag[tag], sequences=by_owner)


def _replay_dataset(dataset_root: str | Path) -> Any:
    import pyarrow.dataset as arrow_dataset

    replay_root = Path(dataset_root) / REPLAY_DIRECTORY
    if not replay_root.is_dir():
        raise FileNotFoundError(f"missing replay Parquet directory: {replay_root}")
    return arrow_dataset.dataset(replay_root, format="parquet")


def load_replay_payloads(dataset_root: str | Path, tags: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Read the source payloads of the given replay tags, skipping absent ones."""

    import pyarrow.dataset as arrow_dataset

    table = _replay_dataset(dataset_root).to_table(
        columns=["replay_tag", "payload_json"], filter=arrow_dataset.field("replay_tag").isin(list(tags))
    )
    return {
        str(tag): json.loads(str(payload))
        for tag, payload in zip(
            table.column("replay_tag").to_pylist(), table.column("payload_json").to_pylist(), strict=True
        )
    }


def match_context(payload: Mapping[str, Any]) -> MatchContext:
    """Bind the source replay's sides, result and actions to engine owners."""

    replay = prepare_collected_replay(payload).replay
    events = payload["events"]
    side_by_owner: dict[int, str] = {}
    source_actions: dict[int, list[tuple[int, int | None]]] = {0: [], 1: []}
    for operation in replay.operations:
        for action in operation.actions:
            if action.kind not in {ActionKind.PLAY_CARD, ActionKind.ACTIVATE_ABILITY}:
                continue
            side_by_owner.setdefault(action.owner, str(events[int(action.metadata["source_event_index"])]["side"]))
            source_actions[action.owner].append(
                (
                    int(action.metadata["source_command_tick"]),
                    action.card_id if action.kind is ActionKind.PLAY_CARD else None,
                )
            )
    for owner in (0, 1):
        if owner not in side_by_owner and 1 - owner in side_by_owner:
            side_by_owner[owner] = SIDES[1 - SIDES.index(side_by_owner[1 - owner])]

    terminal = replay.terminal
    winner = terminal.get("winner")
    crowns = {
        int(player["owner"]): int(player["crowns"])
        for player in terminal.get("players", ())
        if isinstance(player, Mapping) and type(player.get("crowns")) is int
    }
    battle = payload.get("battle", {})
    players: dict[int, PlayerContext] = {}
    for owner in (0, 1):
        side = side_by_owner.get(owner)
        if winner in (0, 1):
            result = "win" if winner == owner else "loss"
        else:
            result = "draw" if str(terminal.get("source_result") or "").casefold() == "draw" else None
        leaked = None
        if side is not None:
            source_players = battle.get(side, {}).get("players", ())
            raw = source_players[0].get("elixir_leaked") if source_players else None
            leaked = float(raw) if isinstance(raw, (int, float)) else None
        players[owner] = PlayerContext(
            side=side,
            result=result,
            crowns=crowns.get(owner),
            source_elixir_leaked=leaked,
            source_actions=tuple(sorted(source_actions[owner], key=lambda item: item[0])),
        )
    duration = payload.get("replay", {}).get("duration", {}).get("timeline_seconds")
    return MatchContext(
        game_mode=battle.get("game_mode"),
        duration_seconds=float(duration) if isinstance(duration, (int, float)) else None,
        players=players,
    )


def _deck_keys(payload: Mapping[str, Any], side: str) -> set[str]:
    players = payload.get("battle", {}).get(side, {}).get("players", ())
    if not players:
        return set()
    return {base_card_key(card.get("card_key")) for card in players[0].get("deck", ())}


def _reconstructable(payload: Mapping[str, Any]) -> bool:
    """Apply the cache builder's own engine-free acceptance checks."""

    if not screen_replay_payload(payload).accepted:
        return False
    try:
        prepare_collected_replay(payload)
    except Exception:
        return False
    return True


def select_replays(
    dataset_root: str | Path,
    output_root: str | Path,
    *,
    tags: Sequence[str] = (),
    deck_cards: Sequence[str] = (),
    limit: int | None = None,
) -> list[str]:
    """Copy chosen replays into a small dataset the cache builder can convert.

    ``tags`` picks exact replays.  ``deck_cards`` keeps matches where one
    player's deck holds every listed card key (forms such as ``-ev1`` are
    ignored).  Replays the cache builder would reject before touching the
    engine are left out, so every selected row is worth an engine slot.
    """

    import pyarrow as arrow
    import pyarrow.parquet as parquet

    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    destination = Path(output_root) / REPLAY_DIRECTORY
    if destination.is_dir() and any(destination.iterdir()):
        raise FileExistsError(f"refusing to add replays to an existing dataset: {destination}")
    wanted_tags = set(tags)
    wanted_cards = {base_card_key(card) for card in deck_cards}
    dataset = _replay_dataset(dataset_root)
    batches = []
    selected: list[str] = []
    for batch in dataset.to_batches(batch_size=256):
        keep = []
        for tag, raw in zip(
            batch.column("replay_tag").to_pylist(), batch.column("payload_json").to_pylist(), strict=True
        ):
            accepted = False
            if (limit is None or len(selected) < limit) and (not wanted_tags or tag in wanted_tags):
                payload = json.loads(str(raw))
                accepted = (
                    not wanted_cards or any(wanted_cards <= _deck_keys(payload, side) for side in SIDES)
                ) and _reconstructable(payload)
            if accepted:
                selected.append(str(tag))
            keep.append(accepted)
        if any(keep):
            batches.append(batch.filter(arrow.array(keep)))
        if limit is not None and len(selected) >= limit:
            break
    if not selected:
        raise ValueError("no reconstructable replay matched the selection")
    destination.mkdir(parents=True, exist_ok=True)
    parquet.write_table(arrow.Table.from_batches(batches), destination / SELECTED_PART_NAME)
    return selected
