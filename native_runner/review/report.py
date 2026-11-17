"""Turn frame scores into a review document and a readable summary.

The document is plain JSON so a later report page can draw it directly.  Every
number here is the policy's opinion of a reconstructed match, not ground truth:
``value`` is the model's shaped-return estimate (higher is better for that
player, but it is not a calibrated win probability), and probabilities say how
typical a choice is for the policy, not whether it was correct.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from statistics import median
from typing import Any, Mapping, Sequence

from ..card_names import CARD_NAMES
from ..training.v4.expert import NATIVE_TICK_MS, POLICY_DECISION_TICKS
from .scoring import CardOption, FrameScore, ModelChoice, PlayedAction, TowerHitpoints


REVIEW_SCHEMA = "firstlight-replay-review.v1"
TICKS_PER_SECOND = 1000 // NATIVE_TICK_MS
DECISION_SECONDS = POLICY_DECISION_TICKS / TICKS_PER_SECOND
ELIXIR_CAP = 10.0
ELIXIR_CAP_TOLERANCE = 0.01
ELIXIR_SECONDS_PER_POINT = 2.8
SWING_WINDOW_FRAMES = 20
SWING_LIMIT = 3
# Value is roughly win (+1) to loss (-1); smaller moves are noise.
SWING_MINIMUM_CHANGE = 0.05
SWING_CONTEXT_TICKS = 2 * TICKS_PER_SECOND
MISSED_MINIMUM_FRAMES = 4
MISSED_LIMIT = 5
DISAGREEMENT_LIMIT = 5
PLACEMENT_LIMIT = 3
PLACEMENT_MINIMUM_DISTANCE = 3.0
ACT_THRESHOLD = 0.5

CONVENTIONS = {
    "tick_rate_hz": TICKS_PER_SECOND,
    "decision_interval_ticks": POLICY_DECISION_TICKS,
    "cell": "[x, y] in the reviewed player's own view: x 0..17 left to right, "
    "y 0..31 from the player's king tower (0) to the enemy king tower (31)",
    "towers": "[king, princess_left, princess_right] hitpoints, left/right in the reviewed player's view",
    "value": "policy value estimate from that player's view: the expected discounted match result "
    "(win +1, loss -1) plus tower shaping; zero-sum by construction, not a calibrated win probability",
    "probabilities": "the policy's own distribution at temperature 1; act_probability comes from a gate "
    "trained with up-weighted ACT labels, so read it as a relative score",
}


@dataclass(frozen=True, slots=True)
class PlayerContext:
    """Facts about one owner that come from the source replay, if available."""

    side: str | None = None
    result: str | None = None
    crowns: int | None = None
    source_elixir_leaked: float | None = None
    source_actions: tuple[tuple[int, int | None], ...] | None = None


@dataclass(frozen=True, slots=True)
class MatchContext:
    game_mode: str | None = None
    duration_seconds: float | None = None
    players: Mapping[int, PlayerContext] = field(default_factory=dict)


def clock(tick: int) -> str:
    seconds = tick // TICKS_PER_SECOND
    return f"{seconds // 60}:{seconds % 60:02d}"


def card_label(card_id: int) -> dict[str, Any]:
    name, english = CARD_NAMES.get(card_id, (str(card_id), str(card_id)))
    return {"id": card_id, "name": name, "name_en": english}


def zone_label(cell: Sequence[int]) -> str:
    x, y = cell
    if y <= 6:
        depth = "己方后场"
    elif y <= 14:
        depth = "己方前场"
    elif y <= 16:
        depth = "河道"
    elif y <= 24:
        depth = "对方前场"
    else:
        depth = "对方后场"
    return f"{depth}·{'左路' if x <= 8 else '右路'}"


def _rounded(value: float | None, digits: int = 4) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(value, digits)


def _towers(towers: TowerHitpoints) -> list[int]:
    return [round(towers.king), round(towers.princess_left), round(towers.princess_right)]


def _option(option: CardOption) -> dict[str, Any]:
    return {
        "card": card_label(option.card_id),
        "kind": option.kind,
        "form": option.form,
        "cost": _rounded(option.cost, 2),
        "probability": _rounded(option.probability),
    }


def _model_choice(choice: ModelChoice | None, *, played: PlayedAction | None = None) -> dict[str, Any] | None:
    if choice is None:
        return None
    result = {
        "card": card_label(choice.card_id),
        "kind": choice.kind,
        "form": choice.form,
        "cost": _rounded(choice.cost, 2),
        "card_probability": _rounded(choice.card_probability),
        "cell": list(choice.cell) if choice.cell is not None else None,
        "zone": zone_label(choice.cell) if choice.cell is not None else None,
        "cell_probability": _rounded(choice.cell_probability),
    }
    if played is not None:
        result["same_card"] = (choice.kind, choice.card_id) == (played.kind, played.card_id)
    return result


def _value_after(frames: Sequence[FrameScore], index: int) -> float:
    """Value one swing window later, or at the last trusted frame."""

    last = index
    for candidate in range(index + 1, min(len(frames), index + SWING_WINDOW_FRAMES + 1)):
        if not frames[candidate].valid:
            break
        last = candidate
    return frames[last].value


def player_actions(frames: Sequence[FrameScore]) -> list[dict[str, Any]]:
    """Flatten every scored action with the context of its decision frame."""

    result: list[dict[str, Any]] = []
    for frame in frames:
        if not frame.valid:
            continue
        for step, action in enumerate(frame.played):
            distance = None
            if action.cell is not None and action.model_cell is not None:
                distance = math.dist(action.cell, action.model_cell)
            value_after = _value_after(frames, frame.index)
            result.append(
                {
                    "index": len(result),
                    "frame": frame.index,
                    "tick": action.tick,
                    "time": clock(action.tick),
                    "kind": action.kind,
                    "card": card_label(action.card_id),
                    "form": action.form,
                    "cost": _rounded(action.cost, 2),
                    "cell": list(action.cell) if action.cell is not None else None,
                    "zone": zone_label(action.cell) if action.cell is not None else None,
                    "elixir_before": _rounded(frame.elixir, 2),
                    "value_before": _rounded(frame.value),
                    "value_after": _rounded(value_after),
                    "value_change": _rounded(value_after - frame.value),
                    "act_probability": _rounded(frame.act_probability),
                    "card_probability": _rounded(action.card_probability),
                    "cell_probability": _rounded(action.cell_probability, 6),
                    "cell_rank": action.cell_rank,
                    "legal_cell_count": action.legal_cell_count,
                    "model_cell": list(action.model_cell) if action.model_cell is not None else None,
                    "model_cell_probability": _rounded(action.model_cell_probability),
                    "cell_distance": _rounded(distance, 2),
                    "top_cells": [[cell.x, cell.y, _rounded(cell.probability)] for cell in action.top_cells],
                    # Options and the policy's own pick describe the frame before
                    # anything was played, so they belong to the first action only.
                    "model_choice": _model_choice(frame.model_choice, played=action) if step == 0 else None,
                    "options": [_option(option) for option in frame.options] if step == 0 else [],
                }
            )
    return result


def estimate_leaked_elixir(frames: Sequence[FrameScore]) -> tuple[float, float]:
    """Return ``(leaked elixir, seconds at the cap)`` over trusted frames."""

    leaked = 0.0
    seconds = 0.0
    for frame in frames:
        if frame.valid and frame.elixir >= ELIXIR_CAP - ELIXIR_CAP_TOLERANCE:
            seconds += DECISION_SECONDS
            leaked += DECISION_SECONDS * frame.elixir_multiplier / ELIXIR_SECONDS_PER_POINT
    return leaked, seconds


def find_missed_windows(frames: Sequence[FrameScore], *, limit: int = MISSED_LIMIT) -> list[dict[str, Any]]:
    """Stretches where the player waited while the policy wanted to act."""

    windows: list[list[FrameScore]] = []
    current: list[FrameScore] = []
    for frame in frames:
        wanted = (
            frame.valid
            and frame.label_usable
            and not frame.played
            and frame.model_choice is not None
            and frame.act_probability > ACT_THRESHOLD
        )
        if wanted:
            current.append(frame)
            continue
        if current:
            windows.append(current)
            current = []
    if current:
        windows.append(current)

    result = []
    for window in windows:
        if len(window) < MISSED_MINIMUM_FRAMES:
            continue
        peak = max(window, key=lambda frame: frame.act_probability)
        end_tick = window[-1].tick + POLICY_DECISION_TICKS
        result.append(
            {
                "start_tick": window[0].tick,
                "end_tick": end_tick,
                "time": f"{clock(window[0].tick)}–{clock(end_tick)}",
                "duration_seconds": _rounded(len(window) * DECISION_SECONDS, 2),
                "peak_tick": peak.tick,
                "peak_act_probability": _rounded(peak.act_probability),
                "elixir_at_peak": _rounded(peak.elixir, 2),
                "model_choice": _model_choice(peak.model_choice),
            }
        )
    result.sort(key=lambda item: (-item["duration_seconds"] * item["peak_act_probability"], item["start_tick"]))
    return sorted(result[:limit], key=lambda item: item["start_tick"])


def find_swings(
    frames: Sequence[FrameScore], actions: Sequence[Mapping[str, Any]], *, limit: int = SWING_LIMIT
) -> list[dict[str, Any]]:
    """Largest non-overlapping value changes over a five-second window."""

    trusted = [frame for frame in frames if frame.valid]
    window = min(SWING_WINDOW_FRAMES, len(trusted) - 1)
    if window <= 0:
        return []
    changes = [(trusted[start + window].value - trusted[start].value, start) for start in range(len(trusted) - window)]
    picked: list[tuple[float, int]] = []
    for direction in (1.0, -1.0):
        chosen: list[tuple[float, int]] = []
        for change, start in sorted(changes, key=lambda item: direction * item[0]):
            if direction * change > -SWING_MINIMUM_CHANGE or len(chosen) >= limit:
                break
            if all(abs(start - other) >= window for _change, other in chosen):
                chosen.append((change, start))
        picked.extend(chosen)

    result = []
    for change, start in sorted(picked, key=lambda item: item[1]):
        first = trusted[start]
        last = trusted[start + window]
        result.append(
            {
                "start_tick": first.tick,
                "end_tick": last.tick,
                "time": f"{clock(first.tick)}–{clock(last.tick)}",
                "value_start": _rounded(first.value),
                "value_end": _rounded(last.value),
                "value_change": _rounded(change),
                "own_actions": [
                    int(action["index"])
                    for action in actions
                    if first.tick - SWING_CONTEXT_TICKS <= int(action["tick"]) <= last.tick
                ],
            }
        )
    return result


def _unscored_actions(
    source_actions: Sequence[tuple[int, int | None]] | None, actions: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]] | None:
    """Source actions the reconstruction could not turn into a scored label."""

    if source_actions is None:
        return None
    scored: dict[int, list[int]] = {}
    for action in actions:
        scored.setdefault(int(action["tick"]), []).append(int(action["card"]["id"]))
    result = []
    # Abilities carry no card in the source, so let card plays claim their match first.
    for tick, card_id in sorted(source_actions, key=lambda item: (item[0], item[1] is None)):
        remaining = scored.get(tick, [])
        match = card_id if card_id in remaining else (remaining[0] if card_id is None and remaining else None)
        if match is not None:
            remaining.remove(match)
            continue
        result.append({"tick": tick, "time": clock(tick), "card": card_label(card_id) if card_id is not None else None})
    return result


def _summary(
    frames: Sequence[FrameScore], actions: Sequence[Mapping[str, Any]], context: PlayerContext
) -> dict[str, Any]:
    trusted = [frame for frame in frames if frame.valid]
    leaked, seconds_at_cap = estimate_leaked_elixir(frames)
    first_actions = [action for action in actions if action["model_choice"] is not None]
    acted_frames = [frame for frame in trusted if frame.played]
    waited_frames = [
        frame for frame in trusted if frame.label_usable and not frame.played and frame.model_choice is not None
    ]
    distances = [action["cell_distance"] for action in actions if action["cell_distance"] is not None]

    def share(count: int, total: int) -> float | None:
        return _rounded(count / total) if total else None

    low = min(trusted, key=lambda frame: frame.value, default=None)
    high = max(trusted, key=lambda frame: frame.value, default=None)
    return {
        "frames": len(frames),
        "trusted_frames": len(trusted),
        "unusable_label_frames": sum(not frame.label_usable for frame in trusted),
        "actions": len(actions),
        "value": {
            "start": _rounded(trusted[0].value) if trusted else None,
            "end": _rounded(trusted[-1].value) if trusted else None,
            "min": _rounded(low.value) if low else None,
            "min_tick": low.tick if low else None,
            "max": _rounded(high.value) if high else None,
            "max_tick": high.tick if high else None,
        },
        "elixir": {
            "mean": _rounded(sum(frame.elixir for frame in trusted) / len(trusted), 2) if trusted else None,
            "leaked_estimate": _rounded(leaked, 2),
            "seconds_at_cap": _rounded(seconds_at_cap, 2),
            "leaked_reported_by_source": context.source_elixir_leaked,
        },
        "agreement": {
            "act_when_player_acted": share(
                sum(frame.act_probability > ACT_THRESHOLD for frame in acted_frames), len(acted_frames)
            ),
            "wait_when_player_waited": share(
                sum(frame.act_probability <= ACT_THRESHOLD for frame in waited_frames), len(waited_frames)
            ),
            "same_card": share(
                sum(action["model_choice"]["same_card"] for action in first_actions), len(first_actions)
            ),
            "mean_card_probability": (
                _rounded(sum(action["card_probability"] for action in actions) / len(actions)) if actions else None
            ),
            "median_cell_distance": _rounded(median(distances), 2) if distances else None,
        },
        "towers_end": {
            "own": _towers(trusted[-1].own_towers) if trusted else None,
            "enemy": _towers(trusted[-1].enemy_towers) if trusted else None,
        },
    }


def build_player_review(
    owner: int, frames: Sequence[FrameScore], *, context: PlayerContext | None = None
) -> dict[str, Any]:
    actual = context or PlayerContext()
    actions = player_actions(frames)
    return {
        "owner": owner,
        "side": actual.side,
        "result": actual.result,
        "crowns": actual.crowns,
        "summary": _summary(frames, actions, actual),
        "actions": actions,
        "unscored_source_actions": _unscored_actions(actual.source_actions, actions),
        "missed_windows": find_missed_windows(frames),
        "swings": find_swings(frames, actions),
        "timeline": {
            "tick": [frame.tick for frame in frames],
            "valid": [frame.valid for frame in frames],
            "value": [_rounded(frame.value) for frame in frames],
            "realized_return": [_rounded(frame.realized_return) for frame in frames],
            "act_probability": [_rounded(frame.act_probability) for frame in frames],
            "elixir": [_rounded(frame.elixir, 2) for frame in frames],
            "remaining_seconds": [_rounded(frame.remaining_seconds, 2) for frame in frames],
            "own_towers": [_towers(frame.own_towers) for frame in frames],
            "enemy_towers": [_towers(frame.enemy_towers) for frame in frames],
        },
    }


def fidelity_report(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize how faithfully the engine reproduced the source match.

    ``faithful``: the whole timeline replayed and the winner and crowns match
    the source.  Single actions the engine rejected are still possible; they
    show up as unscored source actions.
    ``unverified``: the whole timeline replayed, but the simulation had not
    ended when the real match did, so its outcome could not be compared.
    ``partial``: the reconstruction stopped early; frames before
    ``first_untrusted_tick`` are still usable.
    ``diverged``: the simulated match ended differently from the real one, so
    somewhere it stopped being the match that was played.
    """

    result = {
        key: metadata.get(key)
        for key in (
            "completed",
            "failure_reason",
            "first_untrusted_tick",
            "simulated_end_tick",
            "source_end_tick",
            "expert_action_count",
            "executed_expert_action_count",
            "winner_matches",
            "crowns_match",
        )
    }
    if metadata.get("winner_matches") is False or metadata.get("crowns_match") is False:
        verdict = "diverged"
    elif metadata.get("completed") is True:
        verdict = "faithful" if metadata.get("winner_matches") is True else "unverified"
    else:
        verdict = "partial"
    result["verdict"] = verdict
    return result


def _attach_opponent_actions(players: Sequence[dict[str, Any]]) -> None:
    if len(players) != 2:
        return
    for player, opponent in ((players[0], players[1]), (players[1], players[0])):
        for swing in player["swings"]:
            swing["opponent_actions"] = [
                int(action["index"])
                for action in opponent["actions"]
                if swing["start_tick"] - SWING_CONTEXT_TICKS <= int(action["tick"]) <= swing["end_tick"]
            ]


def build_review(
    *,
    replay_tag: str,
    model: Mapping[str, Any],
    fidelity: Mapping[str, Any],
    frames_by_owner: Mapping[int, Sequence[FrameScore]],
    match: MatchContext | None = None,
) -> dict[str, Any]:
    actual = match or MatchContext()
    players = [
        build_player_review(owner, frames, context=actual.players.get(owner))
        for owner, frames in sorted(frames_by_owner.items())
    ]
    _attach_opponent_actions(players)
    return {
        "schema": REVIEW_SCHEMA,
        "replay_tag": replay_tag,
        "model": dict(model),
        "conventions": CONVENTIONS,
        "fidelity": fidelity_report(fidelity),
        "match": {"game_mode": actual.game_mode, "duration_seconds": actual.duration_seconds},
        "players": players,
    }


_VERDICT_TEXT = {
    "faithful": "可信（重演走完全程，胜负与皇冠数一致）",
    "unverified": "未核对（重演走完全程，但模拟没有在原对局结束时分出胜负，结果无法比对）",
    "partial": "部分可信（重演提前中止，只有中止前的帧可用）",
    "diverged": "已偏离（模拟结果与真实对局不同，以下内容仅供参考）",
}
_RESULT_TEXT = {"win": "胜", "loss": "负", "draw": "平"}


def _number(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.0f}%"


def _choice_text(choice: Mapping[str, Any] | None) -> str:
    if choice is None:
        return "—"
    text = f"{choice['card']['name']}（{_percent(choice['card_probability'])}）"
    if choice["cell"] is not None:
        text += f" → {choice['zone']} {tuple(choice['cell'])}"
    return text


def render_summary(review: Mapping[str, Any]) -> str:
    """Render a short Chinese digest of one review document."""

    fidelity = review["fidelity"]
    lines = [
        f"回放 {review['replay_tag']} | 模型 {review['model'].get('checkpoint_file', '?')}",
        f"重演保真度：{_VERDICT_TEXT[fidelity['verdict']]}",
    ]
    if fidelity["failure_reason"]:
        lines.append(f"  原因：{fidelity['failure_reason']}")
    for player in review["players"]:
        summary = player["summary"]
        agreement = summary["agreement"]
        title = f"玩家 {player['owner']}"
        details = [item for item in (player["side"], _RESULT_TEXT.get(player["result"])) if item]
        if player["crowns"] is not None:
            details.append(f"{player['crowns']} 冠")
        if details:
            title += f"（{'，'.join(details)}）"
        lines += ["", title]
        lines.append(
            f"  可信决策帧 {summary['trusted_frames']}/{summary['frames']}，已评分出牌 {summary['actions']} 次"
        )
        unscored = player["unscored_source_actions"]
        if unscored:
            lines.append(f"  有 {len(unscored)} 次出牌未能评分：" + "、".join(item["time"] for item in unscored[:8]))
        value = summary["value"]
        if value["start"] is not None:
            lines.append(
                f"  局势值：开局 {_number(value['start'])} → 终局 {_number(value['end'])}，"
                f"最低 {_number(value['min'])}（{clock(value['min_tick'])}），"
                f"最高 {_number(value['max'])}（{clock(value['max_tick'])}）"
            )
        elixir = summary["elixir"]
        leak = f"  圣水：平均 {_number(elixir['mean'], 1)}，估计漏水 {_number(elixir['leaked_estimate'], 1)}"
        if elixir["leaked_reported_by_source"] is not None:
            leak += f"（来源统计 {_number(elixir['leaked_reported_by_source'], 1)}）"
        lines.append(leak)
        lines.append(
            f"  与模型一致：出牌时机 {_percent(agreement['act_when_player_acted'])}，"
            f"等待 {_percent(agreement['wait_when_player_waited'])}，"
            f"选牌 {_percent(agreement['same_card'])}，"
            f"落点中位偏差 {_number(agreement['median_cell_distance'], 1)} 格"
        )
        disputed = sorted(
            (
                action
                for action in player["actions"]
                if action["model_choice"] is not None and not action["model_choice"]["same_card"]
            ),
            key=lambda action: action["card_probability"],
        )[:DISAGREEMENT_LIMIT]
        if disputed:
            lines.append("  模型最不认同的选牌：")
            for action in sorted(disputed, key=lambda action: action["tick"]):
                where = f" → {action['zone']} {tuple(action['cell'])}" if action["cell"] is not None else ""
                lines.append(
                    f"    {action['time']} {action['card']['name']}{where}，"
                    f"模型给这张牌 {_percent(action['card_probability'])}，"
                    f"更想出 {_choice_text(action['model_choice'])}"
                )
        misplaced = sorted(
            (
                action
                for action in player["actions"]
                if action["cell_distance"] is not None and action["cell_distance"] >= PLACEMENT_MINIMUM_DISTANCE
            ),
            key=lambda action: -action["cell_distance"],
        )[:PLACEMENT_LIMIT]
        if misplaced:
            lines.append("  落点分歧最大的出牌：")
            for action in sorted(misplaced, key=lambda action: action["tick"]):
                lines.append(
                    f"    {action['time']} {action['card']['name']} 放在 {action['zone']} {tuple(action['cell'])}，"
                    f"模型同一张牌首选 {zone_label(action['model_cell'])} {tuple(action['model_cell'])}"
                    f"（相距 {_number(action['cell_distance'], 1)} 格，"
                    f"你的落点排第 {action['cell_rank']}/{action['legal_cell_count']}）"
                )
        if player["missed_windows"]:
            lines.append("  模型想出牌而你在等：")
            for window in player["missed_windows"]:
                lines.append(
                    f"    {window['time']}（{_number(window['duration_seconds'], 1)} 秒），"
                    f"圣水 {_number(window['elixir_at_peak'], 1)}，"
                    f"出牌意愿 {_percent(window['peak_act_probability'])}，"
                    f"首选 {_choice_text(window['model_choice'])}"
                )
        if player["swings"]:
            lines.append("  局势转折（5 秒窗口）：")
            for swing in player["swings"]:
                cards = "、".join(player["actions"][index]["card"]["name"] for index in swing["own_actions"])
                lines.append(
                    f"    {swing['time']} 局势 {swing['value_change']:+.2f}"
                    f"（{_number(swing['value_start'])} → {_number(swing['value_end'])}）"
                    + (f"，期间己方出牌：{cards}" if cards else "")
                )
    return "\n".join(lines)
