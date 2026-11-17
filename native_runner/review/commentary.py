"""Ask Claude for a written review of one player's match.

The policy scores every decision; this turns those scores into the commentary
a coach would give: how the match went, which moments are worth rewatching and
what to try next time.  The model only sees a brief distilled from the review
document, never the raw timeline, and is told what that brief cannot show.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping

import anthropic
from pydantic import BaseModel, Field, ValidationError

from ..arena import GRID_HEIGHT, GRID_WIDTH
from .report import ELIXIR_CAP, ELIXIR_CAP_TOLERANCE, TICKS_PER_SECOND, clock, zone_label


COMMENTARY_MODEL = "claude-opus-5-5"
# Server-side fallback re-runs a request the safety classifiers decline on the
# model Anthropic recommends for that category, inside the same call.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
DEFAULT_EFFORT = "high"
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
# Thinking is always on for this model and shares the output budget.
MAX_TOKENS = 32000
BRIEF_TIMELINE_SECONDS = 5
_CLOCK = re.compile(r"^(\d{1,2}):([0-5]\d)$")

SYSTEM_PROMPT = """你是一名《皇室战争》教练，正在帮一位玩家复盘他打过的一局。

你的讲解会显示在一个报告页面里。页面上已经有局势曲线、圣水和塔血的图表，以及逐次出牌的数据表，所以玩家不需要你把数字再念一遍。他需要的是把这些数字连成一个故事：这局的走向是怎么形成的，哪几个时刻值得回看，下次可以怎么打得不一样。

## 数据是怎么来的

这局对局是用回放里记录的出牌，在游戏引擎里重演出来的，再由一个对战模型逐帧评价。你拿到的是这份评价的摘要（JSON）。

数据里有：
- 双方每次出牌的时间、卡牌和落点；
- 被复盘玩家的圣水，以及双方塔血随时间的变化；
- 对战模型的评价：局势值、出牌意愿、它对每次选牌和落点的看法。

数据里没有：
- 场上单位的交战过程。谁打了谁、哪张牌被什么解掉、伤害是谁造成的，这些都不在数据里。
- 对手的手牌和圣水。

你可以结合卡牌常识去推测原因，比如根据双方出牌的先后和位置判断一次进攻大概是怎么被化解的。但推测要说成推测（“很可能是……”），不要写成你看到的事实。

## 这些数字是什么意思

- value（局势值）：模型对胜负的期望（胜 +1、负 −1），外加塔血的影响。它不是校准过的胜率，重点看它的变化，而不是绝对值。
- act_probability（出牌意愿）：模型此刻有多想出牌，0 到 1，只适合当相对分数看。
- card_probability：在当时能出的牌里，模型选这张牌的概率。model_pick 是模型的首选，"same" 表示和玩家一样。
- cell_rank：玩家的落点在全部合法格子里按模型偏好排第几；model_cell 是模型用同一张牌会放的位置；cell_distance 是两者相距几格。
- value_change_5s：这次出牌之后 5 秒局势值的变化。它同时受对手行动影响，不能全算在这一张牌头上。

模型的看法是线索，不是标准答案。概率低只说明这个模型很少这么打，不等于打错了。尤其要注意：对战模型在场上没有单位时往往偏被动，倾向于等对手先出牌，所以它在开局和空场阶段不认同主动出牌，并不能说明那次出牌有问题。

## 坐标

落点都已换算成被复盘玩家自己的视角：x 从 0 到 17 是从左到右，y 从 0 到 31 是从己方国王塔到对方国王塔，河道在 y=15 到 16，左右两路各有一座桥。zone 字段给出了对应的文字描述，讲解时用文字描述位置，不要报坐标。

## 重演可信吗

fidelity.verdict 说明这次重演与真实对局的吻合程度：
- faithful：重演走完全程，胜负和皇冠数与真实对局一致。
- unverified：走完了全程，但模拟没有在原对局结束的时刻分出胜负，结果无法核对。
- partial：重演提前中止，只有 trusted_until 之前的内容可用。
- diverged：模拟的结果和真实对局不同，说明中途某处已经偏离。

如果不是 faithful，请在 caveats 里告诉玩家，并相应降低结论的确定程度；partial 时只评价可信的那一段。

## 怎么写

直接对玩家说话，用“你”。具体：带上时间点，点名卡牌。口吻像一个认真看过录像的教练，不夸大，也不为了挑毛病而挑毛病——如果某一段打得没问题，可以直说。

- overview：两到四句，讲清这局的走向和关键所在。
- moments：三到六个最值得回看的时刻，按时间先后排列。每个时刻给出起止时间（m:ss，必须在对局时间之内）、一句话标题，以及分析：数据显示发生了什么，模型怎么看，换一种打法会是什么样。优先选局势值变化大的时段、模型明显不认同的出牌、满圣水和模型想出牌而玩家在等的时段。
- habits：贯穿全局、反复出现的习惯问题，最多四条。只写数据里确实重复出现的模式，没有就留空。
- suggestions：两到四条下一局可以直接尝试的具体做法。
- caveats：一两句，说明这份讲解的局限，包括数据缺什么、重演是否可信。"""


class CommentaryError(RuntimeError):
    """Raised when Claude returned no usable commentary."""


class MomentText(BaseModel):
    start: str = Field(description="这个时刻的开始时间，格式 m:ss，是开局后的经过时间")
    end: str = Field(description="这个时刻的结束时间，格式 m:ss")
    title: str = Field(description="一句话标题")
    analysis: str = Field(description="发生了什么、模型怎么看、可以怎么打得不同")


class CommentaryText(BaseModel):
    overview: str = Field(description="两到四句，这局的走向和关键所在")
    moments: list[MomentText] = Field(description="三到六个最值得回看的时刻，按时间先后")
    habits: list[str] = Field(description="反复出现的习惯问题，最多四条，没有就留空")
    suggestions: list[str] = Field(description="两到四条下一局可以直接尝试的做法")
    caveats: str = Field(description="这份讲解的局限：数据缺什么、重演是否可信")


def _mirror(cell: list[int] | None) -> list[int] | None:
    """Turn a cell from one player's view into the other player's view."""

    return None if cell is None else [GRID_WIDTH - 1 - cell[0], GRID_HEIGHT - 1 - cell[1]]


def _own_action(action: Mapping[str, Any]) -> dict[str, Any]:
    choice = action["model_choice"]
    if choice is None:
        pick: Any = None
    elif choice["same_card"]:
        pick = "same"
    else:
        pick = {"name": choice["name"], "probability": choice["card_probability"], "zone": choice["zone"]}
    return {
        "time": action["time"],
        "name": action["name"],
        "cost": action["cost"],
        "zone": action["zone"],
        "elixir_before": action["elixir_before"],
        "act_probability": action["act_probability"],
        "card_probability": action["card_probability"],
        "model_pick": pick,
        "cell_rank": f"{action['cell_rank']}/{action['legal_cell_count']}" if action["cell_rank"] else None,
        "model_cell_zone": zone_label(action["model_cell"]) if action["model_cell"] else None,
        "cell_distance": action["cell_distance"],
        "value_before": action["value_before"],
        "value_change_5s": action["value_change"],
    }


def _opponent_actions(opponent: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if opponent is None:
        return []
    plays = [
        {
            "tick": action["tick"],
            "time": action["time"],
            "name": action["name"],
            "cost": action["cost"],
            "zone": zone_label(_mirror(action["cell"])) if action["cell"] else None,
        }
        for action in opponent["actions"]
    ]
    for entry in opponent["unscored_source_actions"] or ():
        name = entry["card"]["name"] if entry["card"] else "技能"
        plays.append({"tick": entry["tick"], "time": entry["time"], "name": name, "cost": None, "zone": None})
    plays.sort(key=lambda item: item["tick"])
    return [{key: value for key, value in play.items() if key != "tick"} for play in plays]


def _full_elixir_periods(timeline: Mapping[str, Any]) -> list[dict[str, str]]:
    periods: list[dict[str, str]] = []
    start: int | None = None
    last = 0
    for tick, valid, elixir in zip(timeline["tick"], timeline["valid"], timeline["elixir"], strict=True):
        full = valid and elixir is not None and elixir >= ELIXIR_CAP - ELIXIR_CAP_TOLERANCE
        if full and start is None:
            start = tick
        if not full and start is not None:
            periods.append({"start": clock(start), "end": clock(last)})
            start = None
        last = tick
    if start is not None:
        periods.append({"start": clock(start), "end": clock(last)})
    return periods


def build_brief(review: Mapping[str, Any], owner: int) -> dict[str, Any]:
    """Distil one player's review into what Claude needs to comment on it."""

    players = {int(player["owner"]): player for player in review["players"]}
    if owner not in players:
        raise ValueError(f"review has no player {owner}; it has {sorted(players)}")
    player = players[owner]
    opponent = players.get(1 - owner)
    timeline = player["timeline"]
    stride = max(1, BRIEF_TIMELINE_SECONDS * TICKS_PER_SECOND // int(review["conventions"]["decision_interval_ticks"]))
    trusted = [index for index, valid in enumerate(timeline["valid"]) if valid]
    sampled = [index for index in trusted if index % stride == 0]
    if trusted and trusted[-1] not in sampled:
        sampled.append(trusted[-1])
    fidelity = review["fidelity"]
    return {
        "match": {
            "game_mode": review["match"]["game_mode"],
            "duration": clock(timeline["tick"][-1]) if timeline["tick"] else None,
            "evaluating_model": review["model"].get("checkpoint_file"),
        },
        "fidelity": {
            "verdict": fidelity["verdict"],
            "failure_reason": fidelity["failure_reason"],
            "executed_actions": fidelity["executed_expert_action_count"],
            "source_actions": fidelity["expert_action_count"],
            "trusted_until": clock(timeline["tick"][trusted[-1]]) if trusted else None,
        },
        "player": {"side": player["side"], "result": player["result"], "crowns": player["crowns"]},
        "opponent": (
            None
            if opponent is None
            else {"side": opponent["side"], "result": opponent["result"], "crowns": opponent["crowns"]}
        ),
        "summary": player["summary"],
        "timeline": [
            {
                "time": clock(timeline["tick"][index]),
                "value": timeline["value"][index],
                "act_probability": timeline["act_probability"][index],
                "elixir": timeline["elixir"][index],
                "own_towers": timeline["own_towers"][index],
                "enemy_towers": timeline["enemy_towers"][index],
            }
            for index in sampled
        ],
        "tower_order": ["king", "princess_left", "princess_right"],
        "own_actions": [_own_action(action) for action in player["actions"]],
        "own_actions_not_scored": [
            {"time": entry["time"], "name": entry["card"]["name"] if entry["card"] else "技能"}
            for entry in player["unscored_source_actions"] or ()
        ],
        "opponent_actions": _opponent_actions(opponent),
        "full_elixir_periods": _full_elixir_periods(timeline),
        "model_wanted_to_act_while_player_waited": [
            {
                "time": window["time"],
                "seconds": window["duration_seconds"],
                "elixir": window["elixir_at_peak"],
                "peak_act_probability": window["peak_act_probability"],
                "model_pick": window["model_choice"]["name"] if window["model_choice"] else None,
                "model_pick_zone": window["model_choice"]["zone"] if window["model_choice"] else None,
            }
            for window in player["missed_windows"]
        ],
        "largest_value_swings": [
            {
                "time": swing["time"],
                "value_start": swing["value_start"],
                "value_end": swing["value_end"],
                "value_change": swing["value_change"],
            }
            for swing in player["swings"]
        ],
    }


def commentary_request(brief: Mapping[str, Any], *, effort: str = DEFAULT_EFFORT) -> dict[str, Any]:
    """Build the Messages API arguments for one commentary."""

    if effort not in EFFORT_LEVELS:
        raise ValueError(f"effort must be one of {EFFORT_LEVELS}")
    data = json.dumps(brief, ensure_ascii=False, allow_nan=False)
    return {
        "model": COMMENTARY_MODEL,
        "max_tokens": MAX_TOKENS,
        "betas": [FALLBACK_BETA],
        "fallbacks": "default",
        "output_config": {
            "effort": effort,
            "format": {"type": "json_schema", "schema": anthropic.transform_schema(CommentaryText)},
        },
        "system": SYSTEM_PROMPT,
        "messages": [
            {"role": "user", "content": f"下面是这局对局的数据。请为 player 字段所指的这名玩家写复盘讲解。\n\n{data}"}
        ],
    }


def _tick(text: str) -> int | None:
    match = _CLOCK.match(text.strip())
    return None if match is None else (int(match.group(1)) * 60 + int(match.group(2))) * TICKS_PER_SECOND


def request_commentary(
    brief: Mapping[str, Any], *, client: anthropic.Anthropic | None = None, effort: str = DEFAULT_EFFORT
) -> dict[str, Any]:
    """Send one brief to Claude and return the commentary stored in a review."""

    actual = client or anthropic.Anthropic()
    with actual.beta.messages.stream(**commentary_request(brief, effort=effort)) as stream:
        message = stream.get_final_message()
    if message.stop_reason == "refusal":
        category = getattr(getattr(message, "stop_details", None), "category", None)
        raise CommentaryError(f"Claude declined to write this commentary (category: {category})")
    if message.stop_reason == "max_tokens":
        raise CommentaryError(f"the commentary was cut off at {MAX_TOKENS} output tokens")
    # The stop reason is checked first: a declined or truncated answer is not valid JSON either.
    try:
        text = CommentaryText.model_validate_json(
            "".join(block.text for block in message.content if block.type == "text")
        )
    except ValidationError as error:
        raise CommentaryError(f"Claude's answer did not match the commentary format: {error}") from error
    moments = []
    for moment in text.moments:
        start, end = _tick(moment.start), _tick(moment.end)
        # A time Claude wrote in another shape still reads fine; it just cannot be linked to the timeline.
        linked = start is not None and end is not None and end >= start
        moments.append(
            {
                "time": moment.start if moment.start == moment.end else f"{moment.start}–{moment.end}",
                "start_tick": start if linked else None,
                "end_tick": end if linked else None,
                "title": moment.title,
                "analysis": moment.analysis,
            }
        )
    return {
        "model": message.model,
        "effort": effort,
        "usage": {"input_tokens": message.usage.input_tokens, "output_tokens": message.usage.output_tokens},
        "overview": text.overview,
        "moments": moments,
        "habits": list(text.habits),
        "suggestions": list(text.suggestions),
        "caveats": text.caveats,
    }


def render_commentary(commentary: Mapping[str, Any]) -> str:
    """Render one stored commentary as plain text for the terminal."""

    lines = [commentary["overview"], ""]
    if commentary["moments"]:
        lines.append("值得回看的时刻：")
        for moment in commentary["moments"]:
            lines += [f"  {moment['time']}  {moment['title']}", f"    {moment['analysis']}"]
        lines.append("")
    for title, items in (("习惯问题：", commentary["habits"]), ("下一局可以试试：", commentary["suggestions"])):
        if items:
            lines.append(title)
            lines += [f"  - {item}" for item in items]
            lines.append("")
    if commentary["caveats"]:
        lines.append(f"局限：{commentary['caveats']}")
    return "\n".join(lines).rstrip()
