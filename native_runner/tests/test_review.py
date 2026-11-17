from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest
import torch

from native_runner.match_factory import STANDARD_DECK
from native_runner.review import __main__ as review_cli
from native_runner.review.page import render_page
from native_runner.review.report import (
    PlayerContext,
    build_player_review,
    build_review,
    estimate_leaked_elixir,
    fidelity_report,
    find_missed_windows,
    find_swings,
    render_summary,
)
from native_runner.review.scoring import (
    ENCODE_CHUNK_FRAMES,
    REVIEW_TEMPERATURE,
    FrameScore,
    ModelChoice,
    PlayedAction,
    TowerHitpoints,
    score_sequence,
)
from native_runner.review.sources import (
    iter_cached_replays,
    list_cached_replays,
    load_replay_payloads,
    match_context,
    select_replays,
)
from native_runner.tests.review_helpers import (
    HERO_ABILITY_COST,
    HERO_CARD_ID,
    SyntheticPlay,
    scripted_review,
    synthetic_owner_sequence,
    synthetic_result,
)
from native_runner.training.v4.cache import write_il_cache_index, write_il_cache_shard
from native_runner.training.v4.expert import FIRST_POLICY_DECISION_TICK, POLICY_DECISION_TICKS
from native_runner.training.v4.factory import build_production_model_v4
from native_runner.training.v4.learning import evaluate_recurrent_sequence
from native_runner.training.v4.producer import ReplayProductionResultV4
from native_runner.training.v4.tensors import GATE_ACT


FRAMES = ENCODE_CHUNK_FRAMES + 8
PLAYS = (
    SyntheticPlay(frame=4, hand_position=1, native_cell=(3, 10), delay_ticks=2),
    SyntheticPlay(frame=20, hand_position=0, native_cell=(9, 12)),
    SyntheticPlay(frame=30, hand_position=2, native_cell=(3, 3), alignable=False),
)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(7)
    return build_production_model_v4().eval()


@pytest.fixture(scope="module")
def owner0_sequence():
    return synthetic_owner_sequence(owner=0, frames=FRAMES, plays=PLAYS, damage_per_frame=25.0)


@pytest.fixture(scope="module")
def owner0_frames(model, owner0_sequence):
    return score_sequence(model, owner0_sequence)


def test_scoring_matches_teacher_forced_evaluation(model, owner0_sequence, owner0_frames) -> None:
    with torch.inference_mode():
        reference = evaluate_recurrent_sequence(
            model,
            tuple(item.to_model_input("cpu") for item in owner0_sequence.observations),
            owner0_sequence.actions,
            owner0_sequence.episode_start,
            gate_temperature=REVIEW_TEMPERATURE,
            action_temperature=REVIEW_TEMPERATURE,
            continue_temperature=REVIEW_TEMPERATURE,
        )

    assert len(owner0_frames) == FRAMES > ENCODE_CHUNK_FRAMES
    for frame in owner0_frames:
        assert frame.value == pytest.approx(float(reference.value[frame.index, 0]), abs=1e-5)
        gate = math.exp(float(reference.components.gate_log_prob[frame.index, 0]))
        acted = int(owner0_sequence.actions[frame.index].gate[0]) == GATE_ACT
        assert frame.act_probability == pytest.approx(gate if acted else 1.0 - gate, abs=1e-5)
        assert bool(frame.played) == acted
        for step, action in enumerate(frame.played):
            expected = math.exp(float(reference.components.candidate_log_prob[frame.index, 0, step]))
            assert action.card_probability == pytest.approx(expected, abs=1e-5)
            expected_cell = math.exp(float(reference.components.target_log_prob[frame.index, 0, step]))
            assert action.cell_probability == pytest.approx(expected_cell, rel=1e-4, abs=1e-7)


def test_scoring_reports_frame_facts_in_the_players_own_view(model, owner0_frames) -> None:
    first = owner0_frames[0]
    assert first.tick == FIRST_POLICY_DECISION_TICK
    assert owner0_frames[4].tick == FIRST_POLICY_DECISION_TICK + 4 * POLICY_DECISION_TICKS
    assert first.elixir == pytest.approx(5.0, abs=0.01)
    assert first.elixir_multiplier == pytest.approx(1.0, abs=0.01)
    assert first.remaining_seconds == pytest.approx(180.0 - first.tick / 20.0, abs=0.1)
    assert first.own_towers.king == pytest.approx(4800.0, abs=3.0)
    assert owner0_frames[10].own_towers.total == pytest.approx(10800.0, abs=6.0)
    assert owner0_frames[10].enemy_towers.princess_left == pytest.approx(3000.0 - 250.0, abs=3.0)
    assert owner0_frames[10].enemy_towers.princess_right == pytest.approx(3000.0, abs=3.0)

    (played,) = owner0_frames[4].played
    assert played.kind == "play_card"
    assert played.card_id == STANDARD_DECK[1]
    assert played.cell == (3, 10)
    assert played.tick == owner0_frames[4].tick + 2
    assert owner0_frames[20].played[0].card_id == STANDARD_DECK[0]

    # The unalignable play stays a WAIT frame that must not be judged.
    assert not owner0_frames[30].played
    assert not owner0_frames[30].label_usable
    assert all(frame.label_usable for frame in owner0_frames if frame.index != 30)

    # Owner 1 sees the arena rotated: the same own-half cell and a mirrored tower side.
    mirrored = synthetic_owner_sequence(
        owner=1,
        frames=12,
        plays=(SyntheticPlay(frame=4, hand_position=1, native_cell=(14, 21)),),
        damage_per_frame=25.0,
        untrusted_from_frame=9,
    )
    frames = score_sequence(model, mirrored)
    assert frames[4].played[0].cell == (3, 10)
    assert frames[8].enemy_towers.princess_right == pytest.approx(3000.0 - 200.0, abs=3.0)
    assert frames[8].enemy_towers.princess_left == pytest.approx(3000.0, abs=3.0)
    assert [frame.valid for frame in frames] == [True] * 9 + [False] * 3
    assert frames[8].realized_return is not None and frames[9].realized_return is None


def test_options_and_model_choice_follow_the_policy_distribution(model, owner0_sequence, owner0_frames) -> None:
    frame = owner0_frames[4]
    (played,) = frame.played
    assert sum(option.probability for option in frame.options) == pytest.approx(1.0, abs=1e-5)
    assert [option.probability for option in frame.options] == sorted(
        (option.probability for option in frame.options), reverse=True
    )
    matching = next(option for option in frame.options if option.card_id == played.card_id)
    assert matching.probability == pytest.approx(played.card_probability, abs=1e-5)

    assert played.legal_cell_count == 15 * 18
    assert 1 <= played.cell_rank <= played.legal_cell_count
    assert (played.cell_rank == 1) == (played.cell == played.model_cell)
    assert (played.top_cells[0].x, played.top_cells[0].y) == played.model_cell
    assert played.top_cells[0].probability == pytest.approx(played.model_cell_probability)
    assert all(cell.y < 15 for cell in played.top_cells)

    # The reported preference is exactly what the policy plays when forced to act.
    with torch.inference_mode():
        state = model.initial_state(1)
        for index in range(5):
            batch = owner0_sequence.observations[index].to_model_input("cpu")
            context = model.forward(batch, state, episode_start=owner0_sequence.episode_start[index], validate=False)
            state = context.next_state
        forced = model.act_after_preselected_act(batch, context, validate=False)
    row = int(forced.actions.candidate_index[0, 0])
    target = int(forced.actions.target_cell[0, 0])
    choice = frame.model_choice
    assert choice.card_id == int(batch.candidates.native_visible_card_id[0, row])
    assert choice.cell == (target % 18, target // 18)
    assert choice.card_probability == pytest.approx(frame.options[0].probability)

    # No affordable card means there is nothing the policy could prefer.
    broke = synthetic_owner_sequence(owner=0, frames=2, start_elixir=1.0)
    (empty, _) = score_sequence(model, broke)
    assert empty.options == () and empty.model_choice is None and empty.act_probability == 0.0


def test_abilities_are_scored_and_named_after_the_card_that_owns_them(model) -> None:
    sequence = synthetic_owner_sequence(
        owner=1, frames=8, plays=(SyntheticPlay(frame=5, hand_position=1, native_cell=(14, 21)),), ability_frames=(2,)
    )

    frames = score_sequence(model, sequence)

    (ability,) = frames[2].played
    assert (ability.kind, ability.card_id, ability.cost) == ("activate_ability", HERO_CARD_ID, HERO_ABILITY_COST)
    assert ability.cell is None and ability.cell_probability is None and ability.top_cells == ()
    # The hero card is also in hand, so the ability and the card are separate options.
    kinds = {option.kind for option in frames[2].options if option.card_id == HERO_CARD_ID}
    assert kinds == {"activate_ability", "play_card"}
    matching = next(option for option in frames[2].options if option.kind == "activate_ability")
    assert ability.card_probability == pytest.approx(matching.probability, abs=1e-5)
    assert frames[3].elixir == pytest.approx(frames[2].elixir + 0.09 - HERO_ABILITY_COST, abs=0.01)

    context = PlayerContext(source_actions=((ability.tick, None), (frames[5].played[0].tick, STANDARD_DECK[1])))
    player = build_player_review(1, frames, context=context)
    first, second = player["actions"]
    assert (first["kind"], first["card"]["id"], first["cell"], first["zone"]) == (
        "activate_ability",
        HERO_CARD_ID,
        None,
        None,
    )
    assert first["cell_distance"] is None and first["top_cells"] == [] and first["model_choice"] is not None
    assert second["kind"] == "play_card" and second["cell"] == [3, 10]
    assert player["unscored_source_actions"] == []


def _frame(index: int, **changes: object) -> FrameScore:
    base = FrameScore(
        index=index,
        tick=FIRST_POLICY_DECISION_TICK + index * POLICY_DECISION_TICKS,
        valid=True,
        label_usable=True,
        value=0.0,
        realized_return=None,
        act_probability=0.1,
        elixir=5.0,
        elixir_multiplier=1.0,
        remaining_seconds=100.0,
        overtime=False,
        own_towers=TowerHitpoints(4800.0, 3000.0, 3000.0),
        enemy_towers=TowerHitpoints(4800.0, 3000.0, 3000.0),
        options=(),
        played=(),
        model_choice=ModelChoice("play_card", STANDARD_DECK[0], "normal", 3.0, 0.8, (3, 10), 0.4),
    )
    return replace(base, **changes)


def _play(tick: int, card_id: int = STANDARD_DECK[1]) -> PlayedAction:
    return PlayedAction(
        kind="play_card",
        card_id=card_id,
        form="normal",
        cost=3.0,
        tick=tick,
        cell=(3, 10),
        card_probability=0.2,
        cell_probability=0.01,
        cell_rank=5,
        legal_cell_count=270,
        model_cell=(3, 14),
        model_cell_probability=0.3,
    )


def test_leak_estimate_counts_only_trusted_frames_at_the_cap() -> None:
    frames = [
        _frame(0, elixir=10.0),
        _frame(1, elixir=10.0, elixir_multiplier=2.0),
        _frame(2, elixir=9.9),
        _frame(3, elixir=10.0, valid=False),
    ]

    leaked, seconds = estimate_leaked_elixir(frames)

    assert seconds == pytest.approx(0.5)
    assert leaked == pytest.approx(0.25 / 2.8 + 0.25 * 2.0 / 2.8)


def test_missed_windows_need_a_sustained_wish_to_act_while_the_player_waited() -> None:
    eager = {"act_probability": 0.9}
    frames = [_frame(index) for index in range(30)]
    for index in range(2, 8):
        frames[index] = _frame(index, **eager)
    frames[5] = _frame(5, act_probability=0.97, elixir=9.5)
    for index in range(10, 12):  # too short to matter
        frames[index] = _frame(index, **eager)
    for index in range(14, 20):  # the player was acting, not waiting
        frames[index] = _frame(index, **eager, played=(_play(frames[index].tick),))
    for index in range(22, 28):  # the label is unknown, so no judgement
        frames[index] = _frame(index, **eager, label_usable=False)

    (window,) = find_missed_windows(frames)

    assert window["start_tick"] == frames[2].tick
    assert window["end_tick"] == frames[8].tick
    assert window["duration_seconds"] == pytest.approx(1.5)
    assert window["peak_tick"] == frames[5].tick
    assert window["elixir_at_peak"] == pytest.approx(9.5)
    assert window["model_choice"]["card"]["id"] == STANDARD_DECK[0]
    assert window["model_choice"]["zone"] == "己方前场·左路"


def test_swings_pick_the_largest_separate_moves_and_skip_noise() -> None:
    values = [0.0] * 30 + [-0.02 * step for step in range(1, 21)] + [-0.4] * 30 + [-0.4 + 0.01] * 20
    frames = [_frame(index, value=value) for index, value in enumerate(values)]
    frames[32] = replace(frames[32], played=(_play(frames[32].tick),))
    player = build_player_review(0, frames)

    (swing,) = player["swings"]

    assert swing == find_swings(frames, player["actions"])[0]
    assert swing["value_change"] == pytest.approx(-0.4)
    assert swing["start_tick"] == frames[29].tick and swing["end_tick"] == frames[49].tick
    assert swing["own_actions"] == [0]
    assert player["actions"][0]["value_change"] == pytest.approx(values[52] - values[32])


def test_player_review_summarises_agreement_and_unscored_source_actions() -> None:
    frames = [_frame(index) for index in range(12)]
    frames[3] = _frame(3, act_probability=0.7, played=(_play(frames[3].tick + 1, STANDARD_DECK[0]),))
    frames[6] = _frame(6, act_probability=0.2, played=(_play(frames[6].tick),))
    frames[9] = _frame(9, label_usable=False)
    context = PlayerContext(
        side="team",
        result="win",
        crowns=2,
        source_elixir_leaked=1.5,
        source_actions=(
            (frames[3].tick + 1, None),
            (frames[3].tick + 1, STANDARD_DECK[0]),
            (frames[6].tick, STANDARD_DECK[1]),
            (frames[9].tick + 3, STANDARD_DECK[2]),
        ),
    )

    player = build_player_review(0, frames, context=context)

    agreement = player["summary"]["agreement"]
    assert agreement["act_when_player_acted"] == pytest.approx(0.5)
    assert agreement["wait_when_player_waited"] == pytest.approx(1.0)
    assert agreement["same_card"] == pytest.approx(0.5)
    assert agreement["median_cell_distance"] == pytest.approx(4.0)
    assert player["summary"]["unusable_label_frames"] == 1
    assert player["summary"]["elixir"]["leaked_reported_by_source"] == 1.5
    assert [action["time"] for action in player["actions"]] == ["0:05", "0:06"]
    ability, unscored = player["unscored_source_actions"]
    assert ability == {"tick": frames[3].tick + 1, "time": "0:05", "card": None}
    assert unscored["tick"] == frames[9].tick + 3 and unscored["card"]["id"] == STANDARD_DECK[2]
    assert len(player["timeline"]["value"]) == len(frames)


@pytest.mark.parametrize(
    ("metadata", "verdict"),
    [
        ({"completed": True, "winner_matches": True, "crowns_match": True}, "faithful"),
        ({"completed": True, "winner_matches": None, "crowns_match": None}, "unverified"),
        ({"completed": False, "winner_matches": False, "crowns_match": True}, "diverged"),
        ({"completed": False, "winner_matches": True, "crowns_match": False}, "diverged"),
        ({"completed": False, "winner_matches": None, "crowns_match": None, "first_untrusted_tick": 800}, "partial"),
    ],
)
def test_fidelity_verdict_separates_divergence_from_early_stops(metadata: dict, verdict: str) -> None:
    report = fidelity_report(metadata)

    assert report["verdict"] == verdict
    assert report["first_untrusted_tick"] == metadata.get("first_untrusted_tick")


def _write_cache(root: Path) -> Path:
    owner0 = synthetic_owner_sequence(owner=0, frames=10, plays=PLAYS[:1], sequence_id="match-a:owner-0")
    owner1 = synthetic_owner_sequence(
        owner=1, frames=10, plays=(SyntheticPlay(6, 0, (14, 21)),), start_elixir=9.9, sequence_id="match-a:owner-1"
    )
    failed = ReplayProductionResultV4(
        replay_tag="match-b",
        sequences=None,
        completed=False,
        failure_reason="calibrate: RoyaleAPIReplayError: deal is incompatible",
        first_untrusted_tick=0,
        simulated_end_tick=0,
        source_end_tick=3000,
        expert_action_count=30,
        executed_expert_action_count=0,
        winner_matches=None,
        crowns_match=None,
    )
    data_path = root / "part-00-shard-000000.cril.zst"
    summary = write_il_cache_shard(
        data_path,
        (synthetic_result("match-a", owner0, owner1, expert_action_count=2), failed),
        shard_id="part-00-shard-000000",
        contract={"fixture": True},
    )
    summary["summary_file"] = data_path.with_suffix(".json").name
    write_il_cache_index(root / "index.json", [summary], contract={"fixture": True})
    return root / "index.json"


def test_cache_listing_and_cli_review_round_trip(model, monkeypatch, tmp_path: Path, capsys) -> None:
    index = _write_cache(tmp_path / "cache")
    cached = list_cached_replays(index)
    assert {tag: item["has_sequences"] for tag, item in cached.items()} == {"match-a": True, "match-b": False}
    (replay,) = iter_cached_replays(index)
    assert replay.replay_tag == "match-a" and sorted(replay.sequences) == [0, 1]
    assert list(iter_cached_replays(index, ["match-b"])) == []

    monkeypatch.setattr(
        review_cli,
        "load_policy_v4",
        lambda checkpoint, *, device: SimpleNamespace(
            model=model,
            checkpoint_path=Path(checkpoint),
            checkpoint_id="fixture-model",
            checkpoint_sha256="0" * 64,
            update_step=1,
            training_stage="fixture",
        ),
    )
    output = tmp_path / "reviews"

    assert review_cli.main(["list", str(index)]) == 0
    listing = capsys.readouterr().out
    assert "match-a  faithful" in listing and "match-b  partial" in listing and "deal is incompatible" in listing
    assert review_cli.main(["analyze", str(index), "--output", str(tmp_path / "none"), "--tag", "missing"]) == 1
    assert "missing: not in this cache" in capsys.readouterr().err and not (tmp_path / "none").exists()

    assert review_cli.main(["analyze", str(index), "--output", str(output), "--checkpoint", "model.pt"]) == 0
    assert "回放 match-a" in capsys.readouterr().out
    review = json.loads((output / "match-a.review.json").read_text(encoding="utf-8"))
    assert review["schema"] == "firstlight-replay-review.v1"
    assert review["model"]["checkpoint_file"] == "model.pt"
    assert review["fidelity"]["verdict"] == "faithful"
    assert [player["owner"] for player in review["players"]] == [0, 1]
    assert [len(player["actions"]) for player in review["players"]] == [1, 1]
    assert review["players"][0]["actions"][0]["card"]["name_en"] == "Archers"
    assert review["players"][1]["summary"]["elixir"]["seconds_at_cap"] > 0
    assert all(len(player["timeline"]["tick"]) == 10 for player in review["players"])
    assert review["players"][0]["unscored_source_actions"] is None
    page = output / "match-a.review.html"
    assert _embedded_review(page.read_text(encoding="utf-8")) == review
    page.unlink()
    assert review_cli.main(["page", str(output / "match-a.review.json")]) == 0
    assert _embedded_review(page.read_text(encoding="utf-8")) == review

    # Asking for a replay that never produced frames is reported, not ignored.
    assert review_cli.main(["analyze", str(index), "--output", str(output), "--tag", "match-b"]) == 1
    assert "deal is incompatible" in capsys.readouterr().err
    assert not (output / "match-b.review.json").exists()


DATA_OPEN = '<script id="review-data" type="application/json">'


def _embedded_review(html: str) -> dict:
    assert html.count(DATA_OPEN) == 1
    start = html.index(DATA_OPEN) + len(DATA_OPEN)
    return json.loads(html[start : html.index("</script>", start)])


def test_page_embeds_the_whole_review_and_cannot_be_closed_by_its_text() -> None:
    review = scripted_review(partial=True)
    review["replay_tag"] = "<tag&>"
    review["fidelity"]["failure_reason"] = 'engine said </script><img src=x onerror="alert(1)"> and <!-- stopped'

    html = render_page(review)

    assert _embedded_review(html) == review
    assert "<title>对局复盘 &lt;tag&amp;&gt;</title>" in html
    assert "<img src=x" not in html and "<!-- stopped" not in html
    assert "__REVIEW_" not in html
    # Every part of the page has something to draw in the scripted match.
    for player in review["players"]:
        assert player["actions"] and player["missed_windows"] and player["swings"]
    assert review["players"][0]["unscored_source_actions"]
    assert not all(review["players"][0]["timeline"]["valid"])


def test_page_script_is_valid_javascript(tmp_path: Path) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    html = render_page(scripted_review())
    start = html.rindex("<script>") + len("<script>")
    script = tmp_path / "page.js"
    script.write_text(html[start : html.index("</script>", start)], encoding="utf-8")

    result = subprocess.run([node, "--check", str(script)], capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr


def _payload(replay_tag: str, *, events: bool = True, result: str = "victory") -> dict:
    hog = ("hog-rider", "musketeer", "cannon", "fireball", "the-log", "skeletons", "ice-golem", "ice-spirit")
    battle: dict = {"game_mode": "Ranked", "result": result}
    for side, crowns, leaked in (("team", 1, 0.5), ("opponent", 0, 2.25)):
        battle[side] = {
            "crowns": crowns,
            "players": [
                {
                    "tag": side,
                    "elixir_leaked": leaked,
                    "deck": [{"card_key": key, "name": key, "level": 11} for key in hog],
                    "tower_card": {"card_key": "tower-princess", "level": 11},
                }
            ],
        }
    plays = ((140, "team", "hog-rider", 10), (180, "opponent", "hog-rider", 21), (400, "team", "skeletons", 10))
    return {
        "schema_version": "royaleapi-battle-replay.v1",
        "source": {"replay_tag": replay_tag},
        "battle": battle,
        "events": [
            {
                "source_index": index,
                "replay_tick_20hz": tick,
                "side": side,
                "kind": "play_card",
                "card_key": card,
                "source_fields": {"data_i": 1},
                "coordinates": {
                    "inside_18x32_arena": True,
                    "grid_cell_floor": {"x": 3, "y": y},
                    "subcell_offset_from_floor_center": {"x": 0.0, "y": 0.0},
                },
            }
            for index, (tick, side, card, y) in enumerate(plays if events else ())
        ],
        "replay": {"duration": {"timeline_seconds": 45}},
    }


def _write_dataset(root: Path, payloads: list[dict]) -> None:
    import pyarrow as arrow
    import pyarrow.parquet as parquet

    (root / "replays").mkdir(parents=True)
    table = arrow.table(
        {
            "replay_tag": [payload["source"]["replay_tag"] for payload in payloads],
            "result": [payload["battle"]["result"] for payload in payloads],
            "payload_json": [json.dumps(payload) for payload in payloads],
        }
    )
    parquet.write_table(table, root / "replays" / "part-000000.parquet")


def test_match_context_binds_sides_results_and_source_actions_to_owners() -> None:
    context = match_context(_payload("2P"))

    by_side = {player.side: (owner, player) for owner, player in context.players.items()}
    assert set(by_side) == {"team", "opponent"}
    _team_owner, team = by_side["team"]
    _opponent_owner, opponent = by_side["opponent"]
    assert (team.result, team.crowns, team.source_elixir_leaked) == ("win", 1, 0.5)
    assert (opponent.result, opponent.crowns, opponent.source_elixir_leaked) == ("loss", 0, 2.25)
    assert [tick for tick, _card in team.source_actions] == [140, 400]
    assert [tick for tick, _card in opponent.source_actions] == [180]
    assert context.game_mode == "Ranked" and context.duration_seconds == 45.0
    assert {player.result for player in match_context(_payload("2Y", result="draw")).players.values()} == {"draw"}


def test_select_keeps_only_reconstructable_replays_matching_the_request(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_dataset(source, [_payload("2P"), _payload("0Y", events=False), _payload("8Q")])

    selected = select_replays(source, tmp_path / "hog", deck_cards=("hog-rider-ev1", "cannon"), limit=5)

    assert selected == ["2P", "8Q"]
    assert sorted(load_replay_payloads(tmp_path / "hog", ["2P", "8Q", "0Y"])) == ["2P", "8Q"]
    assert select_replays(source, tmp_path / "one", tags=("8Q",)) == ["8Q"]
    assert select_replays(source, tmp_path / "first", limit=1) == ["2P"]
    with pytest.raises(FileExistsError):
        select_replays(source, tmp_path / "hog", tags=("2P",))
    with pytest.raises(ValueError, match="no reconstructable replay"):
        select_replays(source, tmp_path / "none", deck_cards=("golem",))
    assert review_cli.main(["select", str(source), str(tmp_path / "cli"), "--tag", "2P", "--tag", "0Y"]) == 1


def test_review_document_and_summary_cover_both_players() -> None:
    def frames(owner_shift: float) -> list[FrameScore]:
        slide = [0.0] * 10 + [-0.02 * step for step in range(1, 21)] + [-0.4] * 10
        result = [_frame(index, value=owner_shift + value) for index, value in enumerate(slide)]
        result[5] = replace(result[5], played=(_play(result[5].tick),), act_probability=0.3)
        return result

    review = build_review(
        replay_tag="doc",
        model={"checkpoint_file": "model.pt"},
        fidelity={"completed": False, "winner_matches": False, "failure_reason": "terminal fidelity mismatch: winner"},
        frames_by_owner={1: frames(0.1), 0: frames(-0.1)},
    )

    assert [player["owner"] for player in review["players"]] == [0, 1]
    assert review["fidelity"]["verdict"] == "diverged"
    # Each side played once, two seconds before the slide began.
    (swing,) = review["players"][0]["swings"]
    assert swing["own_actions"] == [0] and swing["opponent_actions"] == [0]
    json.dumps(review, allow_nan=False)
    text = render_summary(review)
    assert "已偏离" in text and "terminal fidelity mismatch: winner" in text
    assert "模型最不认同的选牌" in text and "弓箭手" in text and "骑士" in text
    assert "落点分歧最大的出牌" in text and "局势转折" in text
