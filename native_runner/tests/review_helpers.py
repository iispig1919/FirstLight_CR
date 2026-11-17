"""Production-shaped synthetic owner sequences for review tests.

No battle engine is involved: hand-written observations go through the real
tensorizer and expert-action alignment, so the tensors have the exact layout a
cache built from a reconstructed match would have.  The "match" itself is not
a real game and its scores carry no meaning.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from native_runner.contracts import (
    ActionKind,
    ActionMaskV1,
    ActionV1,
    ObservationTier,
    ObservationV1,
    PlayerStateV1,
    TimeStateV1,
    TowerStateV1,
)
from native_runner.match_factory import STANDARD_DECK
from native_runner.training.v4.expert import (
    FIRST_POLICY_DECISION_TICK,
    POLICY_DECISION_TICKS,
    ExpertActionAlignmentError,
    TimedExpertActionV4,
    build_expert_action_batch,
)
from native_runner.training.v4.factory import production_semantic_bundle
from native_runner.training.v4.imitation import ILSequenceV4
from native_runner.training.v4.producer import ReplayProductionResultV4
from native_runner.training.v4.tensorizer import UniversalObservationTensorizerV4


GRID_WIDTH = 18
GRID_HEIGHT = 32
OWN_HALF_ROWS = 15


@dataclass(frozen=True, slots=True)
class SyntheticPlay:
    """One card play: frame index, hand position, and native target cell."""

    frame: int
    hand_position: int
    native_cell: tuple[int, int]
    delay_ticks: int = 0
    alignable: bool = True


def card_cost(card_id: int) -> float:
    return float(production_semantic_bundle().card_specs[card_id].elixir_cost)


def own_half_native_rows(owner: int) -> range:
    """Owner 0 defends the low native rows; owner 1 defends the high ones."""

    return range(OWN_HALF_ROWS) if owner == 0 else range(GRID_HEIGHT - OWN_HALF_ROWS, GRID_HEIGHT)


def _towers(damage_to_owner: dict[int, float]) -> tuple[TowerStateV1, ...]:
    return tuple(
        TowerStateV1(
            entity_id=entity_id,
            owner=owner,
            tower_kind=kind,
            position=(x, y),
            hitpoints=maximum - (damage_to_owner.get(owner, 0.0) if kind == "princess_left" else 0.0),
            max_hitpoints=maximum,
            tower_troop_id=(None if kind == "king" else 159_000_000),
        )
        for entity_id, owner, kind, x, y, maximum in (
            (1, 0, "king", 9000, 3000, 4800.0),
            (2, 0, "princess_left", 3500, 6500, 3000.0),
            (3, 0, "princess_right", 14500, 6500, 3000.0),
            (4, 1, "king", 9000, 29000, 4800.0),
            (5, 1, "princess_left", 3500, 25500, 3000.0),
            (6, 1, "princess_right", 14500, 25500, 3000.0),
        )
    )


def _observation(
    *, owner: int, tick: int, elixir: float, hand: tuple[int, ...], cycle: tuple[int, ...], damage: dict[int, float]
) -> ObservationV1:
    legal_rows = set(own_half_native_rows(owner))
    rows = [[y in legal_rows] * GRID_WIDTH for y in range(GRID_HEIGHT)]
    placements = {
        str(slot): {
            "card_id": card_id,
            "effective_cost": card_cost(card_id),
            "form_code": 0,
            "shape": [GRID_HEIGHT, GRID_WIDTH],
            "row_major": rows,
        }
        for slot, card_id in enumerate(hand)
    }
    actor = PlayerStateV1(
        owner=owner,
        elixir_exact=elixir,
        elixir_visible=elixir,
        hand=hand,
        deck=STANDARD_DECK,
        cycle=cycle,
        private_state_visible=True,
        metadata={
            "hand_slot_by_card": {str(card_id): slot for slot, card_id in enumerate(hand)},
            "hand_runtime_by_slot": {str(slot): {"form_code": 0} for slot in range(len(hand))},
        },
    )
    opponent = PlayerStateV1(owner=1 - owner)
    return ObservationV1(
        tier=ObservationTier.FAIR,
        owner=owner,
        tick=tick,
        episode_id="review-fixture",
        phase="regulation",
        time=TimeStateV1(elapsed_ms=tick * 50, remaining_ms=max(0, 180_000 - tick * 50), elixir_multiplier=1.0),
        players=tuple(sorted((actor, opponent), key=lambda player: player.owner)),
        towers=_towers(damage),
        action_mask=ActionMaskV1(
            kinds={ActionKind.WAIT.value: True, ActionKind.PLAY_CARD.value: True},
            hand_slots=tuple(card_cost(card_id) <= elixir for card_id in hand),
            placement_masks=placements,
            reasons={"effective_elixir": elixir, "reserved_elixir": 0.0},
        ),
    )


def synthetic_owner_sequence(
    *,
    owner: int = 0,
    frames: int = 12,
    plays: tuple[SyntheticPlay, ...] = (),
    start_elixir: float = 5.0,
    elixir_per_frame: float = 0.09,
    damage_per_frame: float = 0.0,
    untrusted_from_frame: int | None = None,
    sequence_id: str = "",
) -> ILSequenceV4:
    """Tensorize a scripted single-owner timeline into one IL sequence."""

    bundle = production_semantic_bundle()
    tensorizer = UniversalObservationTensorizerV4(
        actor_owner=owner,
        card_catalog=bundle.card_catalog,
        ability_catalog=bundle.ability_catalog,
        entity_archetype_catalog=bundle.entity_archetype_catalog,
        effect_catalog=bundle.effect_catalog,
        card_specs=bundle.card_specs,
        deck=STANDARD_DECK,
        deck_roles={card_id: (False, False) for card_id in STANDARD_DECK},
    )
    plays_by_frame = {play.frame: play for play in plays}
    hand = tuple(STANDARD_DECK[:4])
    cycle = tuple(STANDARD_DECK[4:])
    elixir = start_elixir
    observations = []
    actions = []
    label_usable = []
    for frame in range(frames):
        tick = FIRST_POLICY_DECISION_TICK + frame * POLICY_DECISION_TICKS
        observation = _observation(
            owner=owner, tick=tick, elixir=elixir, hand=hand, cycle=cycle, damage={1 - owner: damage_per_frame * frame}
        )
        if frame == 0:
            tensorizer.start_episode(observation, initial_elixir=None)
        batch = tensorizer.tensorize(observation)
        play = plays_by_frame.get(frame)
        window: tuple[TimedExpertActionV4, ...] = ()
        if play is not None:
            card_id = hand[play.hand_position]
            # An unalignable play names a card that is not in that hand slot,
            # which is how a diverged reconstruction shows up in a real cache.
            wanted = card_id if play.alignable else cycle[0]
            window = (
                TimedExpertActionV4(
                    source_tick=tick + play.delay_ticks,
                    action=ActionV1.play(owner, play.hand_position, play.native_cell, card_id=wanted),
                ),
            )
        usable = True
        try:
            sequence = build_expert_action_batch(
                batch,
                (window,),
                decision_ticks_by_row=(tick,),
                perspectives=(tensorizer.perspective,),
                config=tensorizer.config,
            )
        except ExpertActionAlignmentError:
            usable = False
            sequence = build_expert_action_batch(
                batch,
                ((),),
                decision_ticks_by_row=(tick,),
                perspectives=(tensorizer.perspective,),
                config=tensorizer.config,
            )
        tensorizer.record_action(sequence, batch, row=0, validate=False)
        observations.append(batch.to_storage("cpu", float_dtype=torch.float16))
        actions.append(sequence)
        label_usable.append(usable)
        elixir = min(10.0, elixir + elixir_per_frame)
        if play is not None and usable:
            played = hand[play.hand_position]
            elixir -= card_cost(played)
            hand = hand[: play.hand_position] + (cycle[0],) + hand[play.hand_position + 1 :]
            cycle = cycle[1:] + (played,)
    tensorizer.end_episode()
    trusted = frames if untrusted_from_frame is None else untrusted_from_frame
    valid_mask = torch.tensor([[frame < trusted] for frame in range(frames)], dtype=torch.bool)
    return ILSequenceV4(
        observations=tuple(observations),
        actions=tuple(actions),
        episode_start=torch.tensor([[frame == 0] for frame in range(frames)], dtype=torch.bool),
        valid_mask=valid_mask,
        returns=torch.linspace(0.1, 1.0, frames)[:, None],
        gate_loss_mask=torch.tensor([[item] for item in label_usable], dtype=torch.bool) & valid_mask,
        value_loss_mask=valid_mask.clone(),
        sequence_id=sequence_id or f"fixture:owner-{owner}",
        natural_time=True,
    )


def synthetic_result(
    replay_tag: str,
    owner0: ILSequenceV4,
    owner1: ILSequenceV4,
    *,
    completed: bool = True,
    failure_reason: str | None = None,
    first_untrusted_tick: int | None = None,
    winner_matches: bool | None = True,
    crowns_match: bool | None = True,
    expert_action_count: int = 0,
    executed_expert_action_count: int | None = None,
) -> ReplayProductionResultV4:
    end_tick = FIRST_POLICY_DECISION_TICK + owner0.time_steps * POLICY_DECISION_TICKS
    return ReplayProductionResultV4(
        replay_tag=replay_tag,
        sequences=(owner0, owner1),
        completed=completed,
        failure_reason=failure_reason,
        first_untrusted_tick=first_untrusted_tick,
        simulated_end_tick=end_tick,
        source_end_tick=end_tick,
        expert_action_count=expert_action_count,
        executed_expert_action_count=(
            expert_action_count if executed_expert_action_count is None else executed_expert_action_count
        ),
        winner_matches=winner_matches,
        crowns_match=crowns_match,
        decision_frame_count=owner0.time_steps,
    )


def scripted_review(*, seed: int = 11, seconds: int = 180, partial: bool = False) -> dict:
    """Build a full review document from an invented three-minute match.

    Nothing here comes from a model or an engine.  The numbers are scripted so
    that every part of the report page has something to draw: swings, missed
    windows, disputed plays, leaked elixir, unscored actions and tower damage.
    """

    import math
    import random

    from native_runner.review.report import MatchContext, PlayerContext, build_review
    from native_runner.review.scoring import (
        CardOption,
        CellProbability,
        FrameScore,
        ModelChoice,
        PlayedAction,
        TowerHitpoints,
    )

    frame_count = (seconds * 20 - FIRST_POLICY_DECISION_TICK) // POLICY_DECISION_TICKS
    trusted_frames = frame_count * 3 // 4 if partial else frame_count
    frames_by_owner: dict[int, list[FrameScore]] = {}
    contexts: dict[int, PlayerContext] = {}
    for owner in (0, 1):
        rng = random.Random(seed + owner)
        sign = 1.0 if owner == 0 else -1.0
        hand = list(STANDARD_DECK[:4])
        cycle = list(STANDARD_DECK[4:])
        elixir = 5.0
        next_play = rng.randint(8, 20)
        frames: list[FrameScore] = []
        source_actions: list[tuple[int, int | None]] = []
        for index in range(frame_count):
            tick = FIRST_POLICY_DECISION_TICK + index * POLICY_DECISION_TICKS
            time = tick / 20.0
            multiplier = 2.0 if time >= 120.0 else 1.0
            # One slow drift, one sharp loss around 1:10 and a recovery near the end.
            value = sign * (
                0.12 * math.sin(time / 21.0)
                - 0.45 / (1.0 + math.exp(-(time - 70.0) / 2.5))
                + 0.6 / (1.0 + math.exp(-(time - 150.0) / 4.0))
            ) + rng.uniform(-0.01, 0.01)
            waiting = 40.0 <= time <= 46.0 or 100.0 <= time <= 103.0
            act_probability = 0.9 if waiting else min(0.45, 0.04 * elixir) + rng.uniform(0.0, 0.03)
            options = tuple(
                sorted(
                    (
                        CardOption("play_card", card_id, "normal", card_cost(card_id), probability)
                        for card_id, probability in zip(hand, (0.52, 0.27, 0.14, 0.07), strict=True)
                    ),
                    key=lambda option: -option.probability,
                )
            )
            best = options[0]
            choice_cell = (3 if rng.random() < 0.5 else 14, rng.randint(8, 14))
            model_choice = ModelChoice(
                "play_card", best.card_id, "normal", best.cost, best.probability, choice_cell, 0.3
            )
            played: tuple[PlayedAction, ...] = ()
            unusable = index in (frame_count // 3, frame_count // 2) and owner == 0
            if index == next_play and not waiting and not unusable:
                position = rng.choice((0, 0, 1, 2, 3))
                card_id = options[position].card_id
                if card_cost(card_id) <= elixir:
                    center = (3 if rng.random() < 0.5 else 14, rng.randint(6, 13))
                    ranked = sorted(
                        ((center[0] + dx, center[1] + dy) for dx in (-1, 0, 1) for dy in (-2, -1, 0, 1)),
                        key=lambda cell: abs(cell[0] - center[0]) + abs(cell[1] - center[1]),
                    )
                    weights = [0.34 * 0.72**rank for rank in range(len(ranked))]
                    top_cells = tuple(
                        CellProbability(x, y, weight) for (x, y), weight in zip(ranked, weights, strict=True)
                    )
                    rank = rng.choice((1, 1, 2, 4, 9, 40))
                    cell = ranked[rank - 1] if rank <= len(ranked) else (rng.randint(1, 16), rng.randint(2, 14))
                    played = (
                        PlayedAction(
                            kind="play_card",
                            card_id=card_id,
                            form="normal",
                            cost=card_cost(card_id),
                            tick=tick + rng.randint(0, 4),
                            cell=cell,
                            card_probability=options[position].probability,
                            cell_probability=weights[rank - 1] if rank <= len(ranked) else 0.002,
                            cell_rank=rank,
                            legal_cell_count=270,
                            model_cell=ranked[0],
                            model_cell_probability=weights[0],
                            top_cells=top_cells,
                        ),
                    )
                    source_actions.append((played[0].tick, card_id))
                    act_probability = rng.choice((0.82, 0.74, 0.61, 0.33, 0.12))
                next_play = index + rng.randint(18, 34)
            elif index == next_play:
                next_play = index + 4
            if unusable:
                source_actions.append((tick + 2, hand[0]))
            enemy_damage = 900.0 * sum(
                time >= moment for moment in ((60.0, 150.0, 165.0) if owner == 0 else (72.0, 78.0))
            )
            own_damage = 900.0 * sum(
                time >= moment for moment in ((72.0, 78.0) if owner == 0 else (60.0, 150.0, 165.0))
            )
            frames.append(
                FrameScore(
                    index=index,
                    tick=tick,
                    valid=index < trusted_frames,
                    label_usable=not unusable and index < trusted_frames,
                    value=value,
                    realized_return=None,
                    act_probability=act_probability,
                    elixir=elixir,
                    elixir_multiplier=multiplier,
                    remaining_seconds=max(0.0, seconds - time),
                    overtime=False,
                    own_towers=TowerHitpoints(4800.0, max(0.0, 3000.0 - own_damage), 3000.0),
                    enemy_towers=TowerHitpoints(4800.0, 3000.0, max(0.0, 3000.0 - enemy_damage)),
                    options=options,
                    played=played,
                    model_choice=model_choice,
                )
            )
            elixir = min(10.0, elixir + 0.25 * multiplier / 2.8)
            if played:
                elixir -= played[0].cost
                slot = hand.index(played[0].card_id)
                cycle.append(hand[slot])
                hand[slot] = cycle.pop(0)
        frames_by_owner[owner] = frames
        contexts[owner] = PlayerContext(
            side="team" if owner == 0 else "opponent",
            result=None if partial else ("win" if owner == 0 else "loss"),
            crowns=None if partial else (1 if owner == 0 else 0),
            source_elixir_leaked=1.8 if owner == 0 else 4.1,
            source_actions=tuple(source_actions),
        )
    fidelity = {
        "completed": not partial,
        "failure_reason": "battle truncated before source timeline" if partial else None,
        "first_untrusted_tick": (
            FIRST_POLICY_DECISION_TICK + trusted_frames * POLICY_DECISION_TICKS if partial else None
        ),
        "expert_action_count": sum(len(context.source_actions or ()) for context in contexts.values()),
        "executed_expert_action_count": sum(
            len(frame.played) for frames in frames_by_owner.values() for frame in frames if frame.valid
        ),
        "winner_matches": None if partial else True,
        "crowns_match": None if partial else True,
    }
    return build_review(
        replay_tag="0" * 32,
        model={"checkpoint_file": "scripted-example.pt"},
        fidelity=fidelity,
        frames_by_owner=frames_by_owner,
        match=MatchContext(game_mode="Scripted example", duration_seconds=float(seconds), players=contexts),
    )
