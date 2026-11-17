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
