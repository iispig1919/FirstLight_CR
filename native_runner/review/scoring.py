"""Score one reconstructed owner sequence with a V4 policy.

Imitation learning already teacher-forces a human player's actions through the
recurrent model.  A review is that same pass without a loss: at every five-tick
decision the policy reports how it values the position, how willing it is to
act, and how it ranks the legal cards and cells.  The recurrent state always
follows what the player actually did, so every frame is judged in the position
the player really faced.

All probabilities use temperature 1, the model's own distribution.  Cells are
``(x, y)`` in the reviewed player's view: ``x`` runs 0..17 left to right and
``y`` runs 0..31 from the player's own king tower to the enemy's.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from ..training.v4.config import MATCH_SCALAR_NAMES, TOWER_FEATURE_NAMES
from ..training.v4.decoding import ShadowCandidateLegality
from ..training.v4.expert import FIRST_POLICY_DECISION_TICK, NATIVE_TICK_MS, POLICY_DECISION_TICKS
from ..training.v4.imitation import ILSequenceV4
from ..training.v4.model import UniversalCardPolicyV4, _distribution
from ..training.v4.tensorizer import (
    FORM_EVOLUTION,
    FORM_HERO,
    FORM_HERO_ACTIVE,
    FORM_NORMAL,
    OWNER_ENEMY,
    OWNER_SELF,
    TOWER_TYPE,
)
from ..training.v4.tensors import (
    CANDIDATE_ABILITY,
    CANDIDATE_DEPLOY,
    GATE_ACT,
    TARGET_GRID,
    ActionEvaluationComponentsV4,
    ActionSequenceV4,
    PolicyContextV4,
    UniversalSemanticBatchV4,
    concatenate_padded_tensor_records,
)


REVIEW_TEMPERATURE = 1.0
TOP_CELL_COUNT = 12
# Observation encoding dominates the cost and batches well; the recurrent core
# and the decoders still advance one frame at a time.
ENCODE_CHUNK_FRAMES = 32
ACTION_KIND = {CANDIDATE_DEPLOY: "play_card", CANDIDATE_ABILITY: "activate_ability"}
FORM_LABEL = {FORM_NORMAL: "normal", FORM_EVOLUTION: "evolution", FORM_HERO: "hero", FORM_HERO_ACTIVE: "hero_active"}
TOWER_KINDS = tuple(sorted(TOWER_TYPE, key=TOWER_TYPE.__getitem__))

_REMAINING_TIME = MATCH_SCALAR_NAMES.index("remaining_time_over_300000ms")
_ELIXIR_MULTIPLIER = MATCH_SCALAR_NAMES.index("elixir_multiplier_over_3")
_OVERTIME = MATCH_SCALAR_NAMES.index("overtime_or_sudden_death")
_OWN_ELIXIR = MATCH_SCALAR_NAMES.index("own_elixir_over_10")
_TOWER_HITPOINTS = TOWER_FEATURE_NAMES.index("hitpoints_over_5000")


@dataclass(frozen=True, slots=True)
class CardOption:
    """One legal candidate and the probability the policy gives it."""

    kind: str
    card_id: int
    form: str
    cost: float
    probability: float


@dataclass(frozen=True, slots=True)
class CellProbability:
    x: int
    y: int
    probability: float


@dataclass(frozen=True, slots=True)
class PlayedAction:
    """One action the player executed, with the policy's opinion of it.

    Placement detail (rank, the policy's own cell, top cells) is only available
    for the first action of a decision window; a second action in the same
    quarter second keeps its exact probabilities but no heat map.
    """

    kind: str
    card_id: int
    form: str
    cost: float
    tick: int
    cell: tuple[int, int] | None
    card_probability: float
    cell_probability: float | None
    cell_rank: int | None = None
    legal_cell_count: int | None = None
    model_cell: tuple[int, int] | None = None
    model_cell_probability: float | None = None
    top_cells: tuple[CellProbability, ...] = ()


@dataclass(frozen=True, slots=True)
class ModelChoice:
    """The card and cell the policy prefers if it has to act on this frame."""

    kind: str
    card_id: int
    form: str
    cost: float
    card_probability: float
    cell: tuple[int, int] | None
    cell_probability: float | None


@dataclass(frozen=True, slots=True)
class TowerHitpoints:
    king: float = 0.0
    princess_left: float = 0.0
    princess_right: float = 0.0

    @property
    def total(self) -> float:
        return self.king + self.princess_left + self.princess_right


@dataclass(frozen=True, slots=True)
class FrameScore:
    """Everything the review knows about one decision frame."""

    index: int
    tick: int
    valid: bool
    label_usable: bool
    value: float
    realized_return: float | None
    act_probability: float
    elixir: float
    elixir_multiplier: float
    remaining_seconds: float
    overtime: bool
    own_towers: TowerHitpoints
    enemy_towers: TowerHitpoints
    options: tuple[CardOption, ...]
    played: tuple[PlayedAction, ...]
    model_choice: ModelChoice | None

    @property
    def time_seconds(self) -> float:
        return self.tick * NATIVE_TICK_MS / 1000.0


def _cell_xy(cell: int, width: int) -> tuple[int, int]:
    return cell % width, cell // width


def _towers(batch: UniversalSemanticBatchV4, owner_type: int) -> TowerHitpoints:
    towers = batch.towers
    hitpoints = {kind: 0.0 for kind in TOWER_KINDS}
    for row in range(int(towers.mask.shape[1])):
        if not bool(towers.mask[0, row]) or int(towers.owner_type[0, row]) != owner_type:
            continue
        kind = TOWER_KINDS[int(towers.tower_type[0, row])]
        hitpoints[kind] = max(0.0, float(towers.features[0, row, _TOWER_HITPOINTS]) * 5000.0)
    return TowerHitpoints(**hitpoints)


class _FrameDecoder:
    """First-action card and cell distributions for one frame's context."""

    def __init__(self, model: UniversalCardPolicyV4, batch: UniversalSemanticBatchV4, context: PolicyContextV4) -> None:
        self.model = model
        self.batch = batch
        self.context = context
        self.shadow = ShadowCandidateLegality(batch.candidates, model.config)
        self.legal = self.shadow.candidate_mask()
        self.any_legal = bool(self.legal.any())
        self._decoder_context: Tensor | None = None
        self._card_probabilities: Tensor | None = None

    @property
    def decoder_context(self) -> Tensor:
        if self._decoder_context is None:
            base = self.model.decoder_initial(self.context.policy_context)
            self._decoder_context = self.model._decoder_context(base, self.context.encoded)
        return self._decoder_context

    @property
    def card_probabilities(self) -> Tensor:
        if self._card_probabilities is None:
            logits = self.model.candidate_logits(self.decoder_context, self.context.encoded.candidate_memory)
            self._card_probabilities = _distribution(
                logits, mask=self.legal, temperature=REVIEW_TEMPERATURE, validate=False
            ).probs[0]
        return self._card_probabilities

    def is_grid(self, row: int) -> bool:
        return int(self.batch.candidates.target_mode[0, row]) == TARGET_GRID

    def cell_probabilities(self, row: int) -> tuple[Tensor, Tensor]:
        """Return the cell distribution and legality mask for one candidate."""

        index = torch.tensor([row], dtype=torch.long, device=self.legal.device)
        memory = self.model._selected_candidate_memory(self.context.encoded.candidate_memory, index)
        logits, _keys = self.model._location_logits(self.decoder_context, memory, self.context.encoded.spatial_memory)
        placement = self.shadow.selected_placement(index).flatten(1)
        distribution = _distribution(logits.flatten(1), mask=placement, temperature=REVIEW_TEMPERATURE, validate=False)
        return distribution.probs[0], placement[0]

    def describe(self, row: int) -> tuple[str, int, str, float]:
        candidates = self.batch.candidates
        return (
            ACTION_KIND[int(candidates.variant[0, row])],
            int(candidates.native_visible_card_id[0, row]),
            FORM_LABEL.get(int(candidates.effective_form[0, row]), "normal"),
            float(candidates.cost[0, row]),
        )

    def options(self) -> tuple[CardOption, ...]:
        if not self.any_legal:
            return ()
        probabilities = self.card_probabilities
        rows = self.legal[0].nonzero(as_tuple=False).flatten().tolist()
        result = [CardOption(*self.describe(row), probability=float(probabilities[row])) for row in rows]
        return tuple(sorted(result, key=lambda item: -item.probability))

    def model_choice(self) -> ModelChoice | None:
        if not self.any_legal:
            return None
        probabilities = self.card_probabilities
        row = int(probabilities.argmax())
        cell: tuple[int, int] | None = None
        cell_probability: float | None = None
        if self.is_grid(row):
            cells, _legal = self.cell_probabilities(row)
            best = int(cells.argmax())
            cell = _cell_xy(best, self.model.config.board_width)
            cell_probability = float(cells[best])
        return ModelChoice(
            *self.describe(row),
            card_probability=float(probabilities[row]),
            cell=cell,
            cell_probability=cell_probability,
        )


def _played_actions(
    decoder: _FrameDecoder, actions: ActionSequenceV4, components: ActionEvaluationComponentsV4, *, decision_tick: int
) -> tuple[PlayedAction, ...]:
    config = decoder.model.config
    width = config.board_width
    resolved = decoder.model._resolve_candidate_indices(decoder.batch, actions)
    result: list[PlayedAction] = []
    for step in range(int(actions.micro_action_count[0])):
        row = int(resolved[0, step])
        target = int(actions.target_cell[0, step])
        delay_ms = config.delay_offset_ms[int(actions.delay_offset_bin[0, step])]
        grid = target >= 0
        cell_rank = legal_cell_count = model_cell = model_cell_probability = None
        top_cells: tuple[CellProbability, ...] = ()
        if step == 0 and grid:
            cells, legal = decoder.cell_probabilities(row)
            best = int(cells.argmax())
            top = torch.topk(cells, k=min(TOP_CELL_COUNT, int(legal.sum())))
            cell_rank = int((cells[legal] > cells[target]).sum()) + 1
            legal_cell_count = int(legal.sum())
            model_cell = _cell_xy(best, width)
            model_cell_probability = float(cells[best])
            top_cells = tuple(
                CellProbability(*_cell_xy(int(cell), width), probability=float(probability))
                for probability, cell in zip(top.values.tolist(), top.indices.tolist(), strict=True)
            )
        result.append(
            PlayedAction(
                *decoder.describe(row),
                tick=decision_tick + delay_ms // NATIVE_TICK_MS,
                cell=_cell_xy(target, width) if grid else None,
                card_probability=math.exp(float(components.candidate_log_prob[0, step])),
                cell_probability=math.exp(float(components.target_log_prob[0, step])) if grid else None,
                cell_rank=cell_rank,
                legal_cell_count=legal_cell_count,
                model_cell=model_cell,
                model_cell_probability=model_cell_probability,
                top_cells=top_cells,
            )
        )
    return tuple(result)


def score_sequence(
    model: UniversalCardPolicyV4,
    sequence: ILSequenceV4,
    *,
    device: torch.device | str = "cpu",
    first_decision_tick: int = FIRST_POLICY_DECISION_TICK,
) -> tuple[FrameScore, ...]:
    """Teacher-force one owner's natural-time sequence and score every frame."""

    sequence.validate()
    if sequence.batch_size != 1:
        raise ValueError("review scoring needs one uncollated owner sequence")
    if not sequence.natural_time:
        raise ValueError("review scoring needs a natural-time sequence")
    actual_device = torch.device(device)
    model.eval()
    state = model.initial_state(1, device=actual_device)
    gate_loss_mask = sequence.valid_mask if sequence.gate_loss_mask is None else sequence.gate_loss_mask
    frames: list[FrameScore] = []
    with torch.inference_mode():
        for chunk_start in range(0, sequence.time_steps, ENCODE_CHUNK_FRAMES):
            indices = range(chunk_start, min(sequence.time_steps, chunk_start + ENCODE_CHUNK_FRAMES))
            chunk = concatenate_padded_tensor_records(
                tuple(sequence.observations[index].to_model_input(actual_device) for index in indices)
            )
            encoded = model.encode_observation(chunk, validate=False)
            for offset, index in enumerate(indices):
                batch = chunk.narrow_batch(offset, 1)
                actions = sequence.actions[index].to(actual_device)
                context = model.advance_core(
                    encoded.narrow_batch(offset, 1),
                    state,
                    episode_start=sequence.episode_start[index].to(actual_device),
                )
                state = context.next_state
                forced = model._decode(
                    batch,
                    context,
                    sample=False,
                    forced_actions=actions,
                    gate_temperature=REVIEW_TEMPERATURE,
                    action_temperature=REVIEW_TEMPERATURE,
                    continue_temperature=REVIEW_TEMPERATURE,
                    validate=False,
                )
                components = forced.action_components
                if components is None:
                    raise RuntimeError("V4 action evaluation did not return head components")
                acted = int(actions.gate[0]) == GATE_ACT
                gate_probability = math.exp(float(components.gate_log_prob[0]))
                decoder = _FrameDecoder(model, batch, context)
                tick = first_decision_tick + index * POLICY_DECISION_TICKS
                scalars = batch.match_scalars[0]
                value_known = sequence.value_loss_mask is not None and bool(sequence.value_loss_mask[index, 0])
                frames.append(
                    FrameScore(
                        index=index,
                        tick=tick,
                        valid=bool(sequence.valid_mask[index, 0]),
                        label_usable=bool(gate_loss_mask[index, 0]),
                        value=float(context.value[0]),
                        realized_return=float(sequence.returns[index, 0]) if value_known else None,
                        act_probability=gate_probability if acted else 1.0 - gate_probability,
                        elixir=float(scalars[_OWN_ELIXIR]) * 10.0,
                        elixir_multiplier=float(scalars[_ELIXIR_MULTIPLIER]) * 3.0,
                        remaining_seconds=float(scalars[_REMAINING_TIME]) * 300.0,
                        overtime=bool(scalars[_OVERTIME] > 0.5),
                        own_towers=_towers(batch, OWNER_SELF),
                        enemy_towers=_towers(batch, OWNER_ENEMY),
                        options=decoder.options(),
                        played=_played_actions(decoder, actions, components, decision_tick=tick) if acted else (),
                        model_choice=decoder.model_choice(),
                    )
                )
    return tuple(frames)
