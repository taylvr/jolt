from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from ..core import plan_tree as pt

class Scorer:
    def init_state(self, smp) -> Any:
        return None

    def step_logits(self, smp, state, chosen: List[str], legal: List[str]) -> List[float]:
        raise NotImplementedError

    def advance(self, smp, state, chosen: List[str], picked: str) -> Any:
        return state

    def side_logit(self, smp, state, chosen: List[str], picked: str) -> float:
        return -1.0

def log_softmax(xs: List[float]) -> List[float]:
    m = max(xs)
    s = sum(math.exp(x - m) for x in xs)
    return [x - m - math.log(s) for x in xs]

@dataclass
class Beam:
    seq: List[str]
    sides: List[int]
    logp: float
    state: Any

    @property
    def tree(self) -> pt.JoinTree:
        return pt.seq_to_tree(self.seq, self.sides)

def beam_search(scorer: Scorer, smp, beam_size: int = 1) -> List[Beam]:
    beams = [Beam([], [], 0.0, scorer.init_state(smp))]
    for k in range(smp.n):
        new_beams: List[Beam] = []
        for b in beams:
            legal = smp.legal_candidates(b.seq)
            logits = scorer.step_logits(smp, b.state, b.seq, legal)
            lps = log_softmax(logits)
            for c, lp in zip(legal, lps):
                side = 0
                if k >= 2:
                    side = 1 if scorer.side_logit(smp, b.state, b.seq, c) > 0 else 0
                new_beams.append(Beam(b.seq + [c], b.sides + [side], b.logp + lp, None))
                new_beams[-1].state = (b, c)
        new_beams.sort(key=lambda x: -x.logp)
        new_beams = new_beams[:beam_size]
        for nb in new_beams:
            parent, c = nb.state
            nb.state = scorer.advance(smp, parent.state, parent.seq, c)
        beams = new_beams
    return beams

def teacher_forced_step_acc(scorer: Scorer, smp) -> Tuple[int, int]:
    state = scorer.init_state(smp)
    correct = total = 0
    for k, tgt in enumerate(smp.target_seq):
        legal = smp.legal_candidates(smp.target_seq[:k])
        if len(legal) > 1:
            logits = scorer.step_logits(smp, state, smp.target_seq[:k], legal)
            pred = legal[max(range(len(legal)), key=lambda i: logits[i])]
            correct += int(pred == tgt)
            total += 1
        state = scorer.advance(smp, state, smp.target_seq[:k], tgt)
    return correct, total
