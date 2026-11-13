from __future__ import annotations

import pickle
from typing import List, Tuple

import numpy as np

from . import candidate_features as F
from .beam_search import Scorer

def _row(smp, c: str, step_vec: List[float]) -> List[float]:
    return F.leaf_vector(smp, c) + [float(x) for x in F.leaf_cat_ids(smp, c)] + step_vec

def feature_names() -> List[str]:
    return F.LEAF_NUM_FEATURES + ["cat_" + c for c in F.LEAF_CAT_FEATURES] + F.STEP_FEATURE_NAMES

def categorical_mask() -> List[bool]:
    names = feature_names()
    return [n.startswith("cat_") for n in names]

def build_step_rows(samples) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Xp, yp, gid, Xs, ys = [], [], [], [], []
    g = 0
    for smp in samples:
        seq, sides = smp.target_seq, smp.target_sides
        for k, tgt in enumerate(seq):
            chosen = seq[:k]
            legal = smp.legal_candidates(chosen)
            sf = F.step_features(smp, chosen, legal)
            if tgt not in sf:
                continue
            if len(legal) > 1:
                for c in legal:
                    Xp.append(_row(smp, c, sf[c]))
                    yp.append(int(c == tgt))
                    gid.append(g)
                g += 1
            if k >= 2:
                Xs.append(_row(smp, tgt, sf[tgt]))
                ys.append(int(sides[k]))
    return (np.asarray(Xp, dtype=np.float64), np.asarray(yp), np.asarray(gid),
            np.asarray(Xs, dtype=np.float64), np.asarray(ys))

def build_step_rows_weighted(samples, weighting):
    from .jo_weights import normalize_mean_one
    Xp, yp, gid, Xs, ys, wp, ws = [], [], [], [], [], [], []
    g = 0
    for smp in samples:
        seq, sides = smp.target_seq, smp.target_sides
        for k, tgt in enumerate(seq):
            chosen = seq[:k]
            legal = smp.legal_candidates(chosen)
            sf = F.step_features(smp, chosen, legal)
            if tgt not in sf:
                continue
            if len(legal) > 1:
                wmap = weighting.pick_weights(smp, k, chosen, legal, tgt)
                for c in legal:
                    Xp.append(_row(smp, c, sf[c]))
                    yp.append(int(c == tgt))
                    gid.append(g)
                    wp.append(float(wmap[c]))
                g += 1
            if k >= 2:
                Xs.append(_row(smp, tgt, sf[tgt]))
                ys.append(int(sides[k]))
                ws.append(float(weighting.side_weight(smp, k, chosen, legal, tgt)))
    return (np.asarray(Xp, dtype=np.float64), np.asarray(yp), np.asarray(gid),
            np.asarray(Xs, dtype=np.float64), np.asarray(ys),
            normalize_mean_one(wp), normalize_mean_one(ws))

class GBDTScorer(Scorer):
    def __init__(self, max_iter: int = 300, learning_rate: float = 0.05, max_leaf_nodes: int = 15,
                 min_samples_leaf: int = 20, l2: float = 1.0, seed: int = 0):
        from sklearn.ensemble import HistGradientBoostingClassifier
        kw = dict(max_iter=max_iter, learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes,
                  min_samples_leaf=min_samples_leaf, l2_regularization=l2, random_state=seed,
                  categorical_features=categorical_mask())
        self.pick = HistGradientBoostingClassifier(**kw)
        self.side = HistGradientBoostingClassifier(**{**kw, "max_iter": max(50, max_iter // 3)})
        self.side_const = None

    def fit(self, samples, dagger_rounds: int = 0, weighting=None):
        if weighting is not None and getattr(weighting, "mode", "none") != "none":
            return self._fit_weighted(samples, dagger_rounds, weighting)
        Xp, yp, _, Xs, ys = build_step_rows(samples)
        self.pick.fit(Xp, yp)
        for _ in range(dagger_rounds):
            Xd, yd = self._dagger_rows(samples)
            if len(Xd):
                Xp, yp = np.vstack([Xp, Xd]), np.concatenate([yp, yd])
                self.pick.fit(Xp, yp)
        if len(set(ys.tolist())) > 1:
            self.side.fit(Xs, ys)
        else:
            self.side_const = int(ys[0]) if len(ys) else 0
        return self

    def _fit_weighted(self, samples, dagger_rounds, weighting):
        Xp, yp, _, Xs, ys, wp, ws = build_step_rows_weighted(samples, weighting)
        self.pick.fit(Xp, yp, sample_weight=wp)
        for _ in range(dagger_rounds):
            Xd, yd = self._dagger_rows(samples)
            if len(Xd):
                Xp, yp = np.vstack([Xp, Xd]), np.concatenate([yp, yd])
                wp = np.concatenate([wp, np.ones(len(yd))])
                self.pick.fit(Xp, yp, sample_weight=wp)
        if len(set(ys.tolist())) > 1:
            self.side.fit(Xs, ys, sample_weight=ws)
        else:
            self.side_const = int(ys[0]) if len(ys) else 0
        self.weighting_desc = weighting.describe()
        return self

    def _dagger_rows(self, samples):
        X, y = [], []
        for smp in samples:
            chosen: List[str] = []
            for _k in range(smp.n):
                legal = smp.legal_candidates(chosen)
                oracle = [t for t in smp.target_seq if t not in chosen and t in legal][0]
                if len(legal) > 1:
                    sf = F.step_features(smp, chosen, legal)
                    if chosen != smp.target_seq[:len(chosen)]:
                        for c in legal:
                            X.append(_row(smp, c, sf[c]))
                            y.append(int(c == oracle))
                    lg = self.step_logits(smp, None, chosen, legal)
                    pick = legal[int(np.argmax(lg))]
                else:
                    pick = legal[0]
                chosen = chosen + [pick]
        return np.asarray(X, dtype=np.float64), np.asarray(y)

    def step_logits(self, smp, state, chosen, legal):
        sf = F.step_features(smp, chosen, legal)
        X = np.asarray([_row(smp, c, sf[c]) for c in legal], dtype=np.float64)
        return self.pick.decision_function(X).reshape(-1).tolist()

    def side_logit(self, smp, state, chosen, picked):
        if self.side_const is not None:
            return 1.0 if self.side_const == 1 else -1.0
        legal = smp.legal_candidates(chosen)
        sf = F.step_features(smp, chosen, legal if picked in legal else legal + [picked])
        X = np.asarray([_row(smp, picked, sf[picked])], dtype=np.float64)
        return float(self.side.decision_function(X)[0])

    def feature_importance(self, samples, n_repeats: int = 3, seed: int = 0):
        from sklearn.inspection import permutation_importance
        Xp, yp, _, _, _ = build_step_rows(samples)
        r = permutation_importance(self.pick, Xp, yp, n_repeats=n_repeats, random_state=seed,
                                   scoring="neg_log_loss")
        names = feature_names()
        return sorted(zip(names, r.importances_mean), key=lambda x: -x[1])

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "GBDTScorer":
        with open(path, "rb") as f:
            return pickle.load(f)
