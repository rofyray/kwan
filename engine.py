"""Kwan core: Elo-style learner model + ZPD frontier recommender with rationale and a Ghana-context setting."""
import math, operator, time, networkx as nx
from dataclasses import asdict, dataclass, field
from context import practice_setting

MASTERY = 0.85     # p(correct) at/above which an indicator counts as mastered
READY = 0.70       # min prerequisite mastery for a node to be "ready"

def sigmoid(z):
    try:
        return 1 / (1 + math.exp(-z))
    except OverflowError:                               # z below about -709: the probability is 0 to machine precision
        return 0.0

@dataclass
class LearnerState:
    learner_id: str
    theta: float = 0.0                                  # global ability
    theta_k: dict = field(default_factory=dict)         # per-indicator ability offset
    n: dict = field(default_factory=dict)               # attempts per indicator (+ "_all")
    last_seen: dict = field(default_factory=dict)       # indicator -> step index
    streak: dict = field(default_factory=dict)          # consecutive correct answers per indicator
    step: int = 0

    def to_dict(self) -> dict:
        """Plain-dict copy of the state. It survives a JSON round trip unchanged, so a platform
        such as Syto can keep it in any store and rebuild it with `from_dict`."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "LearnerState":
        """Rebuild a state written by `to_dict`. Only `learner_id` is required; keys this version
        does not know are ignored. The dictionaries are copied, never shared with `d`."""
        if "learner_id" not in d:
            raise ValueError("learner state needs a 'learner_id'")
        return cls(
            learner_id=d["learner_id"],
            theta=float(d.get("theta", 0.0)),
            theta_k={k: float(v) for k, v in d.get("theta_k", {}).items()},
            n={k: int(v) for k, v in d.get("n", {}).items()},
            last_seen={k: int(v) for k, v in d.get("last_seen", {}).items()},
            streak={k: int(v) for k, v in d.get("streak", {}).items()},
            step=int(d.get("step", 0)),
        )

class EloLearner:
    """p(correct | learner, indicator, item) = sigmoid(theta + theta_k - beta_item). Validated on ASSISTments 2009 (AUC 0.75)."""
    def __init__(self, k_s=0.3, k_k=0.4, decay=0.05, item_difficulty=None, G=None,
                 grade_step=0.4, transfer=0.5, streak_to_master=3):
        self.k_s, self.k_k, self.decay = k_s, k_k, decay
        self.beta = item_difficulty if item_difficulty is not None else {}
        self.G, self.grade_step, self.transfer, self.streak_to_master = G, grade_step, transfer, streak_to_master
    def _offset(self, st, ind):
        """Per-indicator offset; for never-seen indicators, borrow evidence from prerequisites (graph transfer)
        and apply a grade-difficulty prior so B6 skills are not assumed known because B1 skills are."""
        if ind in st.theta_k: return st.theta_k[ind]
        if self.G is None: return 0.0
        if ind not in self.G: raise KeyError(f"unknown indicator {ind!r}: not in the curriculum graph")
        g = self.G.nodes[ind]["grade"]
        preds = [st.theta_k[p] for p in self.G.predecessors(ind) if p in st.theta_k]
        return -self.grade_step * (g - 1) + (self.transfer * sum(preds) / len(preds) if preds else -0.5)
    def p_correct(self, st, ind, item=None):
        return sigmoid(st.theta + self._offset(st, ind) - self.beta.get(item, 0.0))
    def mastery(self, st, ind):
        if st.streak.get(ind, 0) >= self.streak_to_master: return max(0.95, self.p_correct(st, ind))
        return self.p_correct(st, ind)                 # mastery on a typical (beta=0) item
    def update(self, st, ind, correct, item=None):
        """Apply one answer and return the probability predicted before seeing it.
        With a graph, an indicator that is not in it raises KeyError and leaves the state untouched."""
        if self.G is not None and ind not in self.G:
            raise KeyError(f"unknown indicator {ind!r}: not in the curriculum graph")
        correct = bool(correct)
        p = self.p_correct(st, ind, item); e = int(correct) - p
        na, nk = st.n.get("_all", 0), st.n.get(ind, 0)
        st.theta += self.k_s / (1 + self.decay * na) * e
        st.theta_k[ind] = self._offset(st, ind) + self.k_k / (1 + self.decay * nk) * e
        st.streak[ind] = st.streak.get(ind, 0) + 1 if correct else 0
        st.n["_all"], st.n[ind] = na + 1, nk + 1
        st.step += 1; st.last_seen[ind] = st.step
        return p

class Recommender:
    W = dict(readiness=0.30, zpd=0.25, goal=0.25, unlock=0.15, recency=0.05)
    def __init__(self, G: nx.DiGraph, learner: EloLearner):
        self.G, self.L = G, learner
        self.desc_count = {n: len(nx.descendants(G, n)) for n in G}
        self.max_desc = max(self.desc_count.values(), default=0) or 1
    def _goal_dist(self, goal):
        if not goal: return {}
        if goal not in self.G: raise KeyError(f"unknown goal {goal!r}: not in the curriculum graph")
        rev = self.G.reverse(copy=False)
        return nx.single_source_shortest_path_length(rev, goal)   # ancestors of goal -> hops
    def recommend(self, st, k=3, goal=None, grade_max=None):
        """Top-k next skills for a learner, best first.

        k: how many to return at most (0 gives an empty list, a negative k raises ValueError).
        goal: an indicator id to steer towards, or None/"" for no goal (an unknown id raises KeyError).
        grade_max: class cap, nothing above this grade is returned (None for no cap).
        Equal scores are ordered by indicator id, highest id first, so the result is deterministic.
        """
        t0 = time.perf_counter()
        try:
            k = operator.index(k)
        except TypeError:
            raise TypeError(f"k must be an integer, got {k!r}") from None
        if k < 0: raise ValueError(f"k must be 0 or more, got {k}")
        m = {n: self.L.mastery(st, n) for n in self.G}
        gd = self._goal_dist(goal)
        cands = []
        for n, d in self.G.nodes(data=True):
            if m[n] >= MASTERY or (grade_max is not None and d["grade"] > grade_max): continue
            preds = list(self.G.predecessors(n))
            ready = min((m[p] for p in preds), default=1.0)
            if ready < READY: continue
            zpd = 1 - abs(m[n] - 0.6) / 0.6                       # peak when ~60% likely correct
            goal_term = 1 / (1 + gd[n]) if n in gd else (0.0 if gd else 0.5)
            unlock = self.desc_count[n] / self.max_desc
            rec = 1 - math.exp(-(st.step - st.last_seen.get(n, -99)) / 5)
            s = (self.W["readiness"] * ready + self.W["zpd"] * zpd + self.W["goal"] * goal_term
                 + self.W["unlock"] * unlock + self.W["recency"] * rec)
            cands.append((s, n, preds, m[n]))
        cands.sort(key=lambda c: (c[0], c[1]), reverse=True)     # score, then indicator id: ids are unique
        out = [{
            "indicator": n, "score": round(s, 4), "mastery": round(mn, 3),
            "grade": self.G.nodes[n]["grade"], "strand": self.G.nodes[n]["strand_name"],
            "description": self.G.nodes[n]["description"],
            "rationale": self._why(n, preds, m, mn, gd),
            "ghana_context": practice_setting(n, self.G.nodes[n]["strand"], self.G.nodes[n]["grade"]),
        } for s, n, preds, mn in cands[:k]]
        return {"recommendations": out, "latency_ms": round((time.perf_counter() - t0) * 1000, 3)}
    def _why(self, n, preds, m, mn, gd):
        """Plain-language reason built only from the signals that produced the score."""
        parts = []
        if preds:
            best = max(preds, key=lambda p: m[p])
            parts.append(f"Builds on {best}, where you are at {m[best]:.0%}.")
        else:
            parts.append("A starting skill with nothing before it.")
        if mn < 0.5:
            parts.append(f"Still new to you ({mn:.0%}), so expect a few tries.")
        elif mn < 0.75:
            parts.append(f"You are at {mn:.0%}, close enough to master with practice.")
        else:
            parts.append(f"Nearly there at {mn:.0%}. A few correct answers will lock it in.")
        if n in gd and gd[n] > 0:
            parts.append(f"On the way to your goal, {gd[n]} step(s) away.")
        if self.desc_count[n]:
            parts.append(f"Opens the way to {self.desc_count[n]} later skills.")
        return " ".join(parts)
