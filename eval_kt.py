"""Learner-model validation on ASSISTments 2009 (skill-builder, corrected).
Models: global-mean, skill-mean, Elo, Elo-multi (Kwan default), BKT (fixed params per skill fit by grid).
Split: 80/20 by student (no student leakage). Metric: AUC, RMSE on test next-response predictions."""
import argparse, sys, math, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, mean_squared_error

KEY = "assistments_2009"                       # top-level key this script owns in data/eval_results.json
DATASET = "ASSISTments 2009 skill-builder (corrected)"
SPLIT = "80/20 by student, seed 42"
KWAN = "elo-multi (Kwan)"                      # results-file name of the model printed as "elo-multi"
MODEL_ORDER = ("global-mean", "skill-mean", "elo", "bkt", KWAN)

def load(path):
    df = pd.read_csv(path, encoding="latin-1", low_memory=False,
                     usecols=["order_id", "user_id", "problem_id", "skill_id", "correct", "original"])
    df = df[(df.original == 1) & df.skill_id.notna()].drop_duplicates("order_id")  # main problems, one row per attempt
    df["skill_id"] = df.skill_id.astype(int)
    return df.sort_values(["user_id", "order_id"]).reset_index(drop=True)

def split(df, seed=42):
    users = df.user_id.unique(); rng = np.random.default_rng(seed); rng.shuffle(users)
    test_u = set(users[: int(0.2 * len(users))])
    return df[~df.user_id.isin(test_u)], df[df.user_id.isin(test_u)]

class Elo:
    """Per-skill difficulty learned on train; student ability per (student, skill) updated online with uncertainty-decay K."""
    def __init__(self, k0=0.4, decay=0.05): self.k0, self.decay, self.beta = k0, decay, {}
    @staticmethod
    def p(theta, beta): return 1 / (1 + math.exp(-(theta - beta)))
    def run(self, df, learn_beta):
        theta, n, preds = {}, {}, np.empty(len(df))
        for i, (u, s, y) in enumerate(zip(df.user_id.values, df.skill_id.values, df.correct.values)):
            key = (u, s); th = theta.get(key, 0.0); b = self.beta.get(s, 0.0)
            pr = self.p(th, b); preds[i] = pr
            k = self.k0 / (1 + self.decay * n.get(key, 0))
            theta[key] = th + k * (y - pr)
            if learn_beta: self.beta[s] = b - 0.05 * (y - pr)
            n[key] = n.get(key, 0) + 1
        return preds

class BKT:
    """Standard 4-param BKT per skill, params by coarse grid search on train (fast, no deps)."""
    GRID = [(pi, pt, ps, pg) for pi in (0.2, 0.4, 0.6) for pt in (0.05, 0.1, 0.2, 0.3)
            for ps in (0.05, 0.1, 0.2) for pg in (0.1, 0.2, 0.3)]
    def __init__(self): self.params = {}
    @staticmethod
    def seq_preds(ys, pi, pt, ps, pg):
        pl, out = pi, []
        for y in ys:
            pc = pl * (1 - ps) + (1 - pl) * pg; out.append(pc)
            post = pl * (1 - ps) / pc if y else pl * ps / (1 - pc)
            pl = post + (1 - post) * pt
        return out
    def fit(self, df):
        for s, g in df.groupby("skill_id"):
            seqs = [grp.correct.values for _, grp in g.groupby("user_id")]
            best, bp = 1e9, None
            for prm in self.GRID:
                ll = 0.0
                for ys in seqs:
                    pr = np.clip(self.seq_preds(ys, *prm), 1e-4, 1 - 1e-4)
                    ll -= np.sum(ys * np.log(pr) + (1 - ys) * np.log(1 - pr))
                if ll < best: best, bp = ll, prm
            self.params[s] = bp
    def predict(self, df):
        preds = pd.Series(index=df.index, dtype=float)
        for (u, s), grp in df.groupby(["user_id", "skill_id"]):
            prm = self.params.get(s, (0.4, 0.1, 0.1, 0.2))
            preds.loc[grp.index] = self.seq_preds(grp.correct.values, *prm)
        return preds.values

class Elo2:
    """Multi-level Elo: p = sigmoid(theta_student + theta_student_skill - beta_item).
    beta_item learned on train (falls back to skill difficulty for unseen items)."""
    def __init__(self, k_s=0.3, k_ss=0.4, k_b=0.3, decay=0.05):
        self.k_s, self.k_ss, self.k_b, self.decay = k_s, k_ss, k_b, decay
        self.b_item, self.b_skill, self.n_item = {}, {}, {}
    def run(self, df, learn):
        th, ths, n, preds = {}, {}, {}, np.empty(len(df))
        for i, (u, it, s, y) in enumerate(zip(df.user_id.values, df.problem_id.values, df.skill_id.values, df.correct.values)):
            b = self.b_item.get(it, self.b_skill.get(s, 0.0))
            z = th.get(u, 0.0) + ths.get((u, s), 0.0) - b
            p = 1 / (1 + math.exp(-z)); preds[i] = p; e = y - p
            nu, nus = n.get(u, 0), n.get((u, s), 0)
            th[u] = th.get(u, 0.0) + self.k_s / (1 + self.decay * nu) * e
            ths[(u, s)] = ths.get((u, s), 0.0) + self.k_ss / (1 + self.decay * nus) * e
            n[u] = nu + 1; n[(u, s)] = nus + 1
            if learn:
                ni = self.n_item.get(it, 0)
                self.b_item[it] = b - self.k_b / (1 + 0.05 * ni) * e; self.n_item[it] = ni + 1
                self.b_skill[s] = self.b_skill.get(s, 0.0) - 0.01 * e
        return preds


def report(name, y, p):
    """Print one console line for a model and return its unrounded (AUC, RMSE)."""
    auc, rmse = roc_auc_score(y, p), math.sqrt(mean_squared_error(y, p))
    print(f"{name:12s} AUC={auc:.4f} RMSE={rmse:.4f}")
    return auc, rmse

def score_models(tr, te, bkt=False):
    """Fit every model on `tr`, predict each next response in `te`, print one line per model.
    Returns {"AUC": {model: float}, "RMSE": {model: float}}, unrounded, keyed by results-file model names."""
    y = te.correct.values; out = {}
    out["global-mean"] = report("global-mean", y, np.full(len(te), tr.correct.mean()))
    sm = tr.groupby("skill_id").correct.mean()
    out["skill-mean"] = report("skill-mean", y, te.skill_id.map(sm).fillna(tr.correct.mean()).values)
    elo = Elo(); elo.run(tr, learn_beta=True); out["elo"] = report("elo", y, elo.run(te, learn_beta=False))
    m = Elo2(); m.run(tr, True); m.run(tr, True); out[KWAN] = report("elo-multi", y, m.run(te, False))   # Kwan default
    if bkt:
        b = BKT(); b.fit(tr); out["bkt"] = report("bkt", y, b.predict(te))
    return {"AUC": {k: float(v[0]) for k, v in out.items()}, "RMSE": {k: float(v[1]) for k, v in out.items()}}

def evaluate(df, bkt=False, seed=42):
    """Split by student, run all models, print the console report. Returns unrounded AUC and RMSE per model."""
    tr, te = split(df, seed)
    print(f"rows={len(df)} students={df.user_id.nunique()} skills={df.skill_id.nunique()} train={len(tr)} test={len(te)}")
    return score_models(tr, te, bkt)

def rounded(scores):
    """{model: metric} in results-file order, rounded to 4 decimals (same precision as the console lines)."""
    return {k: round(float(scores[k]), 4) for k in MODEL_ORDER if k in scores}

def make_block(df, scores):
    """The `assistments_2009` block of data/eval_results.json (structure: PRD Section 8)."""
    return {"dataset": DATASET, "rows": int(len(df)), "students": int(df.user_id.nunique()),
            "skills": int(df.skill_id.nunique()), "split": SPLIT,
            "AUC": rounded(scores["AUC"]), "RMSE": rounded(scores["RMSE"])}

def write_block(key, block, out):
    """Merge one block into the results file through results.update, the single writer."""
    import results
    results.update(key, block, path=out)
    print(f"wrote {key} to {out}", file=sys.stderr)

def main(argv=None):
    ap = argparse.ArgumentParser(description="Validate Kwan's learner model on ASSISTments 2009 (80/20 split by student).")
    ap.add_argument("csv", help="path to skill_builder_data_corrected.csv")
    ap.add_argument("--bkt", action="store_true", help="also fit the BKT baseline (grid search, a few minutes)")
    ap.add_argument("--out", metavar="PATH", help=f"merge the '{KEY}' block into this results JSON, e.g. data/eval_results.json")
    a = ap.parse_args(argv)
    df = load(a.csv); scores = evaluate(df, bkt=a.bkt); block = make_block(df, scores)
    if a.out: write_block(KEY, block, a.out)
    return block

if __name__ == "__main__":
    main()
