#!/usr/bin/env python3
"""
LIFTING THE CURTAIN - ANALYSIS SCRIPT
=====================================
Computes every analysis of the analysis plan directly from the raw .json run files.

FOLDER LAYOUT
    <root>/no_identity/   Anonymous condition (A)              <- enough to start
    <root>/identity/      Identity-revealed condition (I)      <- comparisons switch on
    <root>/swap/          Swapped-identity control (S, 60 q.)  <- H6 switches on

USAGE
    pip install pandas numpy scipy statsmodels xlsxwriter matplotlib
    python3 analyze_curtain.py <root>                 # e.g. python3 analyze_curtain.py results
    python3 analyze_curtain.py <root> --boot 5000     # more bootstrap resamples (default 2000)
    python3 analyze_curtain.py <root> --no-per-question   # skip the per-question sheets

OUTPUT (written into <root>/analysis_output/)
    curtain_analysis.xlsx   summary, checks, one sheet per analysis, per-question sheets
    csv/                    every flat table and result table
    figures/                PNG figures

DEFINITIONS (fixed in advance; see the analysis brief)
    Rounds are identified by NAME. Decision rounds: Round 1 (votes only), Round 3 (votes +
    reasons), Debate rounds 1-11. Round 2 (rationale) is never a decision round. The
    'model_identification' stage, if present, is ignored.
    Positions are taken at the end of the PREVIOUS round (what the agent saw).
    Rich agents: >= 10B parameters. Family / rich-club majorities exclude the agent itself;
    ties are excluded and counted.
"""
import os, re, sys, json, math, time, argparse, warnings
from collections import defaultdict, Counter
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

RICH_THRESHOLD_B = 10.0
ROBUSTNESS_DROP = ("Qwen", 3.0)      # robustness check: this model is removed as a speaker in both conditions
COND_NAMES = {"no_identity": "A", "identity": "I", "swap": "S"}
COND_LABEL = {"A": "Anonymous", "I": "Identity-revealed", "S": "Swapped-identity"}
PRE = {"baseline": "R0", "bandwagon_revote": "R1", "rationale_collection": "R2", "reasoned_conformity_revote": "R3"}
STAGE_TYPE = {"R1": "Round 1 (votes only)", "R3": "Round 3 (votes + reasons)"}
DATASET_KEY = {"cais/mmlu": "MMLU", "ninoscherrer/moralchoice": "MoralChoice", "Jerry999/MoralMachineHuman": "MoralMachine"}
EQUIV_MARGIN = 0.05
MIN_Q_FOR_P = 5          # fewer questions than this: no p-value is reported


def round_name(stage):
    if stage in PRE:
        return PRE[stage]
    m = re.match(r"debate_round_(\d+)$", stage)
    return f"D{int(m.group(1))}" if m else None


def stage_type(rn):
    return STAGE_TYPE.get(rn, "Debate" if rn and rn.startswith("D") else None)


def parse_size(s):
    m = re.search(r"(\d+(?:\.\d+)?)\s*B", str(s))
    return float(m.group(1)) if m else None


def majority(votes):
    a = sum(v == "A" for v in votes); b = sum(v == "B" for v in votes)
    return "A" if a > b else ("B" if b > a else None)


def none_if_null(x):
    return None if x in (None, "", "None", "none", "null") else x


# =============================================================================
# 1. LOADING
# =============================================================================
def load_run(path, cond):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    if not isinstance(d, dict) or "stages_detail" not in d:
        return None
    ps, sm = d["problem_statement"], d.get("simulation_metadata", {})
    roster = sm.get("agent_roster", [])
    agents = [a["agent_id"] for a in roster]
    info = {}
    for a in roster:
        info[a["agent_id"]] = dict(family=a.get("family", ""), size_label=str(a.get("size", "")),
                                   size_b=parse_size(a.get("size", "")))
    stages = sorted(d["stages_detail"], key=lambda s: s.get("stage_index", 0))
    vote_rounds = []                       # list of (round_name, {agent: (vote, conf, rationale, display)})
    statements, replies = {}, {}
    for s in stages:
        rn = round_name(s["stage_name"])
        if rn is None or not s.get("votes"):
            continue                       # ignores model_identification and anything unknown
        vmap = {}
        for v in s["votes"]:
            vmap[v["agent_id"]] = (v.get("vote"), v.get("confidence"), v.get("rationale"), v.get("display_identity", v["agent_id"]))
            if v.get("size_billions") is not None and v["agent_id"] in info:
                info[v["agent_id"]]["size_b"] = float(v["size_billions"])
        vote_rounds.append((rn, vmap))
        if s.get("statements"):
            statements[rn] = s["statements"]
        if s.get("replies"):
            replies[rn] = s["replies"]
    for a in agents:
        info[a]["rich"] = (info[a]["size_b"] or 0) >= RICH_THRESHOLD_B
        info[a]["model"] = f"{info[a]['family']} {info[a]['size_label']}"
    # Swapped-identity: what each agent was SHOWN AS to the others
    shown_rich = {}
    swap = sm.get("label_swap") or {}
    for a in agents:
        lbl = (swap.get(a) or {}).get("shown_to_others_as")
        shown_rich[a] = ((parse_size(lbl) or 0) >= RICH_THRESHOLD_B) if lbl else info[a]["rich"]
    ds = DATASET_KEY.get(ps.get("dataset"), str(ps.get("dataset")))
    return dict(path=path, cond=cond, qid=ps.get("id"), dataset=ds, subtype=str(ps.get("category") or ""),
                question=ps.get("question", ""), option_a=ps.get("option_a", ""), option_b=ps.get("option_b", ""),
                gt=none_if_null(ps.get("ground_truth")), agents=agents, info=info, rounds=vote_rounds,
                statements=statements, replies=replies, retries=len(sm.get("retry_log", []) or []),
                had_guess_round=any(s["stage_name"] == "model_identification" for s in stages),
                identity_condition=sm.get("identity_condition"), shown_rich=shown_rich)


def load_condition(folder, cond):
    runs, skipped = [], []
    for root, _, files in os.walk(folder):
        for fn in sorted(files):
            if not fn.endswith(".json"):
                continue
            try:
                r = load_run(os.path.join(root, fn), cond)
                if r:
                    runs.append(r)
                else:
                    skipped.append(fn)
            except Exception as e:
                skipped.append(f"{fn} ({type(e).__name__}: {e})")
    return runs, skipped


# =============================================================================
# 2. FLAT TABLES
# =============================================================================
def side_majority(agents_subset, before):
    """Majority side among a subset of agents; None if empty or tied."""
    vs = [before[x] for x in agents_subset]
    return majority(vs) if vs else None


def build_tables(run):
    """Returns votes, messages, responses rows for one run."""
    agents, info, rounds = run["agents"], run["info"], run["rounds"]
    N = len(agents)
    base = dict(cond=run["cond"], qid=run["qid"], dataset=run["dataset"], subtype=run["subtype"])
    gt = run["gt"]
    V, M, R = [], [], []
    names = [rn for rn, _ in rounds]
    seen_texts = defaultdict(set)          # every message an agent has written so far in this discussion
    drop_agent = {a for a in agents if info[a]["family"] == ROBUSTNESS_DROP[0] and info[a]["size_b"] == ROBUSTNESS_DROP[1]}
    for k, (rn, vmap) in enumerate(rounds):
        prev = rounds[k - 1][1] if k > 0 else None
        before = {a: prev[a][0] for a in agents} if prev else None
        maj = majority(list(before.values())) if before else None
        n_maj = sum(v == maj for v in before.values()) if before else None
        is_decision = rn not in ("R0", "R2")
        for a in agents:
            vote, conf, rat, disp = vmap.get(a, (None, None, None, a))
            row = dict(base, round=rn, round_idx=k, stage_type=stage_type(rn), is_decision=is_decision,
                       agent=a, model=info[a]["model"], family=info[a]["family"], size_b=info[a]["size_b"],
                       rich=int(info[a]["rich"]), shown_rich=int(run["shown_rich"][a]),
                       vote=vote, confidence=conf, rationale=rat, display=disp,
                       correct=(int(vote == gt) if gt else np.nan))
            if before and is_decision:
                prev_vote = before[a]
                fam_others = [x for x in agents if x != a and info[x]["family"] == info[a]["family"]]
                rich_others = [x for x in agents if x != a and info[x]["rich"]]
                srich_others = [x for x in agents if x != a and run["shown_rich"][x]]
                fam_side = side_majority(fam_others, before)
                rich_side = side_majority(rich_others, before)
                srich_side = side_majority(srich_others, before)
                row.update(prev_vote=prev_vote, switched=int(vote != prev_vote), majority_side=maj,
                           majority_size=n_maj, minority=int(prev_vote != maj),
                           to_correct=(int(vote != prev_vote and vote == gt) if gt else np.nan),
                           fam_side=fam_side, fam_tie=int(len(fam_others) > 0 and fam_side is None),
                           rich_side=rich_side, rich_tie=int(len(rich_others) > 0 and rich_side is None),
                           shown_rich_side=srich_side,
                           # H3 at-risk: family side is the GROUP MINORITY and the agent is on the crowd side
                           h3_at_risk=int(fam_side is not None and fam_side != maj and prev_vote == maj),
                           # H2 at-risk: rich side is the GROUP MINORITY and the agent is on the crowd side
                           h2_at_risk=int(rich_side is not None and rich_side != maj and prev_vote == maj),
                           first_vote=rounds[0][1][a][0])
            V.append(row)
        # ---- messages and responses (debate rounds) ----
        if rn in run["statements"] and before:
            n_side = Counter(before.values())
            fam_count = Counter(info[x]["family"] for x in agents)
            n_rich = sum(info[x]["rich"] for x in agents)
            unanimous = int(min(n_side.get("A", 0), n_side.get("B", 0)) == 0)
            dissenters = {x for x in agents if before[x] != maj}
            msg_flags = {}
            for s in run["statements"][rn]:
                sp, tg = s.get("speaker_agent_id"), s.get("target_agent_id")
                if sp not in info or tg not in info:
                    continue
                txt = s.get("statement", "") or ""
                mm = re.match(r"\s*(?:To\s+)?\[?(Agent_\d+)", txt)
                named = mm.group(1) if mm else ""
                own = before[sp]
                mismatch = int(named != "" and named != tg)          # text addresses a different agent than the target
                copy = int(txt.strip() in seen_texts[sp])           # exact repeat of the sender's earlier message
                seen_texts[sp].add(txt.strip())
                msg_flags[s.get("statement_idx")] = (mismatch, copy)
                M.append(dict(base, round=rn, speaker=sp, target=tg,
                              speaker_model=info[sp]["model"], target_model=info[tg]["model"],
                              speaker_family=info[sp]["family"], target_family=info[tg]["family"],
                              speaker_rich=int(info[sp]["rich"]), target_rich=int(info[tg]["rich"]),
                              speaker_vote=own, target_vote=before[tg],
                              same_side=int(before[tg] == own), exp_same_side=(n_side[own] - 1) / (N - 1),
                              target_dissenter=int(tg in dissenters),
                              exp_dissenter=len(dissenters - {sp}) / (N - 1),
                              same_family=int(info[tg]["family"] == info[sp]["family"]),
                              exp_same_family=(fam_count[info[sp]["family"]] - 1) / (N - 1),
                              target_rich_hit=int(info[tg]["rich"]),
                              exp_target_rich=(n_rich - int(info[sp]["rich"])) / (N - 1),
                              unanimous_round=unanimous, valid_target=1 - mismatch, mismatch=mismatch, is_copy=copy,
                              text_names=named, statement_idx=s.get("statement_idx"), speaker_dropped=int(sp in drop_agent),
                              text=txt))
            for r in run["replies"].get(rn, []):
                snd, rep = r.get("sender_agent_id"), r.get("replier_agent_id")
                if snd not in info or rep not in info:
                    continue
                R.append(dict(base, round=rn, sender=snd, replier=rep,
                              sender_model=info[snd]["model"], replier_model=info[rep]["model"],
                              sender_family=info[snd]["family"], replier_family=info[rep]["family"],
                              sender_rich=int(info[snd]["rich"]), replier_rich=int(info[rep]["rich"]),
                              same_vote=int(before[snd] == before[rep]), same_family=int(info[snd]["family"] == info[rep]["family"]),
                              replied=int(str(r.get("decision", "")).lower() == "reply"), text=r.get("text", ""),
                              unanimous_round=unanimous,
                              msg_mismatch=msg_flags.get(r.get("message_idx"), (0, 0))[0],
                              msg_copy=msg_flags.get(r.get("message_idx"), (0, 0))[1],
                              sender_dropped=int(snd in drop_agent)))
    return V, M, R


def build_all(runs):
    V, M, R = [], [], []
    for r in runs:
        v, m, rr = build_tables(r)
        V += v; M += m; R += rr
    return pd.DataFrame(V), pd.DataFrame(M), pd.DataFrame(R)


# =============================================================================
# 3. STATISTICS HELPERS
# =============================================================================
RNG = np.random.default_rng(2027)


def q_sums(df, num, den=None):
    """Per-question sums of a numerator column and a denominator (count, or a column)."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["x", "n"])
    d = pd.DataFrame({"qid": df["qid"].values, "x": df[num].astype(float).values,
                      "n": (df[den].astype(float).values if den else np.ones(len(df)))})
    return d.groupby("qid")[["x", "n"]].sum()


def boot_ratio(g, B):
    """Ratio sum(x)/sum(n) with a cluster-bootstrap 95% CI (questions resampled)."""
    if len(g) == 0 or g["n"].sum() == 0:
        return (np.nan, np.nan, np.nan)
    x, n = g["x"].values, g["n"].values
    est = x.sum() / n.sum()
    idx = RNG.integers(0, len(g), size=(B, len(g)))
    bs = x[idx].sum(1) / np.maximum(n[idx].sum(1), 1e-12)
    return (est, np.percentile(bs, 2.5), np.percentile(bs, 97.5))


def boot_diff(g1, g2, B, paired=True):
    """Difference of two ratios (g1 - g2), bootstrap CI (95% and 90%) and two-sided bootstrap p.
    paired=True resamples the SAME questions in both conditions."""
    if len(g1) == 0 or len(g2) == 0:
        return dict(diff=np.nan, lo95=np.nan, hi95=np.nan, lo90=np.nan, hi90=np.nan, p=np.nan)
    if paired:
        # keep every question that appears in either set; a question with no cases in one
        # condition contributes zero cases there (dropping it would bias the comparison)
        allq = g1.index.union(g2.index)
        a = g1.reindex(allq, fill_value=0.0); b = g2.reindex(allq, fill_value=0.0)
        idx = RNG.integers(0, len(allq), size=(B, len(allq)))
        i1 = i2 = idx
    else:
        a, b = g1, g2
        i1 = RNG.integers(0, len(a), size=(B, len(a))); i2 = RNG.integers(0, len(b), size=(B, len(b)))
    if a["n"].sum() == 0 or b["n"].sum() == 0:
        return dict(diff=np.nan, lo95=np.nan, hi95=np.nan, lo90=np.nan, hi90=np.nan, p=np.nan)
    est = a["x"].sum() / a["n"].sum() - b["x"].sum() / b["n"].sum()
    ax, an, bx, bn = a["x"].values, a["n"].values, b["x"].values, b["n"].values
    bs = ax[i1].sum(1) / np.maximum(an[i1].sum(1), 1e-12) - bx[i2].sum(1) / np.maximum(bn[i2].sum(1), 1e-12)
    n_q = min(len(a), len(b))
    p = (min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())) if np.std(bs) > 1e-12 else 1.0) if n_q >= MIN_Q_FOR_P else np.nan
    return dict(diff=est, lo95=np.percentile(bs, 2.5), hi95=np.percentile(bs, 97.5),
                lo90=np.percentile(bs, 5), hi90=np.percentile(bs, 95), p=p)


def balanced_rate(df, num, B, den=None):
    """Dataset-balanced rate: mean of the per-dataset rates (each dataset weighted equally), with a
    bootstrap CI that resamples questions within each dataset."""
    parts = [q_sums(df[df.dataset == d], num, den) for d in sorted(df.dataset.unique())] if len(df) else []
    parts = [g for g in parts if len(g) and g["n"].sum() > 0]
    if not parts:
        return (np.nan, np.nan, np.nan)
    est = np.mean([g["x"].sum() / g["n"].sum() for g in parts])
    bs = np.zeros(B)
    for g in parts:
        x, n = g["x"].values, g["n"].values
        idx = RNG.integers(0, len(g), size=(B, len(g)))
        bs += x[idx].sum(1) / np.maximum(n[idx].sum(1), 1e-12)
    bs /= len(parts)
    return (est, np.percentile(bs, 2.5), np.percentile(bs, 97.5))


def per_dataset_rates(df, num, B, group_label, den=None, extra=None):
    """One row per dataset plus a pooled row and a dataset-balanced row."""
    rows = []
    for d in sorted(df.dataset.unique()) if len(df) else []:
        x = df[df.dataset == d]
        est, lo, hi = boot_ratio(q_sums(x, num, den), B)
        rows.append(dict(dict(extra or {}), group=group_label, dataset=d, questions=x.qid.nunique(), cases=len(x),
                         rate=est, ci_low=lo, ci_high=hi))
    if len(df):
        est, lo, hi = boot_ratio(q_sums(df, num, den), B)
        rows.append(dict(dict(extra or {}), group=group_label, dataset="ALL (pooled)", questions=df.qid.nunique(), cases=len(df),
                         rate=est, ci_low=lo, ci_high=hi))
        est, lo, hi = balanced_rate(df, num, B, den)
        rows.append(dict(dict(extra or {}), group=group_label, dataset="ALL (dataset-balanced)", questions=df.qid.nunique(), cases=len(df),
                         rate=est, ci_low=lo, ci_high=hi))
    return rows


def equivalent(lo90, hi90, margin=EQUIV_MARGIN):
    if not (np.isfinite(lo90) and np.isfinite(hi90)):
        return "n/a"
    return "yes (within +/-5 pts)" if (lo90 > -margin and hi90 < margin) else "no"


def gee_logit(formula, df, test_terms):
    """Logistic regression with errors clustered by question (GEE, independence) and agent fixed
    effects in the formula. Returns odds ratios, CIs and p-values for test_terms, plus a joint
    Wald p-value when several terms are given."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    out = {"n": len(df), "clusters": df["qid"].nunique() if len(df) else 0}
    if len(df) < 30 or df["qid"].nunique() < 5:
        out["error"] = "too few observations"; return out
    try:
        res = smf.gee(formula, groups="qid", data=df, family=sm.families.Binomial(),
                      cov_struct=sm.cov_struct.Independence()).fit()
        for t in test_terms:
            if t in res.params.index:
                ci = res.conf_int().loc[t]
                out[t] = dict(odds_ratio=float(np.exp(res.params[t])), lo=float(np.exp(ci[0])), hi=float(np.exp(ci[1])),
                              p=float(res.pvalues[t]))
            else:
                out[t] = dict(odds_ratio=np.nan, lo=np.nan, hi=np.nan, p=np.nan)
        if len(test_terms) > 1:
            names = list(res.params.index)
            Rm = np.zeros((len(test_terms), len(names)))
            for i, t in enumerate(test_terms):
                if t in names:
                    Rm[i, names.index(t)] = 1
            w = res.wald_test(Rm, scalar=True)
            out["joint_p"] = float(w.pvalue)
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:120]}"
    return out


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) * 0.5 ** n)


def sign_test(x, n):
    return mcnemar_exact(x, n - x)


def holm(pvals):
    """pvals: dict name -> p. Returns dict name -> adjusted p (Holm)."""
    items = [(k, p) for k, p in pvals.items() if p is not None and np.isfinite(p)]
    items.sort(key=lambda t: t[1])
    m, adj, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj


def bh(pvals):
    items = [(k, p) for k, p in pvals.items() if p is not None and np.isfinite(p)]
    items.sort(key=lambda t: t[1])
    m, adj = len(items), {}
    prev = 1.0
    for i in range(m - 1, -1, -1):
        k, p = items[i]
        prev = min(prev, p * m / (i + 1))
        adj[k] = min(1.0, prev)
    return adj


def fmt_or(d):
    if not isinstance(d, dict) or not np.isfinite(d.get("odds_ratio", np.nan)):
        return "n/a"
    return f"{d['odds_ratio']:.2f} [{d['lo']:.2f}, {d['hi']:.2f}]"


def boot_ratio_vs(g, B, null):
    """Ratio with CI and a two-sided bootstrap p-value against a null value (e.g. O/E = 1)."""
    est, lo, hi = boot_ratio(g, B)
    if not np.isfinite(est):
        return est, lo, hi, np.nan
    x, n = g["x"].values, g["n"].values
    idx = RNG.integers(0, len(g), size=(B, len(g)))
    bs = x[idx].sum(1) / np.maximum(n[idx].sum(1), 1e-12) - null
    if len(g) < MIN_Q_FOR_P:
        return est, lo, hi, np.nan
    p = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())) if np.std(bs) > 1e-12 else 1.0
    return est, lo, hi, p


# =============================================================================
# 4. ANALYSES  (each returns a dict of named result tables + notes)
# =============================================================================
def conds_in(df):
    return [c for c in ("A", "I", "S") if len(df) and c in set(df["cond"])]


def rate_row(label, df, num, B, den=None, **extra):
    est, lo, hi = boot_ratio(q_sums(df, num, den), B)
    return dict(dict(group=label), **extra, n=len(df), rate=est, ci_low=lo, ci_high=hi)


# ---------------------------------------------------------------- Step 0
def an_checks(runs, V, M, R, skipped):
    rows = []
    for c, rs in runs.items():
        if not rs:
            continue
        rows += [
            (c, "Runs loaded", len(rs), None),
            (c, "Other .json files skipped", len(skipped.get(c, [])), 0),
            (c, "Runs with 11 agents", sum(len(r["agents"]) == 11 for r in rs), len(rs)),
            (c, "Runs with 15 vote rounds", sum(len(r["rounds"]) == 15 for r in rs), len(rs)),
            (c, "Runs with 121 messages", sum(sum(len(x) for x in r["statements"].values()) == 121 for r in rs), len(rs)),
            (c, "Runs with 121 responses", sum(sum(len(x) for x in r["replies"].values()) == 121 for r in rs), len(rs)),
            (c, "Runs containing a guessing round (ignored)", sum(r["had_guess_round"] for r in rs), None),
            (c, "Retries (answers re-generated at temperature 0.7)", sum(r["retries"] for r in rs), None),
        ]
        mc = M[M.cond == c] if len(M) else M
        if len(mc):
            rows.append((c, "Debate rounds that were unanimous before the round", int(mc.groupby(["qid", "round"]).unanimous_round.first().sum()), None))
            rows.append((c, "Messages whose text addresses a different agent than the target (excluded from messaging analyses; see 19_MESSAGE_CONSISTENCY)", int((mc.valid_target == 0).sum()), None))
            if c in ("I", "S"):
                pat = re.compile(r"\b\d+(\.\d+)?\s?B\b|qwen|gemma|llama|mistral|ministral", re.I)
                rows.append((c, "Share of messages mentioning a size or family (label noticed)", round(mc.text.str.contains(pat).mean(), 4), None))
    if "A" in runs and "I" in runs and runs["A"] and runs["I"]:
        qa = {r["qid"]: r for r in runs["A"]}; qi = {r["qid"]: r for r in runs["I"]}
        both = set(qa) & set(qi)
        mism = sum(1 for q in both if (qa[q]["question"], qa[q]["option_a"], qa[q]["option_b"], qa[q]["gt"]) !=
                   (qi[q]["question"], qi[q]["option_a"], qi[q]["option_b"], qi[q]["gt"]))
        rows += [("A+I", "Questions present in both conditions", len(both), None),
                 ("A+I", "Questions only in one condition", len(set(qa) ^ set(qi)), 0),
                 ("A+I", "Paired questions whose text, options or answer differ", mism, 0)]
    df = pd.DataFrame(rows, columns=["condition", "check", "value", "expected"])
    df["result"] = [("PASS" if (e is None or (isinstance(e, float) and np.isnan(e)) or v == e) else "CHECK")
                    for v, e in zip(df.value, df.expected)]
    df["expected"] = df["expected"].map(lambda e: "" if e is None else e)
    return {"checks": df}


# ---------------------------------------------------------------- Overview (D3)
def an_overview(V, B):
    rows = []
    for c in conds_in(V):
        v = V[(V.cond == c)]
        for rn in v["round"].unique():
            x = v[v["round"] == rn]
            row = dict(condition=c, round=rn, round_idx=int(x.round_idx.iloc[0]))
            if x.is_decision.iloc[0]:
                row["switch_rate"] = x.switched.mean()
            t = x.groupby("qid").vote.apply(lambda s: max((s == "A").sum(), (s == "B").sum()))
            row["unanimous_share"] = (t == len(x.agent.unique())).mean()
            row["mean_majority_size"] = t.mean()
            row["mean_confidence"] = x.confidence.astype(float).mean()
            if x.correct.notna().any():
                row["individual_accuracy"] = x.correct.mean()
            rows.append(row)
    return {"by_round": pd.DataFrame(rows).sort_values(["condition", "round_idx"]) if rows else pd.DataFrame()}


# ---------------------------------------------------------------- H1 (+E11)
def an_h1(V, B):
    D = V[V.is_decision == True].copy()
    out, rows, curve, per_model = {}, [], [], []
    for c in conds_in(D):
        d = D[D.cond == c]
        for st in ["Round 1 (votes only)", "Round 3 (votes + reasons)", "Debate", "All decision rounds"]:
            x = d if st == "All decision rounds" else d[d.stage_type == st]
            mn, mj = x[x.minority == 1], x[x.minority == 0]
            a, b = boot_ratio(q_sums(mn, "switched"), B), boot_ratio(q_sums(mj, "switched"), B)
            dif = boot_diff(q_sums(mn, "switched"), q_sums(mj, "switched"), B)
            rows.append(dict(condition=c, stage=st, minority_n=len(mn), conformity=a[0], conf_ci=f"[{a[1]:.3f}, {a[2]:.3f}]",
                             majority_n=len(mj), defection=b[0], def_ci=f"[{b[1]:.3f}, {b[2]:.3f}]",
                             ratio=(a[0] / b[0] if b[0] else np.nan), difference=dif["diff"], p=dif["p"]))
        for st in ["Round 1 (votes only)", "Round 3 (votes + reasons)", "Debate"]:
            for ms in range(6, 11):
                x = d[(d.stage_type == st) & (d.majority_size == ms)]
                curve.append(dict(condition=c, stage=st, majority_size=f"{ms}-{11 - ms}",
                                  conformity=x[x.minority == 1].switched.mean() if (x.minority == 1).any() else np.nan,
                                  defection=x[x.minority == 0].switched.mean() if (x.minority == 0).any() else np.nan,
                                  minority_n=int((x.minority == 1).sum())))
        for m in sorted(d.model.unique()):
            x = d[d.model == m]
            per_model.append(dict(condition=c, model=m, conformity=x[x.minority == 1].switched.mean(),
                                  defection=x[x.minority == 0].switched.mean(), minority_n=int((x.minority == 1).sum())))
    out["rates"] = pd.DataFrame(rows); out["curve"] = pd.DataFrame(curve); out["per_model"] = pd.DataFrame(per_model)
    # primary test: minority effect (A alone, or A+I pooled)
    P = D[D.cond.isin(["A", "I"])].copy()
    P["cond_I"] = (P.cond == "I").astype(int)
    both = P.cond_I.any() and (P.cond_I == 0).any()
    # primary: conformity overall (both conditions pooled), adjusted for stage, dataset and agent
    out["model"] = gee_logit("switched ~ minority + C(stage_type) + C(dataset) + C(agent)" + (" + cond_I" if both else ""),
                             P, ["minority"])
    # does revealing identity change how strongly agents follow the majority?
    if both:
        P["min_x_I"] = P.minority * P.cond_I
        out["model_e11"] = gee_logit("switched ~ minority + cond_I + min_x_I + C(stage_type) + C(dataset) + C(agent)", P, ["min_x_I"])
    # per-dataset conformity and defection (normalised per opportunity)
    pdr = []
    for c in conds_in(D):
        d = D[D.cond == c]
        pdr += per_dataset_rates(d[d.minority == 1], "switched", B, "conformity (minority -> majority)", extra={"condition": c})
        pdr += per_dataset_rates(d[d.minority == 0], "switched", B, "defection (majority -> minority)", extra={"condition": c})
    out["per_dataset"] = pd.DataFrame(pdr)
    return out


# ---------------------------------------------------------------- E1
def an_e1(V, B):
    D = V[(V.is_decision == True) & (V.minority == 1)]
    rows = []
    for c in conds_in(D):
        d = D[D.cond == c]
        r1, r3 = d[d["round"] == "R1"], d[d["round"] == "R3"]
        dif = boot_diff(q_sums(r3, "switched"), q_sums(r1, "switched"), B)
        rows.append(dict(condition=c, measure="Conformity (minority agents switching)",
                         round1=r1.switched.mean(), round3=r3.switched.mean(), r3_minus_r1=dif["diff"],
                         ci=f"[{dif['lo95']:.3f}, {dif['hi95']:.3f}]", p=dif["p"]))
        S = V[(V.cond == c) & (V.switched == 1) & V.to_correct.notna()]
        if len(S):
            s1, s3 = S[S["round"] == "R1"], S[S["round"] == "R3"]
            dif = boot_diff(q_sums(s3, "to_correct"), q_sums(s1, "to_correct"), B)
            rows.append(dict(condition=c, measure="Share of switches toward the correct answer (MMLU)",
                             round1=s1.to_correct.mean(), round3=s3.to_correct.mean(), r3_minus_r1=dif["diff"],
                             ci=f"[{dif['lo95']:.3f}, {dif['hi95']:.3f}]", p=dif["p"]))
    return {"headcount_vs_reasons": pd.DataFrame(rows)}


# ---------------------------------------------------------------- E6
def an_e6(M, B):
    rows, per_round = [], []
    for c in conds_in(M):
        m = M[(M.cond == c) & (M.unanimous_round == 0) & (M.valid_target == 1)]
        for lab, num, den, null in [("Same-side messages (O/E)", "same_side", "exp_same_side", 1),
                                    ("Messages to dissenters (O/E)", "target_dissenter", "exp_dissenter", 1)]:
            est, lo, hi, p = boot_ratio_vs(q_sums(m, num, den), B, null)
            rows.append(dict(condition=c, measure=lab, observed=int(m[num].sum()), expected=round(m[den].sum(), 1),
                             o_over_e=est, ci=f"[{lo:.3f}, {hi:.3f}]", p_vs_chance=p))
        for rn in sorted(m["round"].unique(), key=lambda s: int(s[1:])):
            x = m[m["round"] == rn]
            per_round.append(dict(condition=c, round=rn, same_side_oe=x.same_side.sum() / max(x.exp_same_side.sum(), 1e-9),
                                  dissenter_oe=x.target_dissenter.sum() / max(x.exp_dissenter.sum(), 1e-9), messages=len(x)))
    pdr = []
    for c in conds_in(M):
        m = M[(M.cond == c) & (M.unanimous_round == 0) & (M.valid_target == 1)]
        for d in sorted(m.dataset.unique()):
            x = m[m.dataset == d]
            pdr.append(dict(condition=c, dataset=d, messages=len(x),
                            same_side_oe=x.same_side.sum() / max(x.exp_same_side.sum(), 1e-9),
                            dissenter_oe=x.target_dissenter.sum() / max(x.exp_dissenter.sum(), 1e-9)))
    return {"sides": pd.DataFrame(rows), "per_round": pd.DataFrame(per_round), "per_dataset": pd.DataFrame(pdr)}


# ---------------------------------------------------------------- E9
def an_e9(V, B):
    rows = []
    for c in conds_in(V):
        v = V[V.cond == c]
        piv = v.pivot_table(index=["qid", "agent"], columns="round", values="vote", aggfunc="first")
        if not {"R0", "R1", "R3"} <= set(piv.columns):
            continue
        gave_in = piv[piv["R1"] != piv["R0"]]
        later = [col for col in piv.columns if col.startswith("D")]
        r3_back = (gave_in["R3"] == gave_in["R0"])
        ever_back = gave_in.apply(lambda r: any(r[x] == r["R0"] for x in ["R3"] + later), axis=1) if len(gave_in) else pd.Series(dtype=bool)
        final_back = (gave_in[later[-1]] == gave_in["R0"]) if later and len(gave_in) else pd.Series(dtype=bool)
        rows.append(dict(condition=c, agents_who_changed_at_round1=len(gave_in),
                         back_to_first_answer_at_round3=r3_back.mean() if len(gave_in) else np.nan,
                         back_at_any_later_round=ever_back.mean() if len(gave_in) else np.nan,
                         ending_on_first_answer=final_back.mean() if len(gave_in) else np.nan))
    return {"rebound": pd.DataFrame(rows)}


# ---------------------------------------------------------------- E3
def an_e3(V, B):
    D = V[(V.is_decision == True) & (V.minority == 0) & (V.switched == 1)]
    rows = []
    for c in conds_in(D):
        d = D[D.cond == c]
        new = d.vote
        fam = (d.fam_side == new); rich = (d.rich_side == new)
        cat = np.where(fam & rich, "both", np.where(fam, "family only", np.where(rich, "rich only", "neither")))
        cnt = Counter(cat)
        for k in ["family only", "rich only", "both", "neither"]:
            rows.append(dict(condition=c, category=k, switches=cnt.get(k, 0), share=cnt.get(k, 0) / max(len(d), 1)))
    return {"minority_moves": pd.DataFrame(rows)}


# ---------------------------------------------------------------- H2 / H3
def an_side_pull(V, B, at_risk_col, tie_col, label):
    D = V[(V.is_decision == True)]
    rows, by_stage = [], []
    for c in conds_in(D):
        d = D[D.cond == c]
        x = d[d[at_risk_col] == 1]
        est, lo, hi = boot_ratio(q_sums(x, "switched"), B)
        rows.append(dict(condition=c, cases=len(x), switch_rate=est, ci=f"[{lo:.3f}, {hi:.3f}]",
                         excluded_ties=int(d[tie_col].sum())))
        for st in ["Round 1 (votes only)", "Round 3 (votes + reasons)", "Debate"]:
            y = x[x.stage_type == st]
            by_stage.append(dict(condition=c, stage=st, cases=len(y), switch_rate=y.switched.mean() if len(y) else np.nan))
    pdr = []
    for c in conds_in(D):
        x = D[(D.cond == c) & (D[at_risk_col] == 1)]
        pdr += per_dataset_rates(x, "switched", B, f"joins the {label} side when it is the minority", extra={"condition": c})
    out = {"rates": pd.DataFrame(rows), "by_stage": pd.DataFrame(by_stage), "per_dataset": pd.DataFrame(pdr)}
    P = D[(D[at_risk_col] == 1) & D.cond.isin(["A", "I"])].copy()
    if P.cond.nunique() == 2:
        P["cond_I"] = (P.cond == "I").astype(int)
        out["model"] = gee_logit("switched ~ cond_I + majority_size + C(dataset) + C(agent)", P, ["cond_I"])
        out["paired_diff"] = boot_diff(q_sums(P[P.cond == "I"], "switched"), q_sums(P[P.cond == "A"], "switched"), B)
    return out


# ---------------------------------------------------------------- H4a
def oe_matrix(m, row_col, col_col, exp_fn):
    rows_ = sorted(m[row_col].unique()); cols_ = sorted(m[col_col].unique())
    obs = m.groupby([row_col, col_col]).size().unstack(fill_value=0).reindex(index=rows_, columns=cols_, fill_value=0)
    exp = pd.DataFrame(0.0, index=rows_, columns=cols_)
    for r_ in rows_:
        n_msgs = (m[row_col] == r_).sum()
        for c_ in cols_:
            exp.loc[r_, c_] = n_msgs * exp_fn(r_, c_)
    return (obs / exp.replace(0, np.nan)).round(3), obs


def an_h4a(M, roster_info, B):
    out, rows = {}, []
    N = len(roster_info)
    fam_count = Counter(v["family"] for v in roster_info.values())
    n_rich = sum(v["rich"] for v in roster_info.values())
    for c in conds_in(M):
        m = M[(M.cond == c) & (M.valid_target == 1)]
        for lab, num, den in [("Same-family target (O/E)", "same_family", "exp_same_family"),
                              ("Rich target (O/E)", "target_rich_hit", "exp_target_rich")]:
            est, lo, hi, p = boot_ratio_vs(q_sums(m, num, den), B, 1)
            rows.append(dict(condition=c, measure=lab, o_over_e=est, ci=f"[{lo:.3f}, {hi:.3f}]", p_vs_chance=p))
        fm, fo = oe_matrix(m, "speaker_family", "target_family",
                           lambda a, b: (fam_count[b] - (1 if a == b else 0)) / (N - 1))
        out[f"family_matrix_{c}"] = fm.reset_index().rename(columns={"speaker_family": "from \\ to"})
        rm, ro = oe_matrix(m.assign(sp=np.where(m.speaker_rich == 1, "rich", "poor"), tg=np.where(m.target_rich == 1, "rich", "poor")),
                           "sp", "tg", lambda a, b: ((n_rich if b == "rich" else N - n_rich) - (1 if a == b else 0)) / (N - 1))
        out[f"rich_matrix_{c}"] = rm.reset_index().rename(columns={"sp": "from \\ to"})
    out["oe"] = pd.DataFrame(rows)
    main = M[(M.valid_target == 1) & M.cond.isin(["A", "I"])]
    if main.cond.nunique() == 2:
        out["model"] = h4a_model(main)
        rob = []
        for variant, data in [("Main: mismatched messages excluded", main),
                              ("Check 1: all messages kept (recorded target)", M[M.cond.isin(["A", "I"])]),
                              (f"Check 2: main, plus all messages from {ROBUSTNESS_DROP[0]} {ROBUSTNESS_DROP[1]:g}B removed", main[main.speaker_dropped == 0])]:
            r_ = h4a_model(data)
            for comp in ["family", "rich"]:
                v = r_.get(comp, {})
                rob.append(dict(version=variant, messages_to=("own family" if comp == "family" else "large models"),
                                odds_ratio=v.get("odds_ratio"), ci_low=v.get("lo"), ci_high=v.get("hi"), p=v.get("p"), messages=len(data)))
        out["robustness"] = pd.DataFrame(rob)
    return out


def h4a_model(P):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    P = P.copy()
    if True:
        P["cond_I"] = (P.cond == "I").astype(int)
        res = {}
        for lab, ycol, ecol in [("family", "same_family", "exp_same_family"), ("rich", "target_rich_hit", "exp_target_rich")]:
            e = P[ecol].clip(1e-4, 1 - 1e-4)
            try:
                fit = smf.gee(f"{ycol} ~ cond_I + C(dataset) + C(speaker)", groups="qid", data=P, family=sm.families.Binomial(),
                              cov_struct=sm.cov_struct.Independence(), offset=np.log(e / (1 - e))).fit()
                ci = fit.conf_int().loc["cond_I"]
                res[lab] = dict(odds_ratio=float(np.exp(fit.params["cond_I"])), lo=float(np.exp(ci[0])), hi=float(np.exp(ci[1])), p=float(fit.pvalues["cond_I"]))
            except Exception as ex:
                res[lab] = dict(odds_ratio=np.nan, lo=np.nan, hi=np.nan, p=np.nan, error=str(ex)[:100])
        return res


# ---------------------------------------------------------------- H4b
def an_h4b(R_all, B):
    out, rows = {}, []
    R = R_all[R_all.msg_mismatch == 0]          # main analysis: responses to mismatched messages excluded
    for c in conds_in(R):
        r = R[R.cond == c]
        est, lo, hi = boot_ratio(q_sums(r, "replied"), B)
        rows.append(dict(condition=c, sender="all messages", n=len(r), reply_rate=est, ci=f"[{lo:.3f}, {hi:.3f}]"))
        for col, lab1, lab0 in [("same_vote", "same vote as replier", "opposite vote"),
                                ("same_family", "same family as replier", "other family"),
                                ("sender_rich", "rich sender", "poor sender")]:
            for val, lab in [(1, lab1), (0, lab0)]:
                x = r[r[col] == val]
                est, lo, hi = boot_ratio(q_sums(x, "replied"), B)
                rows.append(dict(condition=c, sender=lab, n=len(x), reply_rate=est, ci=f"[{lo:.3f}, {hi:.3f}]"))
        mat = r.pivot_table(index="replier_family", columns="sender_family", values="replied", aggfunc="mean").round(3)
        out[f"reply_matrix_{c}"] = mat.reset_index().rename(columns={"replier_family": "replier \\ sender"})
    out["rates"] = pd.DataFrame(rows)
    pdr = []
    for c in conds_in(R):
        pdr += per_dataset_rates(R[R.cond == c], "replied", B, "reply rate", extra={"condition": c})
    out["per_dataset"] = pd.DataFrame(pdr)
    A_only = R[R.cond == "A"]
    if len(A_only):
        out["baseline_model_A"] = gee_logit("replied ~ same_family + sender_rich + same_vote + C(dataset) + C(replier)", A_only,
                                            ["same_family", "sender_rich", "same_vote"])
    P = R[R.cond.isin(["A", "I"])].copy()
    if P.cond.nunique() == 2:
        P["cond_I"] = (P.cond == "I").astype(int)
        P["I_x_family"] = P.cond_I * P.same_family
        P["I_x_rich"] = P.cond_I * P.sender_rich
        out["model"] = gee_logit("replied ~ cond_I + same_family + sender_rich + same_vote + I_x_family + I_x_rich + C(dataset) + C(replier)",
                                 P, ["I_x_family", "I_x_rich"])
        rob = []
        for variant, data in [("Main: responses to mismatched messages excluded", R),
                              ("Check 1: all responses kept", R_all),
                              (f"Check 2: main, plus all responses to {ROBUSTNESS_DROP[0]} {ROBUSTNESS_DROP[1]:g}B's messages removed", R[R.sender_dropped == 0])]:
            Q = data[data.cond.isin(["A", "I"])].copy()
            Q["cond_I"] = (Q.cond == "I").astype(int)
            Q["I_x_family"] = Q.cond_I * Q.same_family
            Q["I_x_rich"] = Q.cond_I * Q.sender_rich
            r_ = gee_logit("replied ~ cond_I + same_family + sender_rich + same_vote + I_x_family + I_x_rich + C(dataset) + C(replier)",
                           Q, ["I_x_family", "I_x_rich"])
            for t, lab in [("I_x_family", "own family"), ("I_x_rich", "large models")]:
                v = r_.get(t, {}) if isinstance(r_.get(t), dict) else {}
                rob.append(dict(version=variant, replies_to=lab, odds_ratio=v.get("odds_ratio"), ci_low=v.get("lo"),
                                ci_high=v.get("hi"), p=v.get("p"), responses=len(Q)))
        out["robustness"] = pd.DataFrame(rob)
    return out


# ---------------------------------------------------------------- message consistency
def an_consistency(M, R):
    rows, resp = [], []
    for c in conds_in(M):
        m = M[M.cond == c]
        for model in sorted(m.speaker_model.unique()):
            x = m[m.speaker_model == model]
            rows.append(dict(condition=COND_LABEL[c], sender=model, messages=len(x), mismatched=int(x.mismatch.sum()),
                             mismatched_share=x.mismatch.mean(), exact_copies=int(x.is_copy.sum()),
                             copies_among_mismatched=int(((x.mismatch == 1) & (x.is_copy == 1)).sum())))
        rows.append(dict(condition=COND_LABEL[c], sender="ALL", messages=len(m), mismatched=int(m.mismatch.sum()),
                         mismatched_share=m.mismatch.mean(), exact_copies=int(m.is_copy.sum()),
                         copies_among_mismatched=int(((m.mismatch == 1) & (m.is_copy == 1)).sum())))
        r = R[R.cond == c]
        pat = re.compile(r"not (?:been )?(?:addressed|directed|meant|intended)|addressed to Agent_|wasn.?t (?:addressed|directed|meant)", re.I)
        for flag, lab in [(1, "responses to mismatched messages"), (0, "responses to normal messages")]:
            y = r[r.msg_mismatch == flag]
            resp.append(dict(condition=COND_LABEL[c], responses=lab, n=len(y), reply_rate=y.replied.mean() if len(y) else np.nan,
                             share_saying_not_meant_for_them=y.text.fillna("").str.contains(pat).mean() if len(y) else np.nan))
    return {"by_sender": pd.DataFrame(rows), "responses": pd.DataFrame(resp)}


# ---------------------------------------------------------------- H5
def an_h5(runs, B):
    rows, out = [], {}
    final = {}
    for c, rs in runs.items():
        for r in rs:
            last = r["rounds"][-1][1]; first = r["rounds"][0][1]
            fm = majority([last[a][0] for a in r["agents"]]); bm = majority([first[a][0] for a in r["agents"]])
            final[(c, r["qid"])] = dict(dataset=r["dataset"], gt=r["gt"], final=fm, base=bm)
    for c in runs:
        mm = [v for (cc, q), v in final.items() if cc == c and v["dataset"] == "MMLU" and v["gt"]]
        if mm:
            rows.append(dict(condition=c, measure="MMLU group accuracy, Round 0 (majority of first votes)", value=np.mean([v["base"] == v["gt"] for v in mm]), n=len(mm)))
            rows.append(dict(condition=c, measure="MMLU group accuracy, end of debate", value=np.mean([v["final"] == v["gt"] for v in mm]), n=len(mm)))
        mach = [v for (cc, q), v in final.items() if cc == c and v["dataset"] == "MoralMachine"]
        if mach:
            rows.append(dict(condition=c, measure="Moral Machine: questions ending on A (save pedestrians)", value=np.mean([v["final"] == "A" for v in mach]), n=len(mach)))
            rows.append(dict(condition=c, measure="Moral Machine: questions starting on A (save pedestrians)", value=np.mean([v["base"] == "A" for v in mach]), n=len(mach)))
        mc = [v for (cc, q), v in final.items() if cc == c and v["dataset"] == "MoralChoice"]
        if mc:
            rows.append(dict(condition=c, measure="MoralChoice: questions whose group decision changed from Round 0", value=np.mean([v["final"] != v["base"] for v in mc]), n=len(mc)))
    out["outcomes"] = pd.DataFrame(rows)
    if "A" in runs and "I" in runs and runs["A"] and runs["I"]:
        qs = {q for (c, q) in final if c == "A"} & {q for (c, q) in final if c == "I"}
        mm = [q for q in qs if final[("A", q)]["dataset"] == "MMLU" and final[("A", q)]["gt"]]
        a_ok = {q: final[("A", q)]["final"] == final[("A", q)]["gt"] for q in mm}
        i_ok = {q: final[("I", q)]["final"] == final[("I", q)]["gt"] for q in mm}
        b = sum(1 for q in mm if i_ok[q] and not a_ok[q]); cc = sum(1 for q in mm if a_ok[q] and not i_ok[q])
        both = sum(1 for q in mm if a_ok[q] and i_ok[q]); neither = len(mm) - b - cc - both
        out["mcnemar"] = pd.DataFrame([dict(right_in_both=both, right_only_in_I=b, right_only_in_A=cc, wrong_in_both=neither,
                                            accuracy_A=np.mean(list(a_ok.values())) if mm else np.nan,
                                            accuracy_I=np.mean(list(i_ok.values())) if mm else np.nan,
                                            p_mcnemar=mcnemar_exact(b, cc))])
        mrows = []
        for dsn in ["MoralMachine", "MoralChoice"]:
            qd = [q for q in qs if final[("A", q)]["dataset"] == dsn]
            rev = [q for q in qd if final[("A", q)]["final"] != final[("I", q)]["final"]]
            to_a = sum(1 for q in rev if final[("I", q)]["final"] == "A")
            mrows.append(dict(dataset=dsn, questions=len(qd), decision_reversed=len(rev),
                              reversals_to_A=to_a, reversals_to_B=len(rev) - to_a,
                              p_direction_sign_test=sign_test(to_a, len(rev)) if rev else np.nan))
        out["moral_shifts"] = pd.DataFrame(mrows)
    return out


# ---------------------------------------------------------------- E5 / E10 (paired A vs I)
def an_e5_e10(V, B):
    out = {}
    if not {"A", "I"} <= set(V.cond.unique()):
        return out
    key = ["qid", "agent", "round"]
    a = V[V.cond == "A"].set_index(key); i = V[V.cond == "I"].set_index(key)
    j = a[["vote", "switched", "round_idx", "model"]].join(i[["vote", "switched"]], rsuffix="_I", how="inner")
    rows = []
    for rn, x in j.groupby(level="round"):
        row = dict(round=rn, round_idx=int(x.round_idx.iloc[0]), vote_differs=(x.vote != x.vote_I).mean())
        s = x.dropna(subset=["switched", "switched_I"])
        if len(s):
            bI = int(((s.switched_I == 1) & (s.switched == 0)).sum()); bA = int(((s.switched == 1) & (s.switched_I == 0)).sum())
            row.update(switch_differs=(s.switched != s.switched_I).mean(), switch_only_in_I=bI, switch_only_in_A=bA,
                       p_direction=mcnemar_exact(bI, bA))
        rows.append(row)
    out["by_round"] = pd.DataFrame(rows).sort_values("round_idx")
    first = []
    for q, x in j.groupby(level="qid"):
        diff = x[x.vote != x.vote_I]
        first.append(dict(qid=q, first_divergence=(diff.round_idx.min() if len(diff) else "never")))
    fd = pd.DataFrame(first)
    out["first_divergence"] = fd.groupby("first_divergence").size().rename("questions").reset_index()
    r0 = j.xs("R0", level="round")
    out["first_vote_by_model"] = r0.assign(d=(r0.vote != r0.vote_I)).groupby("model").d.mean().rename("share_first_vote_differs").reset_index()
    return out


# ---------------------------------------------------------------- E13 confidence
def an_e13(V, B):
    rows = []
    for c in conds_in(V):
        v = V[V.cond == c]
        g = v.groupby(["qid", "round"]).agg(conf=("confidence", "mean"), nA=("vote", lambda s: (s == "A").sum()), n=("vote", "size")).reset_index()
        g["majority_size"] = np.maximum(g.nA, g.n - g.nA)
        for ms, x in g.groupby("majority_size"):
            rows.append(dict(condition=c, majority_size=int(ms), mean_confidence=x.conf.mean(), question_rounds=len(x)))
    return {"confidence_vs_agreement": pd.DataFrame(rows)}


# ---------------------------------------------------------------- D4 single-change agents
def an_d4(V):
    rows = []
    for c in conds_in(V):
        v = V[(V.cond == c) & (V.is_decision == True)]
        sw = v[v.switched == 1]
        cnt = sw.groupby(["qid", "agent"]).size()
        once = cnt[cnt == 1].index
        e = sw.set_index(["qid", "agent"]).loc[once].reset_index() if len(once) else sw.iloc[0:0]
        for _, x in e.iterrows():
            rows.append(dict(condition=c, qid=x.qid, dataset=x.dataset, agent=x.agent, model=x.model, round=x["round"],
                             change=f"{x.prev_vote}->{x.vote}", direction="toward majority" if x.minority == 1 else "toward minority",
                             truth=("toward correct" if x.to_correct == 1 else "toward wrong") if pd.notna(x.to_correct) else "n/a"))
    E = pd.DataFrame(rows)
    out = {"events": E}
    if len(E):
        out["by_round"] = E.groupby(["condition", "round"]).size().rename("single_changes").reset_index()
        out["by_model"] = E.groupby(["condition", "model"]).size().rename("single_changes").reset_index()
    return out


# ---------------------------------------------------------------- H6 swap
def an_h6(V, B):
    out = {}
    S = V[(V.cond == "S") & (V.is_decision == True)]
    if not len(S):
        return out
    conf = S[S.shown_rich_side.notna() & S.rich_side.notna() & (S.shown_rich_side != S.rich_side)]
    on_true = conf[conf.prev_vote == conf.rich_side]          # could move to the side SHOWN as rich
    on_shown = conf[conf.prev_vote == conf.shown_rich_side]   # could move to the side that IS rich
    g1, g2 = q_sums(on_true, "switched"), q_sums(on_shown, "switched")
    d = boot_diff(g1, g2, B)
    out["pull"] = pd.DataFrame([dict(measure="Moves to the side SHOWN as rich", cases=len(on_true), rate=boot_ratio(g1, B)[0]),
                                dict(measure="Moves to the side that IS rich", cases=len(on_shown), rate=boot_ratio(g2, B)[0])])
    out["test"] = d
    qs = set(S.qid)
    ref = []
    for c in ["A", "I"]:
        x = V[(V.cond == c) & V.qid.isin(qs) & (V.h2_at_risk == 1)]
        if len(x):
            ref.append(dict(condition=c, measure="Moves to the truly rich side when it is the minority (same 60 questions)", cases=len(x), rate=x.switched.mean()))
    out["reference"] = pd.DataFrame(ref)
    return out


# =============================================================================
# 5. SUMMARY OF PRIMARY TESTS
# =============================================================================
WAIT_I = "waiting for identity data"
WAIT_S = "waiting for swap data"


def model_rows(res, terms, label):
    rows = []
    if not isinstance(res, dict):
        return pd.DataFrame()
    if "error" in res:
        return pd.DataFrame([dict(outcome=label, term="-", note=res["error"])])
    for t in terms:
        d = res.get(t, {})
        rows.append(dict(outcome=label, term=t, odds_ratio=d.get("odds_ratio"), ci_low=d.get("lo"), ci_high=d.get("hi"), p=d.get("p"),
                         observations=res.get("n"), questions=res.get("clusters")))
    if "joint_p" in res:
        rows.append(dict(outcome=label, term="joint test of: " + ", ".join(terms), p=res["joint_p"],
                         observations=res.get("n"), questions=res.get("clusters")))
    return pd.DataFrame(rows)


WAITING_I = "waiting for the Identity-revealed data"
WAITING_S = "waiting for the Swapped-identity data"


def round_text(rn):
    return f"Round {rn[1:]}" if rn.startswith("R") else f"Debate round {rn[1:]}"


def robust_text(rb, col, lab):
    """Short summary of the robustness versions for one component."""
    if rb is None or not len(rb):
        return None
    parts = []
    for r in rb[rb[col] == lab].itertuples():
        if r.version.startswith("Main"):
            continue
        name = "all messages kept" if "Check 1" in r.version else "without " + f"{ROBUSTNESS_DROP[0]} {ROBUSTNESS_DROP[1]:g}B"
        orv = f"{r.odds_ratio:.2f}" if pd.notna(r.odds_ratio) else "n/a"
        pv = f"{r.p:.3g}" if pd.notna(r.p) else "n/a"
        parts.append(f"{name}: OR {orv}, p {pv}")
    return "; ".join(parts)


def build_results(R_):
    """Every statistical test in one table. Each row: analysis, question, method, effect, p.
    p-values are adjusted together (Benjamini-Hochberg) across ALL tests in the table."""
    rows = []

    def add(analysis, question, method, effect, p, up, down, equiv=None, waiting=None, robust=None):
        rows.append(dict(analysis=analysis, question=question, method=method, effect=effect, p=p,
                         up_text=up, down_text=down, equiv_text=equiv, waiting_text=waiting, robustness_checks=robust))

    # Following the majority
    m = R_["H1"].get("model", {}).get("minority", {})
    add("Following the majority", "Do outnumbered agents switch to the majority more than majority agents switch away?",
        "Logistic regression on every vote change (both conditions); clustered by question; agent and dataset fixed effects",
        f"odds ratio {fmt_or(m)}", m.get("p", np.nan),
        "yes: outnumbered agents switch more", "no: majority agents switch more")
    m = R_["H1"].get("model_e11", {}).get("min_x_I") if R_["H1"].get("model_e11") else None
    if m:
        add("Following the majority", "Does showing identities change how strongly agents follow the majority?",
            "Same regression with an identity x minority term", f"odds ratio {fmt_or(m)}", m.get("p", np.nan),
            "identity strengthens following the majority", "identity weakens following the majority")
    else:
        add("Following the majority", "Does showing identities change how strongly agents follow the majority?", "-", "", np.nan, "", "", waiting=WAITING_I)
    # Votes vs reasons
    for r in R_["E1"]["headcount_vs_reasons"].itertuples():
        add("Votes vs reasons", f"[{COND_LABEL[r.condition]}] {r.measure}: does it differ between Round 1 (votes only) and Round 3 (votes + reasons)?",
            "Paired bootstrap over questions", f"Round 3 - Round 1 = {r.r3_minus_r1:+.3f} {r.ci}", r.p,
            "higher when reasons are visible", "higher when only votes are visible")
    # Whom by opinion
    for r in R_["E6"]["sides"].itertuples():
        add("Whom agents message, by opinion", f"[{COND_LABEL[r.condition]}] {r.measure.replace(' (O/E)', '')}: different from random choice?",
            "Observed / expected (O/E) vs 1, bootstrap over questions", f"O/E = {r.o_over_e:.2f} {r.ci}", r.p_vs_chance,
            "more than random", "less than random")
    # Joining large models / own family
    for key, name, q in [("H2", "Joining the large models", "Do agents join the large models' minority side more often when sizes are visible?"),
                         ("H3", "Joining one's own family", "Do agents join their own family's minority side more often when families are visible?")]:
        mm = R_[key].get("model")
        if mm:
            d = R_[key].get("paired_diff", {})
            add(name, q, "Logistic regression, Identity-revealed vs Anonymous; clustered by question; agent and dataset fixed effects",
                f"odds ratio {fmt_or(mm.get('cond_I'))}; rate difference {d.get('diff', np.nan):+.3f}",
                mm.get("cond_I", {}).get("p", np.nan), "yes: more often when identities are visible",
                "no: less often when identities are visible", equiv=equivalent(d.get("lo90", np.nan), d.get("hi90", np.nan)))
        else:
            add(name, q, "-", "", np.nan, "", "", waiting=WAITING_I)
    # Whom agents message
    h4a = R_["H4a"]
    for r in h4a["oe"].itertuples():
        add("Whom agents message", f"[{COND_LABEL[r.condition]}] {r.measure.replace(' (O/E)', '')}: different from random choice?",
            "Observed / expected (O/E) vs 1, bootstrap over questions", f"O/E = {r.o_over_e:.2f} {r.ci}", r.p_vs_chance,
            "more than random", "less than random")
    if h4a.get("model"):
        rb = h4a.get("robustness")
        for comp, txt, lab in [("family", "their own family", "own family"), ("rich", "large models", "large models")]:
            v = h4a["model"][comp]
            add("Whom agents message", f"Does showing identities change how often agents message {txt}?",
                "Logistic regression with the random-choice level as offset; clustered by question; speaker and dataset fixed effects; mismatched messages excluded",
                f"odds ratio {fmt_or(v)}", v.get("p", np.nan), "more often when identities are visible", "less often when identities are visible",
                robust=robust_text(rb, "messages_to", lab))
    else:
        add("Whom agents message", "Does showing identities change whom agents message?", "-", "", np.nan, "", "", waiting=WAITING_I)
    # Whom agents answer
    bm = R_["H4b"].get("baseline_model_A", {})
    for t, txt in [("same_family", "the sender is from the same family"), ("sender_rich", "the sender is a large model"),
                   ("same_vote", "the sender holds the same vote")]:
        if isinstance(bm.get(t), dict):
            add("Whom agents answer", f"[Anonymous] Do agents reply more when {txt}?",
                "Logistic regression on reply/decline; clustered by question; replier and dataset fixed effects",
                f"odds ratio {fmt_or(bm[t])}", bm[t].get("p", np.nan), "yes: they reply more", "no: they reply less")
    m = R_["H4b"].get("model")
    if m:
        rb = R_["H4b"].get("robustness")
        for t, txt, lab in [("I_x_family", "their own family", "own family"), ("I_x_rich", "large models", "large models")]:
            add("Whom agents answer", f"Does showing identities change how often agents reply to {txt}?",
                "Logistic regression with identity x sender terms; controls for whether sender and replier agree; responses to mismatched messages excluded",
                f"odds ratio {fmt_or(m.get(t))}", m.get(t, {}).get("p", np.nan),
                "more often when identities are visible", "less often when identities are visible",
                robust=robust_text(rb, "replies_to", lab))
    else:
        add("Whom agents answer", "Does showing identities change whom agents answer?", "-", "", np.nan, "", "", waiting=WAITING_I)
    # Final decision
    mc = R_["H5"].get("mcnemar")
    if mc is not None and len(mc):
        x = mc.iloc[0]
        add("The group's final decision", "Does showing identities change the group's accuracy on MMLU?",
            "Paired McNemar test on the same questions",
            f"accuracy {x.accuracy_A:.3f} -> {x.accuracy_I:.3f} (right only with identities: {int(x.right_only_in_I)}; only without: {int(x.right_only_in_A)})",
            x.p_mcnemar, "identity changes accuracy", "identity changes accuracy")
        for r in R_["H5"]["moral_shifts"].itertuples():
            add("The group's final decision", f"[{r.dataset}] When the decision changes, does it move toward one option?",
                "Sign test on reversed decisions", f"{r.decision_reversed} reversed: {r.reversals_to_A} to A, {r.reversals_to_B} to B",
                r.p_direction_sign_test, "moves toward one option", "moves toward one option")
    else:
        add("The group's final decision", "Does showing identities change the group's decision?", "-", "", np.nan, "", "", waiting=WAITING_I)
    # Identity sensitivity
    e5 = R_["E5"].get("by_round")
    if e5 is not None and "p_direction" in e5:
        for r in e5.dropna(subset=["p_direction"]).itertuples():
            add("Identity sensitivity", f"{round_text(r.round)}: do agents change their vote more often when identities are shown?",
                "Paired McNemar on the same agent and question", f"changed only with identities: {int(r.switch_only_in_I)}; only without: {int(r.switch_only_in_A)}",
                r.p_direction, "more changes with identities", "fewer changes with identities")
    else:
        add("Identity sensitivity", "Do agents change their vote more often when identities are shown?", "-", "", np.nan, "", "", waiting=WAITING_I)
    # Name or model
    t = R_["H6"].get("test")
    if t:
        add("Name or model", "With false names, do agents follow the side shown as large more than the side that is large?",
            "Paired bootstrap over questions", f"difference {t['diff']:+.3f} [{t['lo95']:.3f}, {t['hi95']:.3f}]", t["p"],
            "agents follow the name", "agents follow the model", equiv=equivalent(t["lo90"], t["hi90"]))
    else:
        add("Name or model", "With false names, do agents follow the name or the model?", "-", "", np.nan, "", "", waiting=WAITING_S)

    df = pd.DataFrame(rows)
    adj = bh({i: p for i, p in enumerate(df.p) if p is not None and np.isfinite(p)})
    df["p_adjusted"] = [adj.get(i, np.nan) for i in range(len(df))]

    def sign_of(effect):
        mt = re.search(r"odds ratio ([\d.]+)", str(effect))
        if mt:
            return float(mt.group(1)) - 1
        mt = re.search(r"([+-]\d*\.\d+)", str(effect))
        if mt:
            return float(mt.group(1))
        mt = re.search(r"O/E = ([\d.]+)", str(effect))
        if mt:
            return float(mt.group(1)) - 1
        mt = re.search(r"only with identities: (\d+(?:\.\d+)?); only without: (\d+(?:\.\d+)?)", str(effect))
        if mt:
            return float(mt.group(1)) - float(mt.group(2))
        return 1.0

    res = []
    for r in df.itertuples():
        if isinstance(r.waiting_text, str) and r.waiting_text:
            res.append(r.waiting_text)
        elif not np.isfinite(r.p_adjusted):
            res.append("not computable")
        elif r.p_adjusted < 0.05:
            res.append("CLEAR EFFECT: " + (r.up_text if sign_of(r.effect) >= 0 else r.down_text))
        elif isinstance(r.equiv_text, str) and r.equiv_text.startswith("yes"):
            res.append("NO MEANINGFUL DIFFERENCE (within +/-5 points)")
        else:
            res.append("no clear effect")
    df["result"] = res
    return df[["analysis", "question", "method", "effect", "p", "p_adjusted", "result", "robustness_checks"]]


# =============================================================================
# 6. FIGURES
# =============================================================================
def make_figures(R_, figdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(figdir, exist_ok=True)
    made = []
    cur = R_["H1"]["curve"]
    if len(cur):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
        for ax, st in zip(axes, ["Round 1 (votes only)", "Round 3 (votes + reasons)", "Debate"]):
            for c in sorted(cur.condition.unique()):
                x = cur[(cur.condition == c) & (cur.stage == st)]
                ax.plot(x.majority_size, x.conformity, marker="o", label=f"{COND_LABEL[c]}: minority -> majority")
                ax.plot(x.majority_size, x.defection, marker="x", linestyle="--", label=f"{COND_LABEL[c]}: majority -> minority")
            ax.set_title(st, fontsize=10); ax.set_xlabel("split before the round"); ax.grid(alpha=.3)
        axes[0].set_ylabel("switch rate"); axes[0].legend(fontsize=7)
        fig.tight_layout(); p = os.path.join(figdir, "H1_conformity_curve.png"); fig.savefig(p, dpi=150); plt.close(fig); made.append(p)
    ov = R_["OV"]["by_round"]
    if len(ov) and "switch_rate" in ov:
        fig, ax = plt.subplots(figsize=(7, 3.2))
        for c in sorted(ov.condition.unique()):
            x = ov[(ov.condition == c)].dropna(subset=["switch_rate"])
            ax.plot(x["round"], x.switch_rate, marker="o", label=COND_LABEL[c])
        ax.set_ylabel("share of agents switching"); ax.grid(alpha=.3); ax.legend(); fig.tight_layout()
        p = os.path.join(figdir, "switch_rate_by_round.png"); fig.savefig(p, dpi=150); plt.close(fig); made.append(p)
    for c in ["A", "I", "S"]:
        fm = R_["H4a"].get(f"family_matrix_{c}")
        if fm is not None and len(fm):
            mat = fm.set_index(fm.columns[0])
            fig, ax = plt.subplots(figsize=(4, 3.4))
            im = ax.imshow(mat.values.astype(float), cmap="coolwarm", vmin=0, vmax=2)
            ax.set_xticks(range(len(mat.columns))); ax.set_xticklabels(mat.columns, rotation=30)
            ax.set_yticks(range(len(mat.index))); ax.set_yticklabels(mat.index)
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    ax.text(j, i, f"{mat.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
            ax.set_title(f"Messages family -> family, O/E ({COND_LABEL[c]})", fontsize=9); fig.colorbar(im, ax=ax, fraction=.046)
            fig.tight_layout(); p = os.path.join(figdir, f"H4a_family_matrix_{c}.png"); fig.savefig(p, dpi=150); plt.close(fig); made.append(p)
    return made


def question_graph(run, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    agents = run["agents"]; N = len(agents)
    pos = {a: (math.cos(math.pi / 2 - 2 * math.pi * k / N), math.sin(math.pi / 2 - 2 * math.pi * k / N)) for k, a in enumerate(agents)}
    hist = {a: [vm[a][0] for _, vm in run["rounds"]] for a in agents}
    col = {a: ("#F4C430" if set(h) == {"A"} else "#4CAF50" if set(h) == {"B"} else "#E53935") for a, h in hist.items()}
    replied = {}
    for rn, reps in run["replies"].items():
        for r in reps:
            replied[(rn, r["sender_agent_id"], r["message_idx"])] = r.get("decision") == "reply"
    fig, ax = plt.subplots(figsize=(7, 7))
    pair = Counter()
    for rn, sts in run["statements"].items():
        for s in sts:
            u, v = s["speaker_agent_id"], s["target_agent_id"]
            if u not in pos or v not in pos:
                continue
            k = pair[(u, v)]; pair[(u, v)] += 1
            ok = replied.get((rn, u, s.get("statement_idx")), True)
            ax.add_patch(FancyArrowPatch(pos[u], pos[v], connectionstyle=f"arc3,rad={0.1 + 0.06 * k}", arrowstyle="-|>",
                                         mutation_scale=9, lw=.8, color="#555", alpha=.5, linestyle="-" if ok else (0, (3, 3)),
                                         shrinkA=18, shrinkB=18))
    for a in agents:
        x, y = pos[a]
        ax.scatter([x], [y], s=900, c=col[a], edgecolors="#333", zorder=3)
        ax.text(x * 1.28, y * 1.28, f"{a}\n{run['info'][a]['model']}", ha="center", va="center", fontsize=8, fontweight="bold")
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6); ax.axis("off")
    ax.set_title(f"{run['qid']} ({COND_LABEL[run['cond']]}): solid = replied, dashed = declined", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)


# =============================================================================
# 7. EXCEL WRITER
# =============================================================================
class Report:
    def __init__(self, path):
        self.writer = pd.ExcelWriter(path, engine="xlsxwriter")
        wb = self.writer.book
        self.f_title = wb.add_format({"bold": True, "font_size": 13, "font_color": "#1F3864"})
        self.f_note = wb.add_format({"italic": True, "font_color": "#595959", "text_wrap": True, "valign": "top"})
        self.f_head = wb.add_format({"bold": True, "font_color": "#FFFFFF", "bg_color": "#2F5597"})
        self.f_sub = wb.add_format({"bold": True, "font_color": "#1F3864"})

    def sheet(self, name, title, notes, blocks):
        """blocks: list of (subtitle, DataFrame)."""
        name = name[:31]
        ws = self.writer.book.add_worksheet(name)
        self.writer.sheets[name] = ws
        ws.write(0, 0, title, self.f_title)
        r = 1
        for n in notes:
            ws.merge_range(r, 0, r, 9, n, self.f_note); ws.set_row(r, 15 * max(1, math.ceil(len(n) / 140))); r += 1
        r += 1
        for sub, df in blocks:
            if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                ws.write(r, 0, f"{sub}: not available yet", self.f_sub); r += 2; continue
            ws.write(r, 0, sub, self.f_sub); r += 1
            df = df.copy()
            for c in df.columns:
                if df[c].dtype == float:
                    df[c] = df[c].round(4)
            df.to_excel(self.writer, sheet_name=name, startrow=r, index=False)
            for j, c in enumerate(df.columns):
                ws.write(r, j, str(c), self.f_head)
            r += len(df) + 3
        ws.set_column(0, 0, 34); ws.set_column(1, 25, 16)

    def close(self):
        self.writer.close()


# =============================================================================
# 8. MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--no-per-question", action="store_true")
    ap.add_argument("--graphs", action="store_true", help="also draw one network graph per question and condition")
    ap.add_argument("--all-questions", action="store_true",
                    help="do NOT restrict to questions that are complete in both conditions (not recommended)")
    args = ap.parse_args()
    t0 = time.time()
    root = os.path.abspath(args.root)
    outdir = os.path.join(root, "analysis_output"); csvdir = os.path.join(outdir, "csv"); figdir = os.path.join(outdir, "figures")
    os.makedirs(csvdir, exist_ok=True)

    runs, skipped = {}, {}
    for folder, c in COND_NAMES.items():
        p = os.path.join(root, folder)
        if os.path.isdir(p):
            runs[c], skipped[c] = load_condition(p, c)
            print(f"[load] {COND_LABEL[c]}: {len(runs[c])} runs ({len(skipped[c])} files skipped)")
    if not any(runs.values()):
        sys.exit("No run files found. Expected sub-folders: no_identity/, identity/, swap/")

    # ---- completeness and pairing (fixes 4 and 5) ----
    def complete(r):
        return len(r["rounds"]) == 15 and len(r["statements"]) == 11 and len(r["agents"]) == 11
    comp_rows, excluded = [], []
    complete_ids = {}
    for c, rs in runs.items():
        bad = [r["qid"] for r in rs if not complete(r)]
        excluded += [dict(condition=COND_LABEL[c], question=q, reason="incomplete run (fewer than 15 vote rounds)") for q in bad]
        complete_ids[c] = {r["qid"] for r in rs if complete(r)}
        expected_cond = {"A": "anonymous", "I": "real_full_identity", "S": "real_full_identity_swapped"}[c]
        wrong = [r["qid"] for r in rs if r["identity_condition"] and r["identity_condition"] != expected_cond]
        if wrong:
            print(f"[warning] {len(wrong)} files in the {COND_LABEL[c]} folder report a different condition "
                  f"(e.g. {wrong[0]}: {[r['identity_condition'] for r in rs if r['qid'] == wrong[0]][0]})")
    keep = {c: set(ids) for c, ids in complete_ids.items()}
    if not args.all_questions and "A" in keep and "I" in keep and keep["A"] and keep["I"]:
        paired = keep["A"] & keep["I"]
        for c in ("A", "I"):
            excluded += [dict(condition=COND_LABEL[c], question=q, reason="no complete partner in the other condition")
                         for q in sorted(keep[c] - paired)]
            keep[c] = paired
    for c in list(runs):
        before = len(runs[c])
        runs[c] = [r for r in runs[c] if r["qid"] in keep.get(c, set())]
        print(f"[filter] {COND_LABEL[c]}: {before} loaded -> {len(runs[c])} analysed")
    for c, rs in runs.items():
        cnt = Counter(r["dataset"] for r in rs)
        for d in sorted(cnt):
            comp_rows.append(dict(condition=COND_LABEL[c], dataset=d, questions_analysed=cnt[d]))
        comp_rows.append(dict(condition=COND_LABEL[c], dataset="TOTAL", questions_analysed=len(rs)))
    composition = pd.DataFrame(comp_rows)
    excluded_df = pd.DataFrame(excluded)
    allruns = [r for rs in runs.values() for r in rs]
    V, M, R = build_all(allruns)
    roster_info = allruns[0]["info"]
    print(f"[tables] votes {len(V):,} | messages {len(M):,} | responses {len(R):,}  ({time.time() - t0:.0f}s)")

    B = args.boot
    R_ = {}
    R_["CHK"] = an_checks(runs, V, M, R, skipped)
    R_["OV"] = an_overview(V, B)
    R_["H1"] = an_h1(V, B)
    R_["E1"] = an_e1(V, B)
    R_["E6"] = an_e6(M, B)
    R_["E9"] = an_e9(V, B)
    R_["E3"] = an_e3(V, B)
    R_["H2"] = an_side_pull(V, B, "h2_at_risk", "rich_tie", "rich")
    R_["H3"] = an_side_pull(V, B, "h3_at_risk", "fam_tie", "family")
    R_["H4a"] = an_h4a(M, roster_info, B)
    R_["H4b"] = an_h4b(R, B)
    R_["H5"] = an_h5(runs, B)
    R_["E5"] = an_e5_e10(V, B)
    R_["E13"] = an_e13(V, B)
    R_["D4"] = an_d4(V)
    R_["H6"] = an_h6(V, B)
    R_["CONS"] = an_consistency(M, R)
    results = build_results(R_)
    print(f"[analyses] done ({time.time() - t0:.0f}s)")

    # CSV exports
    V.drop(columns=["rationale"]).to_csv(os.path.join(csvdir, "votes.csv"), index=False)
    M.to_csv(os.path.join(csvdir, "messages.csv"), index=False)
    R.to_csv(os.path.join(csvdir, "responses.csv"), index=False)
    results.to_csv(os.path.join(csvdir, "all_results.csv"), index=False)
    slug = {"CHK": "data_checks", "OV": "round_by_round", "H1": "following_majority", "E1": "votes_vs_reasons",
            "E6": "whom_by_opinion", "E9": "return_to_first_answer", "E3": "why_leave_majority", "H2": "joining_large_models",
            "H3": "joining_own_family", "H4a": "whom_agents_message", "H4b": "whom_agents_answer", "H5": "final_decision",
            "E5": "identity_sensitivity", "E13": "confidence", "D4": "one_time_changes", "H6": "name_or_model",
            "CONS": "message_consistency"}
    for key, res in R_.items():
        for name, df in res.items():
            if isinstance(df, pd.DataFrame) and len(df):
                df.to_csv(os.path.join(csvdir, f"{slug.get(key, key)}__{name}.csv"), index=False)
    figs = make_figures(R_, figdir)

    def info(question, method, insight, interpret):
        return [f"QUESTION: {question}", f"METHOD: {method}", f"INSIGHT: {insight}", f"HOW TO INTERPRET: {interpret}"]

    rep = Report(os.path.join(outdir, "curtain_analysis.xlsx"))
    avail = ", ".join(COND_LABEL[c] for c in runs if runs[c])
    guess = sum(r["had_guess_round"] for r in allruns)
    guide = pd.DataFrame([
        ("01_SAMPLE", "Which questions are analysed, and which were left out?"),
        ("02_DATA_CHECKS", "Is the data complete and correct?"),
        ("03_ROUND_BY_ROUND", "What happens in each round?"),
        ("04_FOLLOWING_MAJORITY", "Do outnumbered agents give in to the majority?"),
        ("05_VOTES_VS_REASONS", "Do agents give in more to numbers or to reasons?"),
        ("06_WHOM_BY_OPINION", "Do agents message those who agree or those who disagree?"),
        ("07_RETURN_TO_FIRST_ANSWER", "Do agents return to their first answer?"),
        ("08_JOINING_LARGE_MODELS", "Do agents join the large models more when sizes are visible?"),
        ("09_JOINING_OWN_FAMILY", "Do agents join their own family more when families are visible?"),
        ("10_WHY_LEAVE_MAJORITY", "When agents leave the majority, whose side do they join?"),
        ("11_WHOM_AGENTS_MESSAGE", "Does identity change whom agents message?"),
        ("12_WHOM_AGENTS_ANSWER", "Does identity change whom agents answer?"),
        ("13_FINAL_DECISION", "Does identity change the group's final decision?"),
        ("14_IDENTITY_SENSITIVITY", "How much does behaviour change when identities are shown?"),
        ("15_CONFIDENCE", "Does confidence rise with agreement?"),
        ("16_ONE_TIME_CHANGES", "When do agents change their mind only once?"),
        ("17_NAME_OR_MODEL", "With false names, do agents follow the name or the model?"),
        ("18_DECLINE_REASONS", "Why do agents decline to answer?"),
        ("19_MESSAGE_CONSISTENCY", "Do messages address the agent they were sent to?"),
        ("Q_<question>", "Every agent's vote in every round, both conditions"),
    ], columns=["sheet", "question it answers"])
    rep.sheet("00_ALL_RESULTS", "All results",
              [f"Conditions analysed: {avail}. Only questions complete in BOTH conditions are analysed (see 01_SAMPLE).",
               "Every statistical test is listed here. p_adjusted corrects all tests together (Benjamini-Hochberg).",
               "CLEAR EFFECT = p_adjusted < 0.05. NO MEANINGFUL DIFFERENCE = the 90% interval of the difference lies within +/-5 percentage points.",
               "Regressions are clustered by question and include agent and dataset fixed effects, because the datasets have different numbers of questions.",
               f"Note: {guess} run(s) contain a model-guessing round; it is ignored. Runs made with that code version told agents in advance that they would guess models."],
              [("Results", results), ("Guide to the sheets", guide)])
    rep.sheet("01_SAMPLE", "Sample",
              info("Which questions does each result rest on, and which were left out?",
                   "Only runs with all 15 voting rounds are used, and only questions complete in both conditions, so every comparison is between the same questions.",
                   "The exact sample behind every number in this report, per condition and dataset.",
                   "Datasets have different sizes; results are therefore also given per dataset and as a dataset-balanced average."),
              [("Questions analysed", composition), ("Excluded questions", excluded_df if len(excluded_df) else pd.DataFrame([dict(note="none excluded")]))])
    rep.sheet("02_DATA_CHECKS", "Data checks",
              info("Is the data complete and correct?", "Counts of agents, rounds, messages, responses, retries and label mentions in the analysed runs.",
                   "Whether the results can be trusted.", "Every row should read PASS."),
              [("Checks", R_["CHK"]["checks"])])
    rep.sheet("03_ROUND_BY_ROUND", "Round by round",
              info("What happens in each round?", "For each round and condition: share of agents changing their vote, share of unanimous questions, average majority size, confidence and (MMLU) individual accuracy.",
                   "When the group settles, how often agents change their minds, and how confidence evolves.",
                   "Read down a condition to follow a discussion over time; compare the two conditions row by row."),
              [("By round", R_["OV"]["by_round"])])
    rep.sheet("04_FOLLOWING_MAJORITY", "Following the majority",
              info("Do agents in the minority switch to the majority more often than agents in the majority switch away?",
                   "Each rate is divided by the agents who could make that move, using positions at the end of the previous round. Rates by stage, by size of the majority, per model and per dataset. Logistic regression on every vote change, clustered by question, with agent and dataset fixed effects.",
                   "Whether LLM groups show the human tendency to conform, and in which stage it is strongest.",
                   "Minority-to-majority rate well above majority-to-minority rate = agents follow the majority. A curve rising with the size of the majority = larger majorities pull harder. Odds ratio above 1 = minority agents are more likely to switch."),
              [("Rates by stage", R_["H1"]["rates"]), ("Rates per dataset (pooled and dataset-balanced)", R_["H1"].get("per_dataset")),
               ("Rates by size of the majority", R_["H1"]["curve"]), ("Rates per model", R_["H1"]["per_model"]),
               ("Regression: both conditions together", model_rows(R_["H1"].get("model"), ["minority"], "vote changed")),
               ("Regression: does showing identities change it?", model_rows(R_["H1"].get("model_e11"), ["min_x_I"], "vote changed"))])
    rep.sheet("05_VOTES_VS_REASONS", "Votes vs reasons",
              info("Do agents give in more when they see only how others voted (Round 1) or also their reasons (Round 3)?",
                   "Share of minority agents who switch at Round 1 vs Round 3, within each condition, paired by question; for MMLU, the share of switches that go to the correct answer.",
                   "Whether influence works through numbers alone or through arguments.",
                   "Round 3 minus Round 1 above 0 = reasons add pull. A higher share toward the correct answer at Round 3 = reasons help accuracy."),
              [("Comparison", R_["E1"]["headcount_vs_reasons"])])
    rep.sheet("06_WHOM_BY_OPINION", "Whom agents message, by opinion",
              info("Do agents message those who agree with them, or those who disagree, especially lone dissenters?",
                   "Observed messages divided by the number expected if targets were chosen at random, given how many agents are on each side (O/E). Unanimous rounds are excluded.",
                   "How the group directs its attention during the debate.",
                   "O/E = 1: as random. Same-side O/E above 1: agents talk to those who agree. Dissenter O/E above 1: the group concentrates on dissenters."),
              [("Overall", R_["E6"]["sides"]), ("Per dataset", R_["E6"].get("per_dataset")), ("Per round", R_["E6"]["per_round"])])
    rep.sheet("07_RETURN_TO_FIRST_ANSWER", "Return to the first answer",
              info("Do agents that gave in at Round 1 later return to their own first answer?",
                   "Among agents whose Round 1 vote differs from their Round 0 vote: share back at Round 3, at any later round, and at the end.",
                   "How durable social influence is.",
                   "High values = influence is temporary. Agents are never shown their Round 0 vote again, so a return reflects the model's own preference."),
              [("Return to the first answer", R_["E9"]["rebound"])])
    for key, sh, title, q, cases in [
            ("H2", "08_JOINING_LARGE_MODELS", "Joining the large models (10B or more)",
             "Do agents join the side of the large models more often when sizes are visible?",
             "Cases: the large models (10B or more, excluding the agent itself) are on the group's minority side and the agent is on the majority side. Outcome: the agent switches to the large models' side."),
            ("H3", "09_JOINING_OWN_FAMILY", "Joining one's own family",
             "Do agents join their own family's side more often when families are visible?",
             "Cases: the agent's family (excluding itself) is on the group's minority side and the agent is on the majority side. Outcome: the agent switches to its family's side. Family ties are excluded and counted.")]:
        blocks = [("Rates", R_[key]["rates"]), ("Rates per dataset (pooled and dataset-balanced)", R_[key].get("per_dataset")),
                  ("Rates by stage", R_[key]["by_stage"])]
        if R_[key].get("model"):
            d = R_[key]["paired_diff"]
            blocks += [("Regression: Identity-revealed vs Anonymous", model_rows(R_[key]["model"], ["cond_I"], "switched to that side")),
                       ("Difference in rates (Identity-revealed minus Anonymous)", pd.DataFrame([dict(difference=d["diff"], ci95_low=d["lo95"], ci95_high=d["hi95"],
                                                                                                          ci90_low=d["lo90"], ci90_high=d["hi90"], p=d["p"],
                                                                                                          within_5_points=equivalent(d["lo90"], d["hi90"]))]))]
        rep.sheet(sh, title, info(q, cases + " Only cases where the group is on the other side are used, so this cannot be confused with following the majority. Anonymous vs Identity-revealed; logistic regression clustered by question with agent and dataset fixed effects.",
                                  "The effect of the labels themselves on whose side agents join.",
                                  "Odds ratio above 1 = more often when identities are visible. The Anonymous rate is the baseline: similar models may agree even when names are hidden."),
                  blocks)
    rep.sheet("10_WHY_LEAVE_MAJORITY", "Why agents leave the majority",
              info("When an agent leaves the majority, whose side does it join?",
                   "Every majority-to-minority switch, classified as: own family only, large models only, both, or neither; Anonymous vs Identity-revealed.",
                   "What explains going against the majority.",
                   "A higher share of 'family' or 'rich' when identities are visible = the labels pull agents away from the crowd."),
              [("Categories", R_["E3"]["minority_moves"])])
    h4a = R_["H4a"]
    blocks = [("Compared with random choice", h4a["oe"])] + [(f"Family -> family, O/E ({COND_LABEL[c]})", h4a.get(f"family_matrix_{c}")) for c in runs if runs[c]] \
             + [(f"Large/small, O/E ({COND_LABEL[c]})", h4a.get(f"rich_matrix_{c}")) for c in runs if runs[c]]
    if h4a.get("model"):
        m = h4a["model"]
        blocks.append(("Regression: Identity-revealed vs Anonymous (main: mismatched messages excluded)",
                       pd.DataFrame([dict(component=k, odds_ratio=v["odds_ratio"], ci_low=v["lo"], ci_high=v["hi"], p=v["p"])
                                     for k, v in m.items() if isinstance(v, dict)])))
        blocks.append(("Robustness checks: the same regression on different sets of messages", h4a.get("robustness")))
    rep.sheet("11_WHOM_AGENTS_MESSAGE", "Whom agents message",
              info("Does seeing identities change whom agents choose to message (their own family, the large models)?",
                   "Messages to the same family and to large models, compared with random choice (O/E; the expectation accounts for family sizes 3,3,3,2 and excludes the speaker); Anonymous vs Identity-revealed; logistic regression with the random-choice level as an offset. Messages whose text addresses a different agent than their target are excluded (see 19_MESSAGE_CONSISTENCY); two robustness checks repeat the regression with all messages, and without Qwen 2.5 3B as a sender.",
                   "Whether identity shapes who talks to whom.",
                   "Diagonal O/E above 1 = agents message their own family more than chance. A rise from Anonymous to Identity-revealed = the labels cause it."),
              blocks)
    h4b = R_["H4b"]
    blocks = [("Reply rates", h4b["rates"]), ("Reply rates per dataset (pooled and dataset-balanced)", h4b.get("per_dataset"))] \
             + [(f"Reply rate: replier family x sender family ({COND_LABEL[c]})", h4b.get(f"reply_matrix_{c}")) for c in runs if runs[c]]
    if h4b.get("baseline_model_A"):
        blocks.append(("Regression, Anonymous only (names hidden)", model_rows(h4b["baseline_model_A"], ["same_family", "sender_rich", "same_vote"], "replied")))
    if h4b.get("model"):
        blocks.append(("Regression: Identity-revealed vs Anonymous (main: responses to mismatched messages excluded)", model_rows(h4b["model"], ["I_x_family", "I_x_rich"], "replied")))
        blocks.append(("Robustness checks: the same regression on different sets of responses", h4b.get("robustness")))
    rep.sheet("12_WHOM_AGENTS_ANSWER", "Whom agents answer",
              info("Does seeing identities change whom agents reply to and whom they decline?",
                   "Reply rate = replies / messages received, by the sender's family (same/other), size (large/small) and vote (same/opposite); Anonymous vs Identity-revealed; logistic regression controlling for whether sender and replier agree, and for dataset. Responses to messages whose text addressed a different agent are excluded, because the receiver was answering a message meant for someone else; two robustness checks repeat the regression with all responses, and without responses to Qwen 2.5 3B.",
                   "Whether identity shapes who engages with whom.",
                   "Odds ratio above 1 for I_x_family = with identities visible, agents reply to their own family more; I_x_rich likewise for large senders."),
              blocks)
    rep.sheet("13_FINAL_DECISION", "The group's final decision",
              info("Does seeing identities change what the group finally decides?",
                   "Group decision = majority vote after the last debate round, same questions in both conditions. MMLU: accuracy, paired McNemar test. Moral datasets: how many decisions reverse, and in which direction (sign test).",
                   "Whether identity makes the group more or less accurate, and whether it shifts moral decisions.",
                   "McNemar uses only questions whose outcome differs: more 'right only with identities' than 'right only without' = identities help."),
              [("Outcomes by condition", R_["H5"]["outcomes"]), ("MMLU: paired comparison (McNemar)", R_["H5"].get("mcnemar")),
               ("Moral datasets: reversed decisions", R_["H5"].get("moral_shifts"))])
    rep.sheet("14_IDENTITY_SENSITIVITY", "Identity sensitivity",
              info("How often does the same agent, on the same question, act differently once identities are shown?",
                   "Paired by question, agent and round: share of votes that differ, share of vote changes that differ, and the first round where the two discussions differ.",
                   "The overall sensitivity of the group to identity.",
                   "Differences at Round 0 = labels change even independent judgement. Later rounds also include the effect of earlier differences in the conversation."),
              [("By round", R_["E5"].get("by_round")), ("First round where the two discussions differ", R_["E5"].get("first_divergence")),
               ("First vote differs, by model", R_["E5"].get("first_vote_by_model"))])
    rep.sheet("15_CONFIDENCE", "Confidence and agreement",
              info("Does confidence rise as the group agrees more?", "Mean confidence at each level of agreement (majority size after the round), per condition.",
                   "Whether agreement itself makes agents more certain.",
                   "Confidence rising with agreement, even on wrong answers, suggests the group grows more certain simply by agreeing."),
              [("Confidence vs agreement", R_["E13"]["confidence_vs_agreement"])])
    rep.sheet("16_ONE_TIME_CHANGES", "Agents who changed their mind once",
              info("When an agent changes its vote only once, in which round and in which direction?",
                   "All single changes, counted by round and by model, per condition.",
                   "Which stage of the discussion is decisive.", "Peaks show the rounds that move opinions most."),
              [("By round", R_["D4"].get("by_round")), ("By model", R_["D4"].get("by_model")), ("All events", R_["D4"].get("events"))])
    rep.sheet("17_NAME_OR_MODEL", "Name or model",
              info("With false names, do agents follow the side SHOWN as large or the side that IS large?",
                   "Swapped-identity runs: cases where the two sides differ; rates among agents who could move to each side.",
                   "Whether deference comes from the name or from the model behind it.",
                   "Moving to the shown-large side more often = the name drives behaviour."),
              [("Pull of each side", R_["H6"].get("pull")),
               ("Difference (shown minus true)", pd.DataFrame([R_["H6"]["test"]]) if R_["H6"].get("test") else None),
               ("Same questions, other conditions", R_["H6"].get("reference"))])
    dec = R[R.replied == 0][["cond", "qid", "dataset", "round", "replier_model", "sender_model", "same_vote", "text"]] if len(R) else None
    rep.sheet("18_DECLINE_REASONS", "Decline reasons",
              info("Why do agents decline to answer a message?", "Every decline with its stated reason.",
                   "The reasons agents give for not engaging.", "Group the reasons into types (for example: already agree, weak argument)."),
              [("Declines", dec)])
    rep.sheet("19_MESSAGE_CONSISTENCY", "Message consistency",
              info("Do messages address the agent they were actually sent to?",
                   "A message is 'mismatched' when its text opens by addressing a different agent ('To Agent_09: ...') than the target the agent selected. 'Exact copy' = the same text the agent already sent in an earlier round. Counted per sender model and condition; responses to mismatched messages are compared with normal ones.",
                   "How reliable the record of who-messaged-whom is, and whether showing identities changes how coherently agents communicate.",
                   "A high mismatched share for a model means its choice of target cannot be read reliably; those messages are excluded from the messaging and answering analyses. A large difference between conditions is itself an effect of showing identities."),
              [("By sender model", R_["CONS"]["by_sender"]), ("Responses to mismatched vs normal messages", R_["CONS"]["responses"])])

    if not args.no_per_question:
        gdir = os.path.join(outdir, "graphs")
        if args.graphs:
            os.makedirs(gdir, exist_ok=True)
        by_q = defaultdict(dict)
        for r in allruns:
            by_q[r["qid"]][r["cond"]] = r
        for qid in sorted(by_q):
            blocks = []
            for c, r in sorted(by_q[qid].items()):
                rows = []
                for a in r["agents"]:
                    seq = [vm[a][0] for _, vm in r["rounds"]]
                    grp = "always A" if set(seq) == {"A"} else "always B" if set(seq) == {"B"} else "changed"
                    rows.append(dict(agent=a, model=r["info"][a]["model"], **{rn: vm[a][0] for rn, vm in r["rounds"]}, group=grp))
                blocks.append((f"{COND_LABEL[c]}: votes in every round", pd.DataFrame(rows)))
                n_rep = sum(1 for reps in r["replies"].values() for x in reps if x.get("decision") == "reply")
                n_msg = sum(len(x) for x in r["statements"].values())
                blocks.append((f"{COND_LABEL[c]}: messages and responses", pd.DataFrame([dict(messages=n_msg, replied=n_rep, declined=n_msg - n_rep)])))
                if args.graphs:
                    question_graph(r, os.path.join(gdir, f"{qid}_{c}.png"))
            r0 = next(iter(by_q[qid].values()))
            rep.sheet(f"Q_{qid}"[:31], f"{qid} ({r0['dataset']} / {r0['subtype']})",
                      [f"Question: {r0['question'][:600]}", f"A: {r0['option_a'][:200]}   |   B: {r0['option_b'][:200]}   |   correct: {r0['gt'] or 'n/a'}"],
                      blocks)
    rep.close()
    print(f"[done] {time.time() - t0:.0f}s -> {outdir}")
    print(results[["analysis", "result"]].drop_duplicates().to_string(index=False))


if __name__ == "__main__":
    main()
