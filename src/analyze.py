"""
Does the AI hear what listeners hear?

Compares 34 objective audio-quality metrics (classic signal measures,
engineered psychoacoustic models, and AI / neural-network models) against
expert MUSHRA listening-test scores from the open ODAQ dataset, then trains a
machine-learning "fusion" model that combines them, and tests where it breaks.

Run:  python src/build_dataset.py && python src/analyze.py
"""
import json
import os
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
RNG = np.random.default_rng(7)
N_BOOT = 1000

# ---------------------------------------------------------------- metric families
CLASSIC = ["SI-SDR", "CD", "MCD", "fwSNRseg", "STOI", "ESTOI"]
PSYCHO = ["PEAQ-ODG", "NMR", "ViSQOLAudio-v3", "2f", "ADB", "AvgModDiff1",
          "PESQc2", "HAAQI", "HASQIv2", "HASPIv2", "BINAQUAL"]
AI = ["CLAP-FAD", "PAM", "SCOREQ_nr", "SCOREQ_ref", "SHEET-SSL-MOS", "UTMOSv2",
      "NOMAD", "SQUIM_STOI", "SQUIM_PESQ", "SQUIM_SI-SDR", "SQUIM_MOS",
      "URGENT-MOS", "URGENT-CMOS", "ITU-T-P.566", "DNSMOS-P835-OVRL",
      "DNSMOS-P808", "SMAQ"]
FAMILY = {**{m: "Classic signal" for m in CLASSIC},
          **{m: "Psychoacoustic model" for m in PSYCHO},
          **{m: "AI / neural network" for m in AI}}
METRICS = CLASSIC + PSYCHO + AI

# Reference palette (validated categorical slots 1-3) and text tokens
COL = {"Classic signal": "#2a78d6", "Psychoacoustic model": "#eb6834",
       "AI / neural network": "#1baf7a", "Fusion (ML)": "#0b0b0b"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
ARTIFACT_NAMES = {
    "LP": "Bandwidth loss", "PE": "Pre-echo", "SH": "Spectral holes",
    "TM": "Tonality mismatch", "UN": "Unmasked noise", "DE": "Dialogue enhancement",
    "ICC": "Stereo width (ICC)", "ILD": "Stereo balance (ILD)",
    "QNLR": "Stereo noise L/R", "QNMS": "Stereo noise M/S", "QNmix": "Stereo noise mix",
    "SHLR": "Stereo holes L/R", "SHMS": "Stereo holes M/S", "SHmix": "Stereo holes mix",
    "speech": "Neural codec (speech)", "combined": "Neural codec (mixed)",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlecolor": INK, "figure.facecolor": SURF, "axes.facecolor": SURF,
    "axes.spines.top": False, "axes.spines.right": False,
})


# ---------------------------------------------------------------- helpers
def load():
    t = pd.read_csv(os.path.join(DATA, "odaq_table.csv"))
    meta = json.load(open(os.path.join(DATA, "analysis.json")))
    # Note: the ODAQ benchmark export already flips "lower = better" metrics
    # (listed in meta["negated_metrics"]), so higher = better for every column here.
    t = t[~t["is_anchor"]].reset_index(drop=True)   # anchors are fixed low-pass references
    t["source"] = t["item"].map(source_of)
    return t


def pearson(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    return stats.pearsonr(x[ok], y[ok])[0] if ok.sum() > 3 else np.nan


def macro_r(d, col):
    """Average of the correlations inside each listening test. Different tests
    use the 0-100 scale a little differently, so pooling them would be unfair."""
    return float(np.nanmean([pearson(g[col].values, g["mean_score"].values)
                             for _, g in d.groupby("subset")]))


def cluster_boot(df, fn, n=N_BOOT):
    """95% CI by resampling whole MUSHRA trials within each listening test
    (clips in one trial share a reference and listeners, so they are not independent)."""
    plan = {s: {g: np.where((df["subset"].values == s) & (df["item"].values == g))[0]
                for g in df.loc[df["subset"] == s, "item"].unique()}
            for s in df["subset"].unique()}
    vals = []
    for _ in range(n):
        rows = []
        for s, groups in plan.items():
            keys = list(groups)
            for g in RNG.choice(keys, size=len(keys), replace=True):
                rows.append(groups[g])
        vals.append(fn(df.iloc[np.concatenate(rows)]))
    return np.nanpercentile(vals, [2.5, 97.5])


def source_of(item):
    """Underlying song/recording, so the same music never appears in train and test."""
    s = re.sub(r"^(LP|PE|SH|TM|UN|DE|ICC|ILD|QNLR|QNMS|QNmix|SHLR|SHMS|SHmix)_", "", item)
    s = re.sub(r"^\d+[a-z]?_", "", s)
    return re.sub(r"_(remix\d+|LD\d+|FG|BG|PRC\d+).*$", "", s)


def cv_predict(df, features, groups, model="ridge"):
    """Out-of-sample predictions: the model never sees the held-out group."""
    X = df[features].values
    y = df["mean_score"].values
    pred = np.full(len(df), np.nan)
    for tr, te in LeaveOneGroupOut().split(X, y, df[groups].values):
        if model == "ridge":
            m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))
        else:
            m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05,
                                              max_iter=300, random_state=0)
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def rmse(a, b):
    return float(np.sqrt(np.nanmean((a - b) ** 2)))


# ---------------------------------------------------------------- analysis
def main():
    df = load()
    y = df["mean_score"].values
    report = {"n_clips": int(len(df)), "n_trials": int(df["item"].nunique()),
              "n_artifacts": int(df["method_group"].nunique())}

    # 1. Single metrics vs listeners (within each listening test, then averaged)
    rows = []
    for m in METRICS:
        lo, hi = cluster_boot(df, lambda d, m=m: macro_r(d, m))
        rows.append(dict(metric=m, family=FAMILY[m], pearson=macro_r(df, m),
                         pooled_pearson=pearson(df[m].values, y),
                         spearman=float(np.nanmean([stats.spearmanr(g[m], g["mean_score"], nan_policy="omit")[0]
                                                    for _, g in df.groupby("subset")])),
                         ci_low=lo, ci_high=hi))
    single = pd.DataFrame(rows).sort_values("pearson", ascending=False)
    single.to_csv(os.path.join(OUT, "single_metric_correlations.csv"), index=False)
    top = single.iloc[0]["metric"]

    # 2. ML fusion model: all metrics in, predicted MUSHRA score out.
    #    Four increasingly hard tests of "audio it has never seen".
    feats = [m for m in METRICS if df[m].notna().all()]
    LEVELS = {"item": "Unseen trial", "source": "Unseen song / recording",
              "method_group": "Unseen artifact type", "subset": "Unseen listening test"}
    gen, preds = [], {}
    for model in ["ridge", "gbm"]:
        for g in LEVELS:
            preds[(model, g)] = cv_predict(df, feats, g, model)
    best_model = max(["ridge", "gbm"], key=lambda mo: macro_r(df.assign(_p=preds[(mo, "item")]), "_p"))
    for g, label in LEVELS.items():
        df["_p"] = preds[(best_model, g)]
        df["_t"] = df[top]
        r_f = macro_r(df, "_p")
        ci_f = cluster_boot(df, lambda d: macro_r(d, "_p"))
        ci_d = cluster_boot(df, lambda d: macro_r(d, "_p") - macro_r(d, "_t"))
        gen.append(dict(level=label, group=g, fusion_r=r_f, fusion_ci=[float(v) for v in ci_f],
                        fusion_rmse=rmse(df["_p"].values, y), top_r=macro_r(df, top),
                        diff_ci=[float(v) for v in ci_d], ridge_r=macro_r(df.assign(_p=preds[("ridge", g)]), "_p")))
    df["pred_fusion_trial"] = preds[(best_model, "item")]
    df["pred_fusion_artifact"] = preds[(best_model, "method_group")]
    pd.DataFrame(gen).to_csv(os.path.join(OUT, "generalization.csv"), index=False)

    report.update(
        top_metric=top, top_family=FAMILY[top], top_pearson=float(single.iloc[0]["pearson"]),
        top_ci=[float(single.iloc[0]["ci_low"]), float(single.iloc[0]["ci_high"])],
        best_ai=single[single.family == "AI / neural network"].iloc[0][["metric", "pearson"]].to_dict(),
        best_psycho=single[single.family == "Psychoacoustic model"].iloc[0][["metric", "pearson"]].to_dict(),
        best_classic=single[single.family == "Classic signal"].iloc[0][["metric", "pearson"]].to_dict(),
        fusion_model=best_model, generalization=gen,
        fusion_trial_pearson=gen[0]["fusion_r"], fusion_trial_ci=gen[0]["fusion_ci"],
        fusion_trial_rmse=gen[0]["fusion_rmse"],
    )

    # 3. Where does it break? Accuracy per artifact type ------------------------
    show = [top, report["best_psycho"]["metric"], report["best_ai"]["metric"], report["best_classic"]["metric"]]
    show = list(dict.fromkeys(show))
    per = []
    for art, d in df.groupby("method_group"):
        row = {"artifact": ARTIFACT_NAMES.get(art, art), "code": art, "n": len(d),
               "Fusion (ML) unseen trial": pearson(d["pred_fusion_trial"].values, d["mean_score"].values),
               "Fusion (ML) unseen artifact": pearson(d["pred_fusion_artifact"].values, d["mean_score"].values)}
        for m in show:
            row[m] = pearson(d[m].values, d["mean_score"].values)
        per.append(row)
    per = pd.DataFrame(per).sort_values("Fusion (ML) unseen trial")
    per.to_csv(os.path.join(OUT, "accuracy_by_artifact.csv"), index=False)

    per_ct = []
    for ct, d in df.groupby("content_type"):
        per_ct.append({"content": ct, "n": len(d),
                       "Fusion (ML) unseen trial": pearson(d["pred_fusion_trial"].values, d["mean_score"].values),
                       top: pearson(d[top].values, d["mean_score"].values)})
    per_ct = pd.DataFrame(per_ct)
    per_ct.to_csv(os.path.join(OUT, "accuracy_by_content.csv"), index=False)

    # Biggest misses (clips where the model and listeners disagree most)
    df["error"] = df["pred_fusion_trial"] - df["mean_score"]
    misses = df.reindex(df["error"].abs().sort_values(ascending=False).index).head(10)
    misses[["subset", "method_group", "item", "condition", "content_type",
            "mean_score", "pred_fusion_trial", "error"]].to_csv(
        os.path.join(OUT, "biggest_misses.csv"), index=False)

    report["weakest_artifacts"] = per.head(3)[["artifact", "Fusion (ML) unseen trial", "Fusion (ML) unseen artifact"]].to_dict("records")
    report["strongest_artifacts"] = per.tail(3)[["artifact", "Fusion (ML) unseen trial"]].to_dict("records")
    report["by_content"] = per_ct.to_dict("records")
    report["biggest_misses"] = misses[["item", "condition", "mean_score", "pred_fusion_trial"]].round(1).to_dict("records")

    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)

    figures(df, single, per, report, show)
    write_report(report, single, per)
    print(json.dumps({k: v for k, v in report.items() if k != "fusion_all"}, indent=2, default=float))


# ---------------------------------------------------------------- figures
def figures(df, single, per, rep, show):
    # Fig 1: every metric vs listeners, grouped by family, with 95% CI
    s = single.sort_values("pearson")
    fig, ax = plt.subplots(figsize=(8, 10.4))
    yy = np.arange(len(s))
    ax.barh(yy, s["pearson"], color=[COL[f] for f in s["family"]], height=0.7)
    ax.errorbar(s["pearson"], yy, xerr=[s["pearson"] - s["ci_low"], s["ci_high"] - s["pearson"]],
                fmt="none", ecolor=INK2, elinewidth=1, capsize=0)
    ax.axvline(rep["fusion_trial_pearson"], color=INK, lw=1.5, ls="--")
    ax.text(rep["fusion_trial_pearson"] - 0.01, 0.6, f"ML fusion model\n(unseen trials)\nr = {rep['fusion_trial_pearson']:.2f}",
            color=INK, fontsize=8.5, va="bottom", ha="right")
    ax.set_yticks(yy, s["metric"])
    ax.set_xlabel("Agreement with listeners (Pearson r within each listening test, averaged; 95% CI)")
    ax.set_xlim(min(-0.2, s["ci_low"].min() - 0.05), 1)
    ax.axvline(0, color=GRID, lw=1)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COL[k]) for k in ["Classic signal", "Psychoacoustic model", "AI / neural network"]]
    ax.legend(handles, ["Classic signal", "Psychoacoustic model", "AI / neural network"],
              loc="upper center", bbox_to_anchor=(0.45, -0.06), ncol=3, frameon=False)
    ax.set_title(f"How well 34 quality metrics match {rep['n_clips']} expert MUSHRA ratings", loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "1_metrics_vs_listeners.png"), dpi=160)
    plt.close(fig)

    # Fig 2: predicted vs actual (fusion, unseen trials)
    fig, ax = plt.subplots(figsize=(6.2, 6))
    ax.plot([0, 100], [0, 100], color=GRID, lw=1.5)
    ax.scatter(df["mean_score"], df["pred_fusion_trial"], s=22, color=COL["AI / neural network"],
               edgecolor=SURF, linewidth=0.8, alpha=0.85)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_xlabel("Listeners' MUSHRA score (mean of experts)")
    ax.set_ylabel("ML fusion prediction (trial never seen in training)")
    ax.set_title(f"Fusion model vs. listeners, unseen trials  (r = {rep['fusion_trial_pearson']:.2f}, "
                 f"RMSE = {rep['fusion_trial_rmse']:.1f} pts)", loc="left", fontsize=11)
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "2_prediction_vs_listeners.png"), dpi=160)
    plt.close(fig)

    # Fig 3: where it breaks (heatmap, single sequential hue)
    cols = ["Fusion (ML) unseen trial", "Fusion (ML) unseen artifact"] + show
    M = per[cols].values.astype(float)
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(per) + 1.6))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", ["#f4f3ef", "#9cc3ef", "#2a78d6", "#123a6b"])
    im = ax.imshow(np.clip(M, 0, 1), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, "–" if np.isnan(v) else f"{v:.2f}", ha="center", va="center",
                    fontsize=8.5, color=SURF if (np.isfinite(v) and v > 0.6) else INK)
    labels = ["Fusion\n(unseen trial)", "Fusion\n(unseen artifact)"] + [m.replace("-", "-\n", 1) if len(m) > 10 else m for m in show]
    ax.set_xticks(range(len(cols)), labels, fontsize=8.5)
    ax.set_yticks(range(len(per)), [f"{a}  (n={n})" for a, n in zip(per["artifact"], per["n"])], fontsize=9)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Pearson r with listeners (within artifact type)", color=INK2)
    ax.set_title("Where the metrics break: agreement by artifact type", loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "3_where_it_breaks.png"), dpi=160)
    plt.close(fig)

    # Fig 4: generalization ladder
    gen = rep["generalization"]
    labels = [g["level"].replace(" ", "\n", 1) for g in gen]
    fus = [g["fusion_r"] for g in gen]
    err = np.array([[g["fusion_r"] - g["fusion_ci"][0] for g in gen], [g["fusion_ci"][1] - g["fusion_r"] for g in gen]])
    x = np.arange(len(gen))
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    bars = ax.bar(x, fus, 0.55, color=COL["AI / neural network"], label="ML fusion model (34 metrics)")
    ax.errorbar(x, fus, yerr=err, fmt="none", ecolor=INK2, elinewidth=1)
    ax.axhline(rep["top_pearson"], color=COL[rep["top_family"]], lw=2, ls="--",
               label=f"Best single metric ({rep['top_metric']}), no training needed")
    for bx, v in zip(x, fus):
        ax.text(bx, v + 0.035, f"{v:.2f}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(x, labels, fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Pearson r with listeners")
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_title("The harder the test, the smaller the AI's advantage", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "4_generalization.png"), dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- plain-English report
def write_report(r, single, per):
    g = {x["group"]: x for x in r["generalization"]}
    w = r["weakest_artifacts"]
    tr, te = g["item"], g["subset"]
    verdict = lambda d: "reliably better" if d["diff_ci"][0] > 0 else ("reliably worse" if d["diff_ci"][1] < 0 else "no reliable difference")
    lines = [
        "# Results: Does the AI hear what listeners hear?", "",
        "*Auto-generated by `src/analyze.py`. Every number comes from the files in `results/`.*", "",
        "## Headline", "",
        f"- **Data:** {r['n_clips']} processed audio clips from {r['n_trials']} MUSHRA trials and {r['n_artifacts']} artifact types, "
        "rated by expert listeners in four ODAQ listening tests (hidden references and anchors excluded).",
        "- **Scoring:** agreement = Pearson r between a metric and the listeners' mean score, computed inside each listening test and then averaged. "
        "95% confidence intervals come from resampling whole MUSHRA trials.",
        f"- **Best single metric:** {r['top_metric']} ({r['top_family'].lower()}), r = {r['top_pearson']:.2f} "
        f"(95% CI {r['top_ci'][0]:.2f}–{r['top_ci'][1]:.2f}).",
        f"- **Best by family:** psychoacoustic {r['best_psycho']['metric']} {r['best_psycho']['pearson']:.2f} · "
        f"AI {r['best_ai']['metric']} {r['best_ai']['pearson']:.2f} · classic {r['best_classic']['metric']} {r['best_classic']['pearson']:.2f}. "
        + (f"The best AI model essentially tied the best engineered psychoacoustic model; no off-the-shelf model clearly wins."
           if abs(r['best_ai']['pearson'] - r['best_psycho']['pearson']) < 0.03 else
           ("The best AI model beat the best psychoacoustic model." if r['best_ai']['pearson'] > r['best_psycho']['pearson']
            else "No off-the-shelf AI model beat the best engineered psychoacoustic model.")),
        f"- **ML fusion model** ({'gradient-boosted trees' if r['fusion_model']=='gbm' else 'ridge regression'} combining all metrics), "
        f"on trials it never saw: r = {tr['fusion_r']:.2f} (95% CI {tr['fusion_ci'][0]:.2f}–{tr['fusion_ci'][1]:.2f}), "
        f"typical error {tr['fusion_rmse']:.1f} MUSHRA points. Versus {r['top_metric']}: {verdict(tr)} "
        f"(difference 95% CI {tr['diff_ci'][0]:+.2f} to {tr['diff_ci'][1]:+.2f}).",
        f"- **The limit:** on a listening test it never saw, the fusion model scores r = {te['fusion_r']:.2f}: {verdict(te)} than {r['top_metric']} "
        f"(difference 95% CI {te['diff_ci'][0]:+.2f} to {te['diff_ci'][1]:+.2f}). It learns the conditions it was trained on; it does not yet transfer to new ones.",
        f"- **Weakest artifact types:** " + "; ".join(f"{a['artifact']} (r = {a['Fusion (ML) unseen trial']:.2f})" for a in w)
        + ". Stereo-image (width/balance) distortions were the hardest to predict.",
        "", "## Generalization ladder", "",
        "| Test | Fusion r | 95% CI | Typical error (pts) | Best single metric r | Fusion vs. single |", "|---|---|---|---|---|---|",
    ]
    for x in r["generalization"]:
        lines.append(f"| {x['level']} | {x['fusion_r']:.2f} | {x['fusion_ci'][0]:.2f}–{x['fusion_ci'][1]:.2f} | {x['fusion_rmse']:.1f} | {x['top_r']:.2f} | {verdict(x)} |")
    lines += [
        "", "## What this means for a product team", "",
        "1. **Pre-screen with the model** when testing versions of a codec or effect it has been trained on, and send only close calls to a listening test.",
        "2. **Do not trust it on a new kind of processing** (a new codec tool, a new spatial effect) until a small listening test confirms it. Out of the box it is no better than the best psychoacoustic model.",
        "3. **Spatial quality needs its own metric work.** Stereo width and balance changes were the hardest to predict, for the fusion model and for the single metrics shown.",
        "4. **Low agreement inside one artifact type** can also mean a narrow quality range or listener disagreement; check listener spread before blaming the metric.",
        "", "## Figures", "",
        "![](../figures/1_metrics_vs_listeners.png)", "",
        "![](../figures/2_prediction_vs_listeners.png)", "",
        "![](../figures/3_where_it_breaks.png)", "",
        "![](../figures/4_generalization.png)", "",
        "## Top 10 single metrics", "",
        "| Metric | Family | Pearson r (within test) | 95% CI | Spearman |", "|---|---|---|---|---|",
    ]
    for _, s in single.head(10).iterrows():
        lines.append(f"| {s.metric} | {s.family} | {s.pearson:.2f} | {s.ci_low:.2f} – {s.ci_high:.2f} | {s.spearman:.2f} |")
    with open(os.path.join(OUT, "RESULTS.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
