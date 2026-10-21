import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Wedge, Patch
from matplotlib.colors import to_rgb
import os

OUT = os.path.dirname(os.path.abspath(__file__))
UP = "../results"
t = pd.read_csv(f"{UP}/training_results.csv")
a = pd.read_csv(f"{UP}/adverse_results.csv")

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.linewidth": 0.6,
                     "savefig.dpi": 300, "figure.dpi": 150})
C_ATT, C_MLP = "#2a6f97", "#d1792b"
C_DENSE, C_PRUNE, C_RAND, C_ORIG = "#444444", "#8a8a8a", "#c0392b", "#1b7f79"
LV = [0.2, 0.4, 0.6, 0.8]
GROUPS = ["attention", "mlp"]
GNAME = {"attention": "Attention", "mlp": "MLP"}
dense = t[t.run_type == "dense"].iloc[0]
DACC = dense.test_accuracy

def tr(g, r, stage):
    if stage == "prune":
        q = t[(t.run_type == "pruning_evaluation") & (t.prune_type == g) & (t.requested_ratio == r)]
    else:
        m = "original_init" if stage == "orig" else "random_reinit"
        q = t[(t.run_type == "retraining") & (t.prune_type == g) & (t.requested_ratio == r) & (t.init_method == m)]
    return q.iloc[0]

def adv(g, r, stage, attack, eps):
    if stage == "dense":
        q = a[(a.run_type == "dense_adversarial") & (a.attack == attack) & (a.epsilon_x255 == eps)]
    elif stage == "prune":
        q = a[(a.run_type == "pruning_adversarial") & (a.prune_type == g) & (a.requested_ratio == r)
              & (a.attack == attack) & (a.epsilon_x255 == eps)]
    else:
        m = "original_init" if stage == "orig" else "random_reinit"
        q = a[(a.run_type == "retraining_adversarial") & (a.prune_type == g) & (a.requested_ratio == r)
              & (a.init_method == m) & (a.attack == attack) & (a.epsilon_x255 == eps)]
    assert len(q) == 1, (g, r, stage, attack, eps, len(q))
    return q.iloc[0]

def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight")
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    plt.close(fig)

def fig3():
    axes_def = [("Clean acc.", None), ("Macro F1", None), ("FGSM 4/255", ("fgsm", 4.0)),
                ("PGD 2/255", ("pgd", 2.0)), ("PGD 4/255", ("pgd", 4.0)), ("PGD 8/255", ("pgd", 8.0))]
    n = len(axes_def)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    def vec(g, r, st):
        if st == "dense":
            row = dense; vals = [row.test_accuracy, row.test_f1]
        else:
            row = tr(g, r, st); vals = [row.test_accuracy, row.test_f1]
        for _, spec in axes_def[2:]:
            vals.append(adv(g, r, st, spec[0], spec[1]).adversarial_accuracy)
        return np.array(vals)
    fig, axs = plt.subplots(1, 2, figsize=(9.0, 4.6), subplot_kw=dict(projection="polar"))
    fig.subplots_adjust(wspace=0.55)
    for ax, g in zip(axs, GROUPS):
        ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
        ax.set_ylim(0, 1); ax.set_yticks([0.25, 0.5, 0.75, 1.0]); ax.set_yticklabels(["25", "50", "75", "100"], fontsize=6, color="#777")
        ax.set_xticks(ang); ax.set_xticklabels([d[0] for d in axes_def], fontsize=7.5)
        ax.tick_params(axis="x", pad=6)
        ax.grid(color="#cccccc", lw=0.5); ax.spines["polar"].set_color("#bbbbbb")
        series = [("dense baseline", vec(g, 0.8, "dense"), C_DENSE, "--", 1.2, 0.0),
                  ("pruned, no retraining", vec(g, 0.8, "prune"), C_PRUNE, ":", 1.4, 0.0),
                  ("random reinit", vec(g, 0.8, "random"), C_RAND, "-", 1.8, 0.10),
                  ("original init", vec(g, 0.8, "orig"), C_ORIG, "-", 1.8, 0.18)]
        for lab, v, c, ls, lw, fa in series:
            vv = np.r_[v, v[:1]]; aa = np.r_[ang, ang[:1]]
            ax.plot(aa, vv, color=c, ls=ls, lw=lw, label=lab)
            if fa: ax.fill(aa, vv, color=c, alpha=fa)
        ax.set_title(f"{GNAME[g]} pruned to 80%\n({tr(g,0.8,'orig').effective_parameter_sparsity*100:.1f}% of the full model)",
                     fontsize=9, color=(C_ATT if g == "attention" else C_MLP), fontweight="bold", pad=16)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.03))
    fig.text(0.5, 0.0, "", fontsize=7)
    save(fig, "fig3_robustness_fingerprint_at_80")

def fig4():
    xs = np.array([0, 0.2, 0.4, 0.6, 0.8]); ys = np.array([0, 2, 4, 8.0])
    conds = [("attention", "orig"), ("attention", "random"), ("mlp", "orig"), ("mlp", "random")]
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.6), sharex=True, sharey=True)
    levels = np.linspace(0, 1, 21)
    cs = None
    for ax, (g, st) in zip(axs.ravel(), conds):
        Z = np.zeros((len(ys), len(xs)))
        for j, r in enumerate(xs):
            if r == 0:
                Z[0, j] = DACC
                for i, e in enumerate(ys[1:], 1): Z[i, j] = adv(None, 0, "dense", "pgd", e).adversarial_accuracy
            else:
                Z[0, j] = tr(g, r, st).test_accuracy
                for i, e in enumerate(ys[1:], 1): Z[i, j] = adv(g, r, st, "pgd", e).adversarial_accuracy
        X, Y = np.meshgrid(xs * 100, ys)
        cs = ax.contourf(X, Y, Z, levels=levels, cmap="magma", vmin=0, vmax=1)
        ax.contour(X, Y, Z, levels=[0.5], colors="white", linewidths=1.8)
        ax.contour(X, Y, Z, levels=[0.25, 0.75], colors="white", linewidths=0.6, linestyles="--")
        ax.plot(X.ravel(), Y.ravel(), ".", ms=2.2, color="white", alpha=0.55)
        col = C_ATT if g == "attention" else C_MLP
        ax.set_title(f"{GNAME[g]}, {'original init' if st=='orig' else 'random reinit'}", fontsize=9, color=col, fontweight="bold")
        ax.set_xticks([0, 20, 40, 60, 80]); ax.set_yticks([0, 2, 4, 8]); ax.set_yticklabels(["0 (clean)", "2", "4", "8"])
    for ax in axs[1]: ax.set_xlabel("group sparsity (%)  [0 = dense model]")
    for ax in axs[:, 0]: ax.set_ylabel("PGD-10 budget (x/255)")
    cb = fig.colorbar(cs, ax=axs, shrink=0.8, pad=0.04); cb.set_label("accuracy under PGD-10", fontsize=8)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    fig.text(0.5, -0.01, "Solid white line: 50% accuracy iso-contour. Dashed: 25% and 75%. Dots: measured points; surface is linearly interpolated.",
             ha="center", fontsize=7, style="italic")
    save(fig, "fig4_robustness_terrain")
if __name__ == "__main__":
    fig3(); fig4()