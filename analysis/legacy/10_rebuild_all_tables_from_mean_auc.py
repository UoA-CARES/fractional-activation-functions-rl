import os
import pandas as pd
import numpy as np

ROOT = os.path.expanduser("~/Desktop/new-Fr")
IN = os.path.join(ROOT, "outputs", "paper",
                  "table_stats_paired_seed_rows_clean.csv")

OUTDIR = os.path.join(ROOT, "outputs", "paper",
                      "tables_best2layers")
os.makedirs(OUTDIR, exist_ok=True)

df = pd.read_csv(IN)

def compute_winners(algo, layer_tag):

    subset = df[
        (df.algo==algo) &
        (df.layers.str.contains(layer_tag))
    ].copy()

    # mean AUC per activation per alpha
    g = (
        subset.groupby(["task","activation","alpha_key"])["auc_frac"]
        .mean()
        .reset_index()
    )

    # best alpha per activation
    best_alpha = g.loc[
        g.groupby(["task","activation"])["auc_frac"].idxmax()
    ]

    # mean relu per task
    relu_mean = (
        subset[subset.activation=="relu"]
        .groupby("task")["auc_frac"]
        .mean()
    )

    rows = []

    for task in sorted(subset.task.unique()):

        task_rows = best_alpha[best_alpha.task==task].copy()
        task_rows["delta"] = 100 * (
            task_rows["auc_frac"] - relu_mean[task]
        ) / relu_mean[task]

        winner = task_rows.loc[task_rows["delta"].idxmax()]

        rows.append({
            "Task": task,
            "Best Activation": winner["activation"],
            "Delta_%": round(winner["delta"],2)
        })

    return pd.DataFrame(rows)


def export_tex(df_out, filename, caption):

    path = os.path.join(OUTDIR, filename)

    with open(path,"w") as f:

        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write("\\begin{tabular}{lll}\n")
        f.write("\\toprule\n")
        f.write("Task & Best Activation & $\\Delta\\%$ \\\\\n")
        f.write("\\midrule\n")

        for _,r in df_out.iterrows():
            f.write(
                f"{r['Task']} & {r['Best Activation']} & "
                f"+{r['Delta_%']} \\\\\n"
            )

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print("Wrote:", path)


# ---------- BUILD ALL ----------

for algo in ["SAC","TD3"]:
    for layer in ["1","2"]:

        winners = compute_winners(algo, layer)

        export_tex(
            winners,
            f"table_winner_{algo.lower()}_{layer}.tex",
            f"Best activation per task ({algo}, {layer}-layer)."
        )
