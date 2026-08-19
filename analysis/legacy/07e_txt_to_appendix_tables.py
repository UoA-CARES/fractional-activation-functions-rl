from __future__ import annotations

import os
import re
import argparse
from typing import Dict, List, Tuple

import numpy as np


ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5]
ACTS = ["frelu", "flrelu", "fprelu", "fswish", "fgelu"]
ACT_LABEL = {"frelu": "FReLU", "flrelu": "FLReLU", "fprelu": "FPReLU", "fswish": "FSwish", "fgelu": "FGELU"}


def tex_escape(s: str) -> str:
    return str(s).replace("_", r"\_")


def fmt(x: float, decimals: int = 2, scale: float = 1e9) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NA"
    return f"{x/scale:.{decimals}f}"


def parse_dump(txt_path: str) -> Tuple[Dict[str, Dict[str, float]], Dict[Tuple[str, str], Dict[str, List[float]]]]:
    """
    Returns:
      relu[algo][task] = relu_value   (from PLACEMENT = r)
      tables[(algo, placement)][task] = [15 nums]  (FReLU 5, FLReLU 5, FPReLU 5)
    """
    with open(txt_path, "r") as f:
        lines = [ln.strip() for ln in f.readlines()]

    relu: Dict[str, Dict[str, float]] = {}
    tables: Dict[Tuple[str, str], Dict[str, List[float]]] = {}

    algo = None
    placement = None

    for ln in lines:
        if not ln or ln.startswith("#"):
            continue

        m_algo = re.match(r"^ALGO\s*=\s*(SAC|TD3)\s*$", ln)
        if m_algo:
            algo = m_algo.group(1)
            placement = None
            continue

        m_pl = re.match(r"^PLACEMENT\s*=\s*(.+)\s*$", ln)
        if m_pl:
            placement = m_pl.group(1).strip()
            if algo is None:
                raise ValueError("Found PLACEMENT before ALGO.")
            if placement != "r":
                tables[(algo, placement)] = {}
            continue

        if "|" in ln and algo is not None and placement is not None:
            parts = [p.strip() for p in ln.split("|")]
            if len(parts) < 2:
                continue

            task = parts[0]
            col2 = parts[1].strip()

            # Skip header rows like: "Task | ReLU | ..."
            if task.lower() == "task":
                continue
            if col2.lower() == "relu":
                continue

            # PLACEMENT = r gives ReLU only: "Task | <number or NA> | ..."
            if placement == "r":
                val_str = col2
                if val_str == "NA":
                    continue
                # guard against any stray non-numeric text
                try:
                    relu.setdefault(algo, {})[task] = float(val_str)
                except ValueError:
                    continue
                continue

            # Normal placement line:
            # Task | ReLU(usually NA) | 15 fractional numbers
            # We ignore the ReLU column here and take len(ACTS) * 5 numbers from parts[2:]
            if len(parts) < 2 + len(ACTS) * 5:
                continue

            nums_str = parts[2:2 + len(ACTS) * 5]  # exactly len(ACTS) * 5
            nums: List[float] = []
            ok = True
            for s in nums_str:
                s = s.strip()
                if s == "NA":
                    nums.append(float("nan"))
                else:
                    try:
                        nums.append(float(s))
                    except ValueError:
                        ok = False
                        break
            if ok:
                tables[(algo, placement)][task] = nums

    return relu, tables


def best_alpha_idx(block_vals: List[float]) -> int:
    """Return index 0..4 of max in block (ignoring NaN)."""
    arr = np.array(block_vals, dtype=float)
    if np.all(np.isnan(arr)):
        return -1
    return int(np.nanargmax(arr))


def overall_best_key(relu_val: float, nums15: List[float]) -> Tuple[str, int]:
    """
    Return:
      ("relu", -1) if relu is best
      (act, alpha_index 0..4) for best fractional cell
    """
    best_v = relu_val
    best = ("relu", -1)

    k = 0
    for act in ACTS:
        for j in range(5):
            v = nums15[k]
            if not np.isnan(v) and v > best_v:
                best_v = v
                best = (act, j)
            k += 1
    return best


def make_table_tex(
    algo: str,
    placement: str,
    tasks: List[str],
    relu_map: Dict[str, float],
    table: Dict[str, List[float]],
    decimals: int = 1,
) -> str:
    algo_tex = tex_escape(algo)
    pl_tex = tex_escape(placement)

    label = f"tab:appendix_2layers_{algo}_{placement}".replace("-", "_")
    caption = (
        rf"Normalised area under the learning curve (AUC) ($\times 10^9$) "
        rf"(interquartile mean over five seeds) for \textbf{{{algo_tex}}} "
        rf"with \textbf{{2layers}} under placement \textbf{{{pl_tex}}}. "
        r"Best $\alpha$ per activation type in \textbf{bold}; "
        r"overall best per task in \textbf{\textcolor{custompurple}{purple}}."
    )

    out: List[str] = []
    out.append(r"\begin{table*}[t]")
    out.append(r"\centering")
    out.append(r"\small")
    out.append(r"\setlength{\tabcolsep}{4pt}")
    out.append(r"\renewcommand{\arraystretch}{1.3}")
    out.append(rf"\caption{{{caption}}}")
    out.append(rf"\label{{{label}}}")
    out.append("")
    out.append(r"\resizebox{\textwidth}{!}{%")
    out.append(r"\begin{tabular}{l || >{\columncolor{gray!20}}c || *{5}{c} || *{5}{c} || *{5}{c} || *{5}{c} || *{5}{c}}")
    out.append(r"\toprule")
    out.append(r"\multirow{2}{*}{\textbf{Task}} & \multirow{2}{*}{\textbf{ReLU}} &")
    out.append(r"\multicolumn{5}{c||}{\textbf{FReLU}} &")
    out.append(r"\multicolumn{5}{c||}{\textbf{FLReLU}} &")
    out.append(r"\multicolumn{5}{c||}{\textbf{FPReLU}} &")
    out.append(r"\multicolumn{5}{c||}{\textbf{FSwish}} &")
    out.append(r"\multicolumn{5}{c}{\textbf{FGELU}} \\")
    out.append(r"\cmidrule(lr){3-7}\cmidrule(lr){8-12}\cmidrule(lr){13-17}\cmidrule(lr){18-22}\cmidrule(lr){23-27}")
    out.append(
        r"& & $\alpha{=}$0.1 & 0.2 & 0.3 & 0.4 & 0.5 &"
        r" 0.1 & 0.2 & 0.3 & 0.4 & 0.5 &"
        r" 0.1 & 0.2 & 0.3 & 0.4 & 0.5 &"
        r" 0.1 & 0.2 & 0.3 & 0.4 & 0.5 &"
        r" 0.1 & 0.2 & 0.3 & 0.4 & 0.5 \\"
    )
    out.append(r"\toprule")

    for t in tasks:
        nums = table.get(t, [float("nan")] * (len(ACTS) * 5))
        relu_val = relu_map.get(t, float("nan"))

        # overall best
        best = overall_best_key(relu_val, nums)

        # ReLU cell
        relu_cell = fmt(relu_val, decimals)
        if best == ("relu", -1) and relu_cell != "NA":
            relu_cell = rf"\textcolor{{custompurple}}{{\textbf{{{relu_cell}}}}}"
        else:
            if relu_cell != "NA":
                relu_cell = rf"\textbf{{{relu_cell}}}"

        # fractional cells
        cells: List[str] = []
        k = 0
        for act in ACTS:
            start = k
            best_idx = best_alpha_idx(nums[start:start + 5])
            for j in range(5):
                v = nums[k]
                s = fmt(v, decimals)
                if s != "NA" and j == best_idx:
                    s = rf"\textbf{{{s}}}"
                if best == (act, j) and s != "NA":
                    # purple overrides
                    raw = fmt(v, decimals)
                    s = rf"\textcolor{{custompurple}}{{\textbf{{{raw}}}}}"
                cells.append(s)
                k += 1

        out.append(rf"\textbf{{{tex_escape(t)}}} & {relu_cell} & " + " & ".join(cells) + r" \\")
        out.append(r"\addlinespace[2pt]")

    out.append(r"\bottomrule")
    out.append(r"\end{tabular}}")
    out.append(r"}")
    out.append(r"\end{table*}")
    out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_txt", required=True, help="Path to appendix_2layers_numbers.txt")
    ap.add_argument("--out_dir", required=True, help="Output directory for generated .tex tables")
    ap.add_argument("--decimals", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    relu, tables = parse_dump(args.in_txt)

    # Write one table per (algo, placement)
    generated = []
    for (algo, placement), table in sorted(tables.items(), key=lambda x: (x[0][0], x[0][1])):
        if algo not in relu:
            raise ValueError(f"Missing ReLU baseline for algo={algo}. Ensure PLACEMENT=r exists in the txt.")
        relu_map = relu[algo]
        tasks = sorted(table.keys())

        tex = make_table_tex(algo, placement, tasks, relu_map, table, decimals=args.decimals)

        fname = f"appendix_2layers_{algo}_{placement}.tex".replace("/", "_").replace(" ", "_")
        out_path = os.path.join(args.out_dir, fname)
        with open(out_path, "w") as f:
            f.write(tex)
        generated.append(fname)
        print(f"[OK] wrote {out_path}")

    # Master include file
    master = os.path.join(args.out_dir, "appendix_2layers_all_placements.tex")
    with open(master, "w") as f:
        f.write("% Auto-generated appendix tables: 2layers, all placements\n\n")
        for algo in ["SAC", "TD3"]:
            f.write(rf"\subsection*{{2layers: {algo}}}" + "\n\n")
            for fname in generated:
                if fname.startswith(f"appendix_2layers_{algo}_"):
                    f.write(rf"\input{{{fname}}}" + "\n\n")

    print(f"[OK] wrote {master}")


if __name__ == "__main__":
    main()
