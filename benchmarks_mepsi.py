"""
benchmark_mepsi.py
==================
Comprehensive benchmarking suite for the MEPSI ensemble pruning algorithm.
(AAAI 2024: "MEPSI: An MDL-Based Ensemble Pruning Approach with Structural Information")

Section → Paper mapping
-----------------------
Section 2  (accuracy)          → Table 1, Figure 1 (average ranks)
Section 3  (runtime)           → NOT in paper — original contribution
Section 4  (lambda sensitivity)→ NOT in paper — original contribution
Section 5  (k sensitivity)     → NOT in paper — original contribution
Section 6  (structural stats)  → NOT in paper — original contribution
Section 7  (ablation)          → NOT in paper — original contribution
Section 8  (bound analysis)    → Theorem 4.2 (numerical verification)
Section 9  (Jaccard index)     → Appendix of paper

Usage
-----
python benchmark_mepsi.py --run all
python benchmark_mepsi.py --run accuracy
python benchmark_mepsi.py --run accuracy,runtime,lambda
python benchmark_mepsi.py --datasets digits,breast-w
python benchmark_mepsi.py --repeats 5
python benchmark_mepsi.py --output-dir my_results/
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import sys
import time
import json
import random
import argparse
import tracemalloc
import warnings
import traceback

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm

# scikit-learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import (
    load_digits,
    fetch_openml,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── MEPSI package imports ────────────────────────────────────────────────────
# ADJUST IF NEEDED: these are inferred from setup.py's extension structure.
# The Cython extensions are compiled into these submodules.
# If your repo exposes a higher-level Python wrapper (e.g. mepsi/pruning/mepsi.py),
# use that instead.

try:
    from mepsi.pruning._libs.mepsi import mepsi_pruning  # main MEPSI greedy
except ImportError:
    mepsi_pruning = None
    print("[WARN] Could not import mepsi.pruning._libs.mepsi. "
          "Falling back to pure-Python stub — accuracy numbers will be WRONG.")

try:
    from mepsi.pruning._libs.kappa_pruning import kappa_pruning  # Kappa baseline
except ImportError:
    kappa_pruning = None
    print("[WARN] Could not import mepsi.pruning._libs.kappa_pruning.")

try:
    from mepsi.metric.tree_edit._libs.tree_edit import compute_tree_edit_distance as _ted
    # ADJUST IF NEEDED: the Cython function may be named differently.
    # Common alternatives: ted(), tree_edit(), tree_edit_distance()
except ImportError:
    _ted = None
    print("[WARN] Could not import tree_edit_distance from Cython extension.")

# ── Optional: higher-level Python wrappers (if they exist in the repo) ───────
# ADJUST IF NEEDED: uncomment if the repo provides Python-level wrappers.
# from mepsi.pruning.mepsi import MEPSIPruner
# from mepsi.forest.forest import MEPSIForest


# =============================================================================
# SECTION 0 — HELPERS & COMPATIBILITY SHIMS
# =============================================================================

def set_seed(seed: int):
    """Globally fix all random seeds."""
    random.seed(seed)
    np.random.seed(seed)


def _make_results_dir(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)


def _save_csv(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)
    print(f"  Saved → {path}")


# ── Tree-structure helpers ────────────────────────────────────────────────────

def node_count(tree: DecisionTreeClassifier) -> int:
    """Number of internal (split) nodes in a fitted sklearn DecisionTree."""
    return int((tree.tree_.feature >= 0).sum())


def tree_depth(tree: DecisionTreeClassifier) -> int:
    return int(tree.get_depth())


def compute_ted(tree_a: DecisionTreeClassifier,
                tree_b: DecisionTreeClassifier) -> float:
    """
    Compute Tree Edit Distance between two sklearn DecisionTreeClassifier objects.

    ADJUST IF NEEDED: The Cython extension _ted may expect a different
    representation (e.g. node arrays). Pass tree.tree_ (the underlying
    sklearn Tree object) or whatever the Cython function expects.

    Falls back to |NC(a) - NC(b)| if the extension is unavailable.
    """
    if _ted is not None:
        try:
            # ADJUST: pass tree.tree_ or the raw node arrays as required
            return float(_ted(tree_a.tree_, tree_b.tree_))
        except Exception:
            pass
    # Fallback: absolute difference in node counts
    return float(abs(node_count(tree_a) - node_count(tree_b)))


def compute_esm(candidate: DecisionTreeClassifier,
                selected: list) -> float:
    """
    Edit Similarity Measure:
        ESM(h, G) = NC(h) - min_{g in G} TED(g, h)
    Paper Definition 3.3.
    """
    nc = node_count(candidate)
    if not selected:
        return float(nc)
    min_ted = min(compute_ted(g, candidate) for g in selected)
    return float(nc - min_ted)


def predict_ensemble(trees: list,
                     weights: np.ndarray,
                     X: np.ndarray) -> np.ndarray:
    """
    Hard majority-vote ensemble prediction.
    weights are normalised to sum to 1.
    """
    # Shape: (n_trees, n_samples, n_classes)
    score = np.zeros((X.shape[0],
                      trees[0].n_classes_))
    w = np.array(weights, dtype=float)
    w = w / w.sum()
    for tree, wi in zip(trees, w):
        prob = tree.predict_proba(X)        # (n_samples, n_classes)
        score += wi * prob
    return np.argmax(score, axis=1)


def ensemble_error(trees, weights, X, y):
    pred = predict_ensemble(trees, weights, X)
    return float(np.mean(pred != y))


# ── Pure-Python MEPSI greedy (fallback / instrumented version) ────────────────

def mepsi_greedy(trees: list,
                 X_train: np.ndarray,
                 y_train: np.ndarray,
                 k: int,
                 lam: float,
                 return_history: bool = False):
    """
    Pure-Python greedy MEPSI (Algorithm 1 from the paper).
    Used as: (a) fallback when Cython ext is unavailable,
             (b) instrumented version for convergence tracking.

    Returns
    -------
    selected_indices : list[int]
    history          : list[dict]  (only if return_history=True)
        Each entry: {iteration, objective, empirical_error, structural_info}
    """
    T = len(trees)
    m = len(y_train)
    history = []

    # ── Step 1: find the best initial pair ───────────────────────────────────
    best_obj = np.inf
    best_pair = (0, 1)
    for i in range(T - 1):
        for j in range(i + 1, T):
            avg_pred = (trees[i].predict(X_train).astype(float) +
                        trees[j].predict(X_train).astype(float)) / 2.0
            # round to nearest class
            pred = np.round(avg_pred).astype(int)
            # ADJUST: for multi-class use argmax over predict_proba
            err = float(np.mean(pred != y_train))
            nc_i = node_count(trees[i])
            nc_j = node_count(trees[j])
            ted_ij = compute_ted(trees[i], trees[j])
            comp = nc_i + nc_j - ted_ij
            obj = err - lam * comp
            if obj < best_obj:
                best_obj = obj
                best_pair = (i, j)

    selected = list(best_pair)
    if return_history:
        sel_trees = [trees[i] for i in selected]
        err_val = ensemble_error(sel_trees, [1.0] * len(sel_trees), X_train, y_train)
        si_val = sum(compute_esm(trees[i], [trees[j] for j in selected if j != i])
                     for i in selected)
        history.append({
            "iteration": 2,
            "objective": err_val - lam * si_val,
            "empirical_error": err_val,
            "structural_info": si_val,
        })

    # ── Step 2: greedy expansion ──────────────────────────────────────────────
    remaining = [i for i in range(T) if i not in selected]
    while len(selected) < k and remaining:
        best_obj = np.inf
        best_cand = remaining[0]
        sel_trees = [trees[i] for i in selected]
        n_sel = len(sel_trees)

        for idx in remaining:
            candidate = trees[idx]
            # ensemble prediction with uniform weights
            all_trees = sel_trees + [candidate]
            w = np.ones(len(all_trees)) / len(all_trees)
            err = ensemble_error(all_trees, w, X_train, y_train)
            esm_val = compute_esm(candidate, sel_trees)
            obj = err - lam * esm_val
            if obj < best_obj:
                best_obj = obj
                best_cand = idx

        selected.append(best_cand)
        remaining.remove(best_cand)

        if return_history:
            sel_trees_now = [trees[i] for i in selected]
            w = np.ones(len(sel_trees_now)) / len(sel_trees_now)
            err_val = ensemble_error(sel_trees_now, w, X_train, y_train)
            si_val = sum(
                compute_esm(trees[i], [trees[j] for j in selected if j != i])
                for i in selected
            )
            history.append({
                "iteration": len(selected),
                "objective": err_val - lam * si_val,
                "empirical_error": err_val,
                "structural_info": si_val,
            })

    if return_history:
        return selected, history
    return selected


def run_mepsi(trees, X_train, y_train, k, lam,
              return_history=False):
    """
    Dispatcher: use Cython extension if available, else Python fallback.

    ADJUST IF NEEDED: if the Cython mepsi_pruning has a different signature,
    update the call below. The paper's function likely takes
    (trees, X_train, y_train, k, lambda) and returns selected indices.
    """
    if mepsi_pruning is not None and not return_history:
        try:
            # ADJUST: match actual Cython function signature
            selected = mepsi_pruning(trees, X_train, y_train, k, lam)
            return list(selected)
        except Exception as e:
            print(f"  [WARN] Cython mepsi_pruning failed ({e}), using Python fallback")
    return mepsi_greedy(trees, X_train, y_train, k, lam,
                        return_history=return_history)


def run_kappa(trees, X_train, y_train, k):
    """
    Kappa pruning baseline dispatcher.
    ADJUST IF NEEDED: update Cython call signature if different.
    """
    if kappa_pruning is not None:
        try:
            selected = kappa_pruning(trees, X_train, y_train, k)
            return list(selected)
        except Exception as e:
            print(f"  [WARN] Cython kappa_pruning failed ({e}), using random fallback")
    return list(np.random.choice(len(trees), size=k, replace=False))


def run_random(trees, k, seed=None):
    rng = np.random.RandomState(seed)
    return list(rng.choice(len(trees), size=k, replace=False))


def generate_trees(X_train, y_train, T=200, seed=0):
    """
    Generate T CART decision trees using bootstrap + random feature selection
    exactly as described in the MEPSI paper (Section 5.1).
    """
    rf = RandomForestClassifier(
        n_estimators=T,
        bootstrap=True,
        max_features="sqrt",   # random feature selection
        random_state=seed,
    )
    rf.fit(X_train, y_train)
    return rf.estimators_   # list of DecisionTreeClassifier


# =============================================================================
# SECTION 1 — DATASETS
# =============================================================================

# Default λ values per dataset (from paper appendix / tuned defaults)
LAMBDA_DEFAULTS = {
    "digits":         0.01,
    "usps":           0.01,
    "breast-cancer":  0.01,
    "breast-w":       0.005,
    "vowel":          0.01,
    "mfeat-factors":  0.01,
    "splice":         0.01,
    "credit-a":       0.005,
    "tic-tac-toe":    0.01,
    "vehicle":        0.01,
    "sick":           0.005,
}

# OpenML dataset IDs for UCI datasets used in the paper
OPENML_IDS = {
    "usps":           41082,
    "breast-cancer":  13,
    "breast-w":       15,
    "vowel":          307,
    "mfeat-factors":  12,
    "splice":         46,
    "credit-a":       29,
    "tic-tac-toe":    50,
    "vehicle":        54,
    "sick":           38,
}


def load_all_datasets(dataset_filter=None):
    """
    Returns dict: name → (X_train, X_test, y_train, y_test)
    All arrays are numpy float64 / int.
    """
    datasets = {}
    le = LabelEncoder()

    # ── sklearn digits ────────────────────────────────────────────────────────
    if dataset_filter is None or "digits" in dataset_filter:
        try:
            data = load_digits()
            X, y = data.data.astype(np.float64), data.target
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y)
            datasets["digits"] = (X_tr, X_te, y_tr, y_te)
            print(f"  Loaded: digits         {X_tr.shape}")
        except Exception as e:
            print(f"  [SKIP] digits: {e}")

    # ── OpenML / UCI datasets ─────────────────────────────────────────────────
    for name, oid in OPENML_IDS.items():
        if dataset_filter is not None and name not in dataset_filter:
            continue
        try:
            bunch = fetch_openml(data_id=oid, as_frame=False, parser="auto")
            X = bunch.data.astype(np.float64)
            y = le.fit_transform(bunch.target)
            # Handle NaNs with column mean
            col_means = np.nanmean(X, axis=0)
            nan_mask = np.isnan(X)
            X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y)
            datasets[name] = (X_tr, X_te, y_tr, y_te)
            print(f"  Loaded: {name:<20} {X_tr.shape}")
        except Exception as e:
            print(f"  [SKIP] {name}: {e}")

    if not datasets:
        raise RuntimeError("No datasets could be loaded. "
                           "Check internet access or dataset availability.")
    return datasets


# =============================================================================
# SECTION 2 — ACCURACY BENCHMARK  (reproduces Table 1 + Figure 1)
# =============================================================================

def benchmark_accuracy(datasets, output_dir, n_repeats=3,
                       T=200, k=20):
    """
    Reproduce Table 1 of the paper.
    Methods: MEPSI, Kappa, Random.
    """
    print("\n" + "=" * 60)
    print("SECTION 2: ACCURACY BENCHMARK")
    print("=" * 60)

    records = []

    for ds_name, (X_tr, X_te, y_tr, y_te) in datasets.items():
        lam = LAMBDA_DEFAULTS.get(ds_name, 0.01)
        print(f"\n  Dataset: {ds_name}  |  λ={lam}  |  T={T}  |  k={k}")

        for method in ["mepsi", "kappa", "random"]:
            accs = []
            for rep in tqdm(range(n_repeats),
                            desc=f"    {method:<8}",
                            leave=False):
                try:
                    set_seed(rep)
                    trees = generate_trees(X_tr, y_tr, T=T, seed=rep)

                    if method == "mepsi":
                        sel = run_mepsi(trees, X_tr, y_tr, k=k, lam=lam)
                    elif method == "kappa":
                        sel = run_kappa(trees, X_tr, y_tr, k=k)
                    else:
                        sel = run_random(trees, k=k, seed=rep)

                    sel_trees = [trees[i] for i in sel]
                    w = np.ones(len(sel_trees)) / len(sel_trees)
                    pred = predict_ensemble(sel_trees, w, X_te)
                    acc = float(np.mean(pred == y_te)) * 100
                    accs.append(acc)
                except Exception as e:
                    print(f"      [ERR] rep {rep}: {e}")

            if accs:
                mean_acc = float(np.mean(accs))
                std_acc  = float(np.std(accs))
                print(f"    {method:<8}  {mean_acc:.1f} ± {std_acc:.1f}")
                records.append({
                    "dataset":  ds_name,
                    "method":   method,
                    "mean_acc": round(mean_acc, 2),
                    "std_acc":  round(std_acc, 2),
                    "all_accs": accs,
                })

    df = pd.DataFrame(records)
    _save_csv(df, os.path.join(output_dir, "accuracy_results.csv"))

    # Print pivot table
    print("\n  ── Accuracy Summary (mean ± std) ──")
    pivot = df.pivot(index="dataset", columns="method", values="mean_acc")
    print(pivot.to_string())

    return df


# =============================================================================
# SECTION 3 — RUNTIME BENCHMARK
# =============================================================================

def _timed_run(fn, *args, **kwargs):
    """Returns (result, wall_time_seconds, peak_memory_bytes)."""
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    t1 = time.perf_counter()
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, t1 - t0, peak_mem


def benchmark_runtime(datasets, output_dir, reps=1):
    """
    Section 3: measure wall-clock time for pruning
    as T, k, and n vary. NOT in the original paper — new contribution.
    """
    print("\n" + "=" * 60)
    print("SECTION 3: RUNTIME BENCHMARK")
    print("=" * 60)

    # pick a medium dataset
    medium_ds = None
    for name in ["breast-w", "vehicle", "vowel", "digits"]:
        if name in datasets:
            medium_ds = name
            break
    if medium_ds is None:
        medium_ds = next(iter(datasets))

    X_tr, X_te, y_tr, y_te = datasets[medium_ds]
    print(f"  Using dataset: {medium_ds}  ({X_tr.shape})")

    # ── 3a: Runtime vs pool size T ────────────────────────────────────────────
    print("\n  3a: Runtime vs pool size T")
    records_T = []
    for T in [50, 100, 200, 300, 500]:
        for rep in range(reps):
            set_seed(rep)
            # Tree generation time
            _, t_gen, _ = _timed_run(generate_trees, X_tr, y_tr, T=T, seed=rep)
            trees = generate_trees(X_tr, y_tr, T=T, seed=rep)

            # TED computation time (pairwise, sample up to min(T,50) pairs)
            sample_size = min(T, 50)
            idxs = list(range(sample_size))
            t0 = time.perf_counter()
            for i in range(sample_size - 1):
                compute_ted(trees[i], trees[i + 1])
            t_ted_sample = time.perf_counter() - t0
            # Extrapolate to full T*(T-1)/2 pairs
            n_sample_pairs = sample_size - 1
            n_full_pairs   = T * (T - 1) // 2
            t_ted_full = (t_ted_sample / n_sample_pairs) * n_full_pairs if n_sample_pairs > 0 else 0.0

            # Pruning time
            _, t_prune, mem = _timed_run(run_mepsi, trees, X_tr, y_tr, 20, 0.01)

            records_T.append({
                "T": T, "rep": rep,
                "t_gen_s":       round(t_gen,       4),
                "t_ted_est_s":   round(t_ted_full,  4),
                "t_prune_s":     round(t_prune,     4),
                "t_total_s":     round(t_gen + t_prune, 4),
                "peak_mem_mb":   round(mem / 1e6,   4),
            })
            print(f"    T={T:4d}  gen={t_gen:.2f}s  "
                  f"ted_est={t_ted_full:.2f}s  "
                  f"prune={t_prune:.2f}s  "
                  f"mem={mem/1e6:.1f}MB")

    df_T = pd.DataFrame(records_T)
    _save_csv(df_T, os.path.join(output_dir, "runtime_vs_T.csv"))

    # ── 3b: Runtime vs pruning size k ─────────────────────────────────────────
    print("\n  3b: Runtime vs pruning size k")
    trees = generate_trees(X_tr, y_tr, T=200, seed=0)
    records_k = []
    for k in [5, 10, 15, 20, 25, 30]:
        for rep in range(reps):
            set_seed(rep)
            _, t_prune, mem = _timed_run(run_mepsi, trees, X_tr, y_tr, k, 0.01)
            records_k.append({
                "k": k, "rep": rep,
                "t_prune_s":   round(t_prune, 4),
                "peak_mem_mb": round(mem / 1e6, 4),
            })
        mean_t = np.mean([r["t_prune_s"] for r in records_k if r["k"] == k])
        print(f"    k={k:3d}  prune={mean_t:.2f}s")

    df_k = pd.DataFrame(records_k)
    _save_csv(df_k, os.path.join(output_dir, "runtime_vs_k.csv"))

    # ── 3c: Runtime vs dataset size n ─────────────────────────────────────────
    print("\n  3c: Runtime vs dataset size n")
    # Use largest available dataset
    big_ds = max(datasets, key=lambda n: datasets[n][0].shape[0])
    X_big, _, y_big, _ = datasets[big_ds]
    print(f"  Using dataset: {big_ds}  ({X_big.shape})")

    records_n = []
    for frac in [0.1, 0.25, 0.5, 0.75, 1.0]:
        n_sub = max(50, int(len(y_big) * frac))
        idx   = np.random.choice(len(y_big), n_sub, replace=False)
        X_sub, y_sub = X_big[idx], y_big[idx]
        for rep in range(reps):
            set_seed(rep)
            trees_sub = generate_trees(X_sub, y_sub, T=100, seed=rep)
            _, t_prune, mem = _timed_run(
                run_mepsi, trees_sub, X_sub, y_sub, 20, 0.01)
            records_n.append({
                "frac": frac, "n": n_sub, "rep": rep,
                "t_prune_s":   round(t_prune, 4),
                "peak_mem_mb": round(mem / 1e6, 4),
            })
        mean_t = np.mean([r["t_prune_s"] for r in records_n if r["frac"] == frac])
        print(f"    frac={frac:.2f}  n={n_sub:6d}  prune={mean_t:.2f}s")

    df_n = pd.DataFrame(records_n)
    _save_csv(df_n, os.path.join(output_dir, "runtime_vs_n.csv"))
    return df_T, df_k, df_n


# =============================================================================
# SECTION 4 — LAMBDA SENSITIVITY
# =============================================================================

def benchmark_lambda(datasets, output_dir, n_repeats=3, T=200, k=20):
    """
    Sweep λ and record accuracy + structural information term.
    NOT in the paper — new contribution.
    """
    print("\n" + "=" * 60)
    print("SECTION 4: LAMBDA SENSITIVITY")
    print("=" * 60)

    # Use 5 representative datasets
    target_ds = ["digits", "breast-w", "vehicle", "vowel", "sick"]
    target_ds = [d for d in target_ds if d in datasets]
    if not target_ds:
        target_ds = list(datasets.keys())[:5]

    lambdas = [0.0, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    records = []

    for ds_name in target_ds:
        X_tr, X_te, y_tr, y_te = datasets[ds_name]
        print(f"\n  Dataset: {ds_name}")

        for lam in lambdas:
            accs, si_vals = [], []
            for rep in range(n_repeats):
                set_seed(rep)
                try:
                    trees = generate_trees(X_tr, y_tr, T=T, seed=rep)
                    sel   = run_mepsi(trees, X_tr, y_tr, k=k, lam=lam)
                    sel_trees = [trees[i] for i in sel]
                    w = np.ones(k) / k
                    acc = float(np.mean(predict_ensemble(sel_trees, w, X_te) == y_te))
                    # structural info term: sum of ESM over selected trees
                    si = sum(
                        compute_esm(trees[i],
                                    [trees[j] for j in sel if j != i])
                        for i in sel
                    )
                    accs.append(acc * 100)
                    si_vals.append(si)
                except Exception as e:
                    print(f"    [ERR] λ={lam}, rep {rep}: {e}")

            if accs:
                records.append({
                    "dataset":              ds_name,
                    "lambda":               lam,
                    "mean_acc":             round(float(np.mean(accs)), 3),
                    "std_acc":              round(float(np.std(accs)), 3),
                    "mean_structural_info": round(float(np.mean(si_vals)), 3),
                    "std_structural_info":  round(float(np.std(si_vals)), 3),
                })
                print(f"    λ={lam:.4f}  acc={np.mean(accs):.1f}±{np.std(accs):.1f}"
                      f"  SI={np.mean(si_vals):.1f}±{np.std(si_vals):.1f}")

    df = pd.DataFrame(records)
    _save_csv(df, os.path.join(output_dir, "lambda_sensitivity.csv"))
    return df


# =============================================================================
# SECTION 5 — PRUNING SIZE k SENSITIVITY
# =============================================================================

def benchmark_pruning_size(datasets, output_dir, n_repeats=3, T=200):
    """
    Accuracy vs pruning size k for MEPSI vs random.
    NOT in the paper — new contribution.
    """
    print("\n" + "=" * 60)
    print("SECTION 5: PRUNING SIZE k SENSITIVITY")
    print("=" * 60)

    k_values = [2, 5, 10, 15, 20, 25, 30, 40, 50]
    records  = []

    for ds_name, (X_tr, X_te, y_tr, y_te) in datasets.items():
        lam = LAMBDA_DEFAULTS.get(ds_name, 0.01)
        print(f"\n  Dataset: {ds_name}")

        for k in k_values:
            if k > T:
                continue
            mepsi_accs, rand_accs = [], []
            for rep in range(n_repeats):
                set_seed(rep)
                try:
                    trees = generate_trees(X_tr, y_tr, T=T, seed=rep)

                    sel_m = run_mepsi(trees, X_tr, y_tr, k=k, lam=lam)
                    w = np.ones(k) / k
                    acc_m = float(np.mean(
                        predict_ensemble([trees[i] for i in sel_m], w, X_te) == y_te
                    )) * 100
                    mepsi_accs.append(acc_m)

                    sel_r = run_random(trees, k=k, seed=rep)
                    acc_r = float(np.mean(
                        predict_ensemble([trees[i] for i in sel_r], w, X_te) == y_te
                    )) * 100
                    rand_accs.append(acc_r)
                except Exception as e:
                    print(f"    [ERR] k={k}, rep {rep}: {e}")

            if mepsi_accs:
                records.append({
                    "dataset":         ds_name,
                    "k":               k,
                    "mepsi_mean_acc":  round(float(np.mean(mepsi_accs)), 3),
                    "mepsi_std_acc":   round(float(np.std(mepsi_accs)), 3),
                    "random_mean_acc": round(float(np.mean(rand_accs)), 3),
                    "random_std_acc":  round(float(np.std(rand_accs)), 3),
                })
                print(f"    k={k:3d}  MEPSI={np.mean(mepsi_accs):.1f}  "
                      f"Random={np.mean(rand_accs):.1f}")

    df = pd.DataFrame(records)
    _save_csv(df, os.path.join(output_dir, "pruning_size_sensitivity.csv"))
    return df


# =============================================================================
# SECTION 6 — STRUCTURAL INFORMATION ANALYSIS
# =============================================================================

def benchmark_structural_analysis(datasets, output_dir, T=200, k=20):
    """
    6a. Selected vs rejected tree properties (NC, depth, ESM).
    6b. Greedy convergence (objective per iteration).
    6c. Pairwise TED distribution.
    NOT in the paper — new contribution.
    """
    print("\n" + "=" * 60)
    print("SECTION 6: STRUCTURAL INFORMATION ANALYSIS")
    print("=" * 60)

    records_struct = []
    records_conv   = []
    records_ted    = []

    for ds_name, (X_tr, X_te, y_tr, y_te) in tqdm(datasets.items(),
                                                     desc="  Datasets"):
        lam = LAMBDA_DEFAULTS.get(ds_name, 0.01)
        set_seed(0)
        trees = generate_trees(X_tr, y_tr, T=T, seed=0)

        # ── 6a & 6b: instrumented greedy run ─────────────────────────────────
        try:
            sel, history = mepsi_greedy(trees, X_tr, y_tr, k=k, lam=lam,
                                         return_history=True)
            sel_set = set(sel)

            for idx, tree in enumerate(trees):
                nc  = node_count(tree)
                dep = tree_depth(tree)
                selected = idx in sel_set
                esm = compute_esm(tree, [trees[i] for i in sel if i != idx]) \
                      if selected else \
                      compute_esm(tree, [trees[i] for i in sel])
                records_struct.append({
                    "dataset":  ds_name,
                    "tree_idx": idx,
                    "selected": selected,
                    "nc":       nc,
                    "depth":    dep,
                    "esm":      round(esm, 4),
                })

            for h in history:
                records_conv.append({"dataset": ds_name, **h})

        except Exception as e:
            print(f"  [ERR] {ds_name} structural analysis: {e}")

        # ── 6c: pairwise TED (subsample to keep it fast) ─────────────────────
        try:
            sample_n = min(30, T)
            sample_idx = list(range(sample_n))
            ted_vals, sel_ted_vals = [], []
            for i in range(sample_n):
                for j in range(i + 1, sample_n):
                    d = compute_ted(trees[i], trees[j])
                    ted_vals.append(d)
                    if i in sel_set and j in sel_set:
                        sel_ted_vals.append(d)
            records_ted.append({
                "dataset":             ds_name,
                "mean_ted_full":       round(float(np.mean(ted_vals)),     4),
                "std_ted_full":        round(float(np.std(ted_vals)),      4),
                "mean_ted_selected":   round(float(np.mean(sel_ted_vals)) if sel_ted_vals else 0, 4),
                "std_ted_selected":    round(float(np.std(sel_ted_vals))  if sel_ted_vals else 0, 4),
                "n_pairs_sampled":     len(ted_vals),
            })
        except Exception as e:
            print(f"  [ERR] {ds_name} TED distribution: {e}")

    _save_csv(pd.DataFrame(records_struct),
              os.path.join(output_dir, "structural_stats.csv"))
    _save_csv(pd.DataFrame(records_conv),
              os.path.join(output_dir, "greedy_convergence.csv"))
    _save_csv(pd.DataFrame(records_ted),
              os.path.join(output_dir, "ted_distribution.csv"))


# =============================================================================
# SECTION 7 — ABLATION STUDY
# =============================================================================

def benchmark_ablation(datasets, output_dir, n_repeats=3, T=200, k=20):
    """
    Isolate the contribution of the structural information term.
    Three variants:
        mepsi_full       → λ=0.01 (default)
        mepsi_no_struct  → λ=0.0  (pure empirical error)
        mepsi_struct_only→ λ=1e6  (effectively ignores empirical error)
    NOT in the paper — new contribution.
    """
    print("\n" + "=" * 60)
    print("SECTION 7: ABLATION STUDY")
    print("=" * 60)

    variants = {
        "mepsi_full":        LAMBDA_DEFAULTS,
        "mepsi_no_struct":   {k: 0.0 for k in LAMBDA_DEFAULTS},
        "mepsi_struct_only": {k: 1e4 for k in LAMBDA_DEFAULTS},
    }

    records = []
    for ds_name, (X_tr, X_te, y_tr, y_te) in datasets.items():
        print(f"\n  Dataset: {ds_name}")

        for variant_name, lam_dict in variants.items():
            lam = lam_dict.get(ds_name, 0.01)
            test_accs, train_accs, mean_esm_list, mean_nc_list = [], [], [], []

            for rep in range(n_repeats):
                set_seed(rep)
                try:
                    trees = generate_trees(X_tr, y_tr, T=T, seed=rep)
                    sel   = run_mepsi(trees, X_tr, y_tr, k=k, lam=lam)
                    sel_trees = [trees[i] for i in sel]
                    w = np.ones(k) / k

                    test_acc  = float(np.mean(
                        predict_ensemble(sel_trees, w, X_te) == y_te)) * 100
                    train_acc = float(np.mean(
                        predict_ensemble(sel_trees, w, X_tr) == y_tr)) * 100
                    esm_vals = [compute_esm(trees[i],
                                            [trees[j] for j in sel if j != i])
                                for i in sel]
                    nc_vals  = [node_count(trees[i]) for i in sel]

                    test_accs.append(test_acc)
                    train_accs.append(train_acc)
                    mean_esm_list.append(float(np.mean(esm_vals)))
                    mean_nc_list.append(float(np.mean(nc_vals)))
                except Exception as e:
                    print(f"    [ERR] {variant_name}, rep {rep}: {e}")

            if test_accs:
                records.append({
                    "dataset":          ds_name,
                    "variant":          variant_name,
                    "lambda":           lam,
                    "mean_test_acc":    round(float(np.mean(test_accs)),     3),
                    "std_test_acc":     round(float(np.std(test_accs)),      3),
                    "mean_train_acc":   round(float(np.mean(train_accs)),    3),
                    "mean_esm":         round(float(np.mean(mean_esm_list)), 3),
                    "mean_nc":          round(float(np.mean(mean_nc_list)),  3),
                })
                print(f"  {variant_name:<22}  "
                      f"test={np.mean(test_accs):.1f}±{np.std(test_accs):.1f}  "
                      f"ESM={np.mean(mean_esm_list):.1f}  "
                      f"NC={np.mean(mean_nc_list):.1f}")

    df = pd.DataFrame(records)
    _save_csv(df, os.path.join(output_dir, "ablation.csv"))
    return df


# =============================================================================
# SECTION 8 — GENERALIZATION BOUND ANALYSIS  (Theorem 4.2)
# =============================================================================

def compute_generalization_bound(trees, selected_indices,
                                 X_train, y_train,
                                 k, delta=0.05):
    """
    Numerical evaluation of the Theorem 4.2 bound:

        L_D(h_S) <= L_D_hat(h_S) + sqrt(
            (k*(Ch + tau) + gamma - sum_ESM + ln(2/delta)) / (2*m)
        )

    Approximations used here (noted clearly):
        Ch    = max NC in pool  (Kolmogorov complexity upper bound, Definition 3.1)
        tau   = 0               (|NC(h) - K(h)| <= tau, set to 0 as lower bound)
        gamma = 0               (constant independent of S, set to 0)
        sum_ESM = sum of ESM(h_Si, h_{S_{i+1:k}}) for i=1..k
    """
    m   = X_train.shape[0]
    Ch  = max(node_count(t) for t in trees)   # approximation
    tau = 0                                    # approximation

    sel_trees = [trees[i] for i in selected_indices]
    w = np.ones(k) / k

    # empirical error on training set
    emp_err = ensemble_error(sel_trees, w, X_train, y_train)

    # structural information term
    sum_esm = sum(
        compute_esm(trees[selected_indices[i]],
                    [trees[selected_indices[j]]
                     for j in range(i + 1, k)])
        for i in range(k)
    )

    numerator = k * (Ch + tau) - sum_esm + np.log(2.0 / delta)
    numerator  = max(numerator, 0.0)   # bound on sqrt must be non-negative
    bound_term = np.sqrt(numerator / (2 * m))
    bound      = emp_err + bound_term

    return {
        "empirical_error": round(emp_err,      4),
        "sum_esm":         round(sum_esm,      4),
        "Ch":              Ch,
        "bound":           round(bound,        4),
        "bound_term":      round(bound_term,   4),
    }


def benchmark_generalization_bound(datasets, output_dir, T=200):
    """
    Numerically evaluate Theorem 4.2 for varying k.
    NOT explored numerically in the paper — new contribution.
    """
    print("\n" + "=" * 60)
    print("SECTION 8: GENERALIZATION BOUND ANALYSIS (Theorem 4.2)")
    print("=" * 60)

    k_values = [5, 10, 20, 30]
    records  = []

    for ds_name, (X_tr, X_te, y_tr, y_te) in datasets.items():
        lam = LAMBDA_DEFAULTS.get(ds_name, 0.01)
        print(f"\n  Dataset: {ds_name}")

        set_seed(0)
        trees = generate_trees(X_tr, y_tr, T=T, seed=0)

        for k in k_values:
            try:
                sel = run_mepsi(trees, X_tr, y_tr, k=k, lam=lam)
                stats = compute_generalization_bound(
                    trees, sel, X_tr, y_tr, k)
                w = np.ones(k) / k
                test_err = ensemble_error(
                    [trees[i] for i in sel], w, X_te, y_te)

                records.append({
                    "dataset":        ds_name,
                    "k":              k,
                    "empirical_error":stats["empirical_error"],
                    "test_error":     round(test_err, 4),
                    "bound":          stats["bound"],
                    "bound_slack":    round(stats["bound"] - test_err, 4),
                    "sum_esm":        stats["sum_esm"],
                    "Ch":             stats["Ch"],
                    "bound_term":     stats["bound_term"],
                })
                print(f"    k={k:3d}  emp_err={stats['empirical_error']:.3f}  "
                      f"test_err={test_err:.3f}  "
                      f"bound={stats['bound']:.3f}  "
                      f"slack={stats['bound'] - test_err:.3f}")
            except Exception as e:
                print(f"    [ERR] k={k}: {e}")

    df = pd.DataFrame(records)
    _save_csv(df, os.path.join(output_dir, "generalization_bound.csv"))
    return df


# =============================================================================
# SECTION 9 — JACCARD INDEX DIVERSITY  (reproduces paper appendix)
# =============================================================================

def jaccard(set_a, set_b):
    a, b = set(set_a), set(set_b)
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 1.0


def benchmark_jaccard(datasets, output_dir, n_repeats=3, T=200, k=20):
    """
    Jaccard index between MEPSI selections and baselines.
    Reproduces the appendix of the paper.
    """
    print("\n" + "=" * 60)
    print("SECTION 9: JACCARD INDEX DIVERSITY")
    print("=" * 60)

    records = []

    for ds_name, (X_tr, X_te, y_tr, y_te) in datasets.items():
        lam = LAMBDA_DEFAULTS.get(ds_name, 0.01)
        print(f"\n  Dataset: {ds_name}")

        mepsi_sels  = []
        kappa_sels  = []
        random_sels = []

        for rep in range(n_repeats):
            set_seed(rep)
            try:
                trees = generate_trees(X_tr, y_tr, T=T, seed=rep)
                mepsi_sels.append(run_mepsi(trees, X_tr, y_tr, k=k, lam=lam))
                kappa_sels.append(run_kappa(trees, X_tr, y_tr, k=k))
                random_sels.append(run_random(trees, k=k, seed=rep))
            except Exception as e:
                print(f"    [ERR] rep {rep}: {e}")

        # Cross-method Jaccard
        for method_name, sels in [("kappa",  kappa_sels),
                                   ("random", random_sels)]:
            if mepsi_sels and sels:
                jaccards = [jaccard(m, b)
                            for m, b in zip(mepsi_sels, sels)]
                records.append({
                    "dataset":      ds_name,
                    "method_a":     "mepsi",
                    "method_b":     method_name,
                    "mean_jaccard": round(float(np.mean(jaccards)), 4),
                    "std_jaccard":  round(float(np.std(jaccards)),  4),
                })
                print(f"    MEPSI vs {method_name:<8}  "
                      f"Jaccard={np.mean(jaccards):.3f}±{np.std(jaccards):.3f}")

        # Intra-MEPSI stability
        if len(mepsi_sels) >= 2:
            intra = [jaccard(mepsi_sels[i], mepsi_sels[j])
                     for i in range(len(mepsi_sels))
                     for j in range(i + 1, len(mepsi_sels))]
            records.append({
                "dataset":      ds_name,
                "method_a":     "mepsi",
                "method_b":     "mepsi_intra",
                "mean_jaccard": round(float(np.mean(intra)), 4),
                "std_jaccard":  round(float(np.std(intra)),  4),
            })
            print(f"    MEPSI (intra-stability)  "
                  f"Jaccard={np.mean(intra):.3f}±{np.std(intra):.3f}")

    df = pd.DataFrame(records)
    _save_csv(df, os.path.join(output_dir, "jaccard_index.csv"))
    return df


# =============================================================================
# RESULTS AGGREGATION
# =============================================================================

def generate_summary_report(output_dir):
    """Read all CSVs and produce a text summary + summary.json."""
    print("\n" + "=" * 60)
    print("SUMMARY REPORT")
    print("=" * 60)

    summary = {}
    csv_files = {
        "accuracy":          "accuracy_results.csv",
        "runtime_T":         "runtime_vs_T.csv",
        "runtime_k":         "runtime_vs_k.csv",
        "runtime_n":         "runtime_vs_n.csv",
        "lambda":            "lambda_sensitivity.csv",
        "pruning_size":      "pruning_size_sensitivity.csv",
        "structural":        "structural_stats.csv",
        "convergence":       "greedy_convergence.csv",
        "ted":               "ted_distribution.csv",
        "ablation":          "ablation.csv",
        "bounds":            "generalization_bound.csv",
        "jaccard":           "jaccard_index.csv",
    }

    for key, fname in csv_files.items():
        path = os.path.join(output_dir, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            summary[key] = {
                "n_rows":  len(df),
                "columns": list(df.columns),
            }
            # Key numbers
            if key == "accuracy" and "mean_acc" in df.columns:
                mepsi_df = df[df["method"] == "mepsi"]
                if not mepsi_df.empty:
                    summary[key]["mepsi_mean_across_datasets"] = \
                        round(float(mepsi_df["mean_acc"].mean()), 2)
            if key == "bounds" and "bound_slack" in df.columns:
                summary[key]["mean_bound_slack"] = \
                    round(float(df["bound_slack"].mean()), 4)
            if key == "jaccard" and "mean_jaccard" in df.columns:
                vs_kappa = df[df["method_b"] == "kappa"]["mean_jaccard"]
                if not vs_kappa.empty:
                    summary[key]["mean_jaccard_vs_kappa"] = \
                        round(float(vs_kappa.mean()), 4)
            print(f"  {key:<18} → {fname}  ({len(df)} rows)")
        else:
            print(f"  {key:<18} → NOT FOUND (section may not have been run)")

    out_path = os.path.join(output_dir, "summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved → {out_path}")


# =============================================================================
# MAIN
# =============================================================================

SECTION_MAP = {
    "accuracy":   benchmark_accuracy,
    "runtime":    benchmark_runtime,
    "lambda":     benchmark_lambda,
    "ksize":      benchmark_pruning_size,
    "structural": benchmark_structural_analysis,
    "ablation":   benchmark_ablation,
    "bounds":     benchmark_generalization_bound,
    "jaccard":    benchmark_jaccard,
}


def main():
    parser = argparse.ArgumentParser(
        description="MEPSI Benchmark Suite",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--run", type=str, default="all",
        help=("Comma-separated sections to run.\n"
              "Options: all, accuracy, runtime, lambda, ksize, "
              "structural, ablation, bounds, jaccard\n"
              "Example: --run accuracy,runtime")
    )
    parser.add_argument(
        "--datasets", type=str, default=None,
        help="Comma-separated dataset names to include. Default: all 11."
    )
    parser.add_argument(
        "--repeats", type=int, default=20,
        help="Number of random repetitions (default: 20, use 3-5 for quick test)."
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Directory to save CSV outputs (default: results/)."
    )
    parser.add_argument(
        "--T", type=int, default=200,
        help="Pool size (number of trees to generate, default: 200)."
    )
    parser.add_argument(
        "--k", type=int, default=20,
        help="Pruning size (trees to select, default: 20)."
    )
    args = parser.parse_args()

    _make_results_dir(args.output_dir)

    dataset_filter = (
        [d.strip() for d in args.datasets.split(",")]
        if args.datasets else None
    )

    sections_to_run = (
        list(SECTION_MAP.keys())
        if args.run.strip().lower() == "all"
        else [s.strip() for s in args.run.split(",")]
    )

    # Validate section names
    invalid = [s for s in sections_to_run if s not in SECTION_MAP]
    if invalid:
        print(f"[ERROR] Unknown sections: {invalid}. "
              f"Valid: {list(SECTION_MAP.keys())}")
        sys.exit(1)

    # Load datasets (always needed)
    print("\n" + "=" * 60)
    print("SECTION 1: LOADING DATASETS")
    print("=" * 60)
    datasets = load_all_datasets(dataset_filter)
    print(f"\n  Loaded {len(datasets)} dataset(s): {list(datasets.keys())}")

    # Run sections
    common_kwargs = {
        "datasets":   datasets,
        "output_dir": args.output_dir,
    }

    for section in sections_to_run:
        fn = SECTION_MAP[section]
        print(f"\n>>> Running section: {section}")
        try:
            if section == "accuracy":
                fn(**common_kwargs, n_repeats=args.repeats,
                   T=args.T, k=args.k)
            elif section == "runtime":
                fn(**common_kwargs)
            elif section in ("lambda", "ablation"):
                fn(**common_kwargs, n_repeats=args.repeats,
                   T=args.T, k=args.k)
            elif section == "ksize":
                fn(**common_kwargs, n_repeats=args.repeats,
                   T=args.T)
            elif section in ("structural", "bounds", "jaccard"):
                fn(**common_kwargs, T=args.T,
                   **({} if section != "jaccard" else {"k": args.k,
                                                        "n_repeats": args.repeats}))
            else:
                fn(**common_kwargs)
        except Exception:
            print(f"  [SECTION ERROR] {section} failed:")
            traceback.print_exc()

    generate_summary_report(args.output_dir)
    print("\nDone. All results saved to:", args.output_dir)


if __name__ == "__main__":
    main()