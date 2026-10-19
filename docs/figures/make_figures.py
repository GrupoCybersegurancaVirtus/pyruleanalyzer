"""Draw the README's figures from real pyRuleAnalyzer runs.

Every number, rule and threshold in the figures is computed here, on small
public datasets that ship with scikit-learn; nothing is typed in by hand
(the diagrams that explain an idea, such as the overview, are drawn, the rest
is measured). Re-run after a change that could move them:

    python docs/figures/make_figures.py

Needs matplotlib (only for this script). The benchmark figure also needs the
compiled C extension to show its speed-up.
"""

import contextlib
import io
import os
import pickle
import sys
import tempfile
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from sklearn.datasets import (  # noqa: E402
    load_breast_cancer, load_iris, load_wine, make_classification,
)
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from pyruleanalyzer import PyRuleAnalyzer, _accel  # noqa: E402

# Colours: the same palette as the CPNCheck figures (validated for colour-vision
# deficiencies), a neutral for context, and text inks.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_LIGHT = "#86b6ef"
NEUTRAL = "#b5b3ad"
SURFACE = "#fcfcfb"
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8984"
CLASS_COLORS = [BLUE, ORANGE, AQUA]
DPI = 200

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
    "axes.edgecolor": INK_3, "axes.labelcolor": INK_2,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


# Function to save a figure into this directory.
def save(fig, name: str) -> None:
    """Write ``name`` as a PNG and close the figure.

    Args:
        fig (Figure): The figure.
        name (str): File name.
    """
    path = os.path.join(HERE, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.15,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print("wrote", os.path.relpath(path, ROOT))


# Function to draw a rounded box with a title and a line of detail.
def box(ax, x, y, w, h, title, detail="", edge=BLUE, fill="white",
        dashed=False, title_size=10.5, detail_size=8.3):
    """A labelled box centred at ``(x, y)``.

    Args:
        ax (Axes): Where to draw.
        x, y (float): Centre.
        w, h (float): Size.
        title (str): Bold first line.
        detail (str): Smaller lines below.
        edge (str): Border colour.
        fill (str): Fill colour.
        dashed (bool): Dashed border (optional parts).
        title_size (float): Size of the title.
        detail_size (float): Size of the detail.
    """
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                linewidth=1.6, edgecolor=edge, facecolor=fill,
                                linestyle="--" if dashed else "-"))
    top = y + h / 2 - 0.2
    ax.text(x, top if detail else y, title, ha="center", va="center",
            fontsize=title_size, fontweight="bold", color=INK)
    if detail:
        ax.text(x, top - 0.17, detail, ha="center", va="top",
                fontsize=detail_size, color=INK_2, linespacing=1.3)


# Function to draw an arrow between two points.
def arrow(ax, a, b, color=INK_3, label="", dashed=False, rad=0.0, lw=1.4,
          label_offset=(0, 0.09), fontsize=8.3):
    """An arrow from ``a`` to ``b``, optionally labelled at its middle.

    Args:
        ax (Axes): Where to draw.
        a, b (tuple): End points.
        color (str): Colour.
        label (str): Text at the middle.
        dashed (bool): Dashed line.
        rad (float): Curvature.
        lw (float): Width.
        label_offset (tuple): Offset of the label.
        fontsize (float): Label size.
    """
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11,
                                 color=color, linewidth=lw,
                                 linestyle="--" if dashed else "-",
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=2, shrinkB=2))
    if label:
        ax.text((a[0] + b[0]) / 2 + label_offset[0],
                (a[1] + b[1]) / 2 + label_offset[1], label, ha="center",
                va="center", fontsize=fontsize, color=INK_2,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.2))


# Function to draw a Petri-net place.
def place(ax, x, y, name, r=0.3, edge=BLUE, label_dy=-0.12, size=8.3):
    """A circle with its name below.

    Args:
        ax (Axes): Where to draw.
        x, y (float): Centre.
        name (str): Name written under the circle.
        r (float): Radius.
        edge (str): Border colour.
        label_dy (float): Gap between the circle and the name.
        size (float): Font size of the name.
    """
    ax.add_patch(Circle((x, y), r, facecolor="white", edgecolor=edge,
                        linewidth=1.7))
    ax.text(x, y - r + label_dy, name, ha="center", va="top", fontsize=size,
            fontweight="bold")


# Function to draw a Petri-net transition.
def transition(ax, x, y, name, w=1.0, h=0.34, sub=False, size=8.0):
    """A box, with a double border when it is a substitution transition.

    Args:
        ax (Axes): Where to draw.
        x, y (float): Centre.
        name (str): Text inside.
        w, h (float): Size.
        sub (bool): Substitution transition (a page lives behind it).
        size (float): Font size.
    """
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="square,pad=0", facecolor="white",
                                edgecolor=INK, linewidth=1.5))
    if sub:
        ax.add_patch(FancyBboxPatch((x - w / 2 - 0.05, y - h / 2 - 0.05),
                                    w + 0.1, h + 0.1, boxstyle="square,pad=0",
                                    facecolor="none", edgecolor=INK,
                                    linewidth=1.0))
    ax.text(x, y, name, ha="center", va="center", fontsize=size,
            fontweight="bold")


# Function to load a wine split used by several figures.
def wine_split(features=None, seed=0):
    """Wine, split into train, validation and test (stratified).

    Args:
        features (list): Columns to keep (all if None).
        seed (int): Random seed of the splits.

    Returns:
        tuple: ``X_train, X_val, X_test, y_train, y_val, y_test``.
    """
    data = load_wine(as_frame=True)
    X, y = data.data, data.target
    if features:
        X = X[features]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                              random_state=seed, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.3,
                                              random_state=seed, stratify=y_tr)
    return X_tr, X_va, X_te, y_tr, y_va, y_te


# ---------------------------------------------------------------------------
# 1. How pyRuleAnalyzer works
# ---------------------------------------------------------------------------

# Function to draw the overview of the toolkit.
def pipeline() -> None:
    """From a scikit-learn model to rules, exports and a verified net."""
    fig, ax = plt.subplots(figsize=(12.5, 4.3))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 4.3)
    ax.axis("off")
    y = 2.95
    steps = [
        (1.15, "scikit-learn", "a Decision Tree,\nRandom Forest or\nGradient Boosting"),
        (3.7, "Extract", "every root-to-leaf\npath becomes a\nreadable rule"),
        (6.25, "Refine", "merge redundant\nsiblings, drop the\nrules nobody uses"),
        (8.8, "Predict", "vectorised, from the\nrules, in C or numpy\n(same answers)"),
        (11.35, "Export", ".py  .bin  .h  .ino\nstandalone, for a\nlaptop or a board"),
    ]
    for x, title, detail in steps:
        box(ax, x, y, 2.15, 1.3, title, detail)
    for (x1, *_), (x2, *_) in zip(steps, steps[1:]):
        arrow(ax, (x1 + 1.1, y), (x2 - 1.1, y), color=BLUE, lw=1.8)
    box(ax, 4.0, 0.78, 5.6, 1.05, "Verify (optional): a Petri net of the model",
        "the rules become a Hierarchical Coloured Petri Net; CPNCheck explores\n"
        "it and proves the refined model still behaves like a classifier",
        edge=AQUA, dashed=True)
    arrow(ax, (3.2, y - 0.68), (3.2, 1.32), color=AQUA, dashed=True)
    arrow(ax, (5.5, 1.32), (6.0, y - 0.68), color=AQUA, dashed=True)
    box(ax, 10.2, 0.78, 3.9, 1.05, "Deploy (optional)",
        "arduino-cli compiles and uploads;\nhardware-in-the-loop checks the board",
        edge=ORANGE, dashed=True)
    arrow(ax, (11.1, y - 0.68), (10.9, 1.32), color=ORANGE, dashed=True)
    ax.text(0.05, 4.25, "How pyRuleAnalyzer works", fontsize=13,
            fontweight="bold", va="top")
    save(fig, "pipeline.png")


# ---------------------------------------------------------------------------
# 2. A tree and its rules
# ---------------------------------------------------------------------------

# Function to lay out a scikit-learn tree for drawing.
def tree_layout(tree):
    """Positions of the nodes of a fitted ``tree_`` (leaves evenly spaced).

    Args:
        tree: A scikit-learn ``Tree``.

    Returns:
        dict: Node id to ``(x, y)``.
    """
    pos = {}
    counter = [0]

    # Function to place a subtree and return its x.
    def walk(node, depth):
        """Place ``node``; leaves take the next slot, parents the middle."""
        left, right = tree.children_left[node], tree.children_right[node]
        if left == -1:
            x = counter[0]
            counter[0] += 1
        else:
            xl = walk(left, depth + 1)
            xr = walk(right, depth + 1)
            x = (xl + xr) / 2
        pos[node] = (x, -depth)
        return x

    walk(0, 0)
    return pos


# Function to draw a tree beside the rules extracted from it.
def rules() -> None:
    """A real iris tree on the left, its rules on the right."""
    data = load_iris(as_frame=True)
    X, y = data.data, data.target
    X.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    model = PyRuleAnalyzer.new_model("Decision Tree",
                                     {"max_depth": 3, "random_state": 0})
    model.fit(X, y)
    rule_list = model.classifier.initial_rules
    # Refit the same estimator to read its structure (same seed, same tree).
    est = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)
    tree = est.tree_
    pos = tree_layout(tree)
    leaves = [n for n in sorted(pos) if tree.children_left[n] == -1]
    leaves.sort(key=lambda n: pos[n][0])
    assert len(leaves) == len(rule_list), (len(leaves), len(rule_list))
    names = list(X.columns)

    fig = plt.figure(figsize=(12.2, 5.0))
    ax = fig.add_axes([0.0, 0.0, 0.58, 1.0])
    ax.axis("off")
    ax.set_xlim(-0.9, max(p[0] for p in pos.values()) + 0.9)
    ax.set_ylim(-3.75, 0.55)
    for node, (x, yy) in pos.items():
        for child, tag in ((tree.children_left[node], "yes"),
                           (tree.children_right[node], "no")):
            if child == -1:
                continue
            cx, cy = pos[child]
            ax.plot([x, cx], [yy - 0.2, cy + 0.2], color=INK_3, linewidth=1.2,
                    zorder=1)
            ax.text((x + cx) / 2 + (-0.12 if tag == "yes" else 0.12),
                    (yy + cy) / 2, tag, fontsize=7.5, color=INK_3,
                    ha="right" if tag == "yes" else "left", va="center")
    for node, (x, yy) in pos.items():
        if tree.children_left[node] == -1:
            cls = int(np.argmax(tree.value[node]))
            idx = leaves.index(node)
            ax.add_patch(FancyBboxPatch((x - 0.42, yy - 0.2), 0.84, 0.4,
                                        boxstyle="round,pad=0.01,rounding_size=0.06",
                                        facecolor=CLASS_COLORS[cls], alpha=0.2,
                                        edgecolor="none", zorder=2))
            ax.add_patch(FancyBboxPatch((x - 0.42, yy - 0.2), 0.84, 0.4,
                                        boxstyle="round,pad=0.01,rounding_size=0.06",
                                        facecolor="none",
                                        edgecolor=CLASS_COLORS[cls],
                                        linewidth=1.6, zorder=3))
            ax.text(x, yy, f"Rule {idx + 1}\nclass {cls}", ha="center",
                    va="center", fontsize=8, fontweight="bold", zorder=4)
        else:
            f = names[tree.feature[node]]
            t = tree.threshold[node]
            ax.add_patch(FancyBboxPatch((x - 0.62, yy - 0.2), 1.24, 0.4,
                                        boxstyle="round,pad=0.01,rounding_size=0.06",
                                        facecolor="white", edgecolor=INK,
                                        linewidth=1.3, zorder=2))
            ax.text(x, yy, f"{f} <= {t:.2f}", ha="center", va="center",
                    fontsize=8, zorder=4)
    ax.text(-0.85, 0.5, "The tree", fontsize=12, fontweight="bold", va="top")

    ax2 = fig.add_axes([0.6, 0.0, 0.4, 1.0])
    ax2.axis("off")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.text(0.0, 0.97, "The rules", fontsize=12, fontweight="bold", va="top")
    n = len(rule_list)
    for i, r in enumerate(rule_list):
        top = 0.86 - i * (0.82 / n)
        cls = int(r.class_)
        ax2.add_patch(Rectangle((0.0, top - 0.115), 0.012, 0.13,
                                facecolor=CLASS_COLORS[cls], edgecolor="none"))
        ax2.text(0.04, top, f"Rule {i + 1}  ->  class {cls}", fontsize=9,
                 fontweight="bold", va="top")
        conds = [c.replace(c.split()[-1], f"{float(c.split()[-1]):.2f}")
                 for c in r.conditions]
        ax2.text(0.04, top - 0.045, "\n".join(("   " if j == 0 else "AND ") + c
                                              for j, c in enumerate(conds)),
                 fontsize=8.1, color=INK_2, va="top", family="DejaVu Sans Mono",
                 linespacing=1.25)
    save(fig, "rules.png")


# ---------------------------------------------------------------------------
# 3. What refinement does to the rules
# ---------------------------------------------------------------------------

# Function to turn a rule into the box of the feature space it covers.
def rule_box(rule, features, limits):
    """Lower and upper bound of a rule on each feature.

    Args:
        rule (Rule): The rule.
        features (list): Feature names (x, y).
        limits (list): ``[(xmin, xmax), (ymin, ymax)]`` to close open sides.

    Returns:
        list: ``[(lo, hi), (lo, hi)]``.
    """
    bounds = [list(l) for l in limits]
    for feat, op, thr in rule.parsed_conditions:
        k = features.index(feat)
        if op == "<=":
            bounds[k][1] = min(bounds[k][1], thr)
        else:
            bounds[k][0] = max(bounds[k][0], thr)
    return [tuple(b) for b in bounds]


# Function to draw the decision regions before and after refinement.
def refinement():
    """Rule boxes of one wine tree, initial and refined, on two features.

    Returns:
        dict: The numbers shown (rules and accuracy before and after).
    """
    feats = ["proline", "flavanoids"]
    X_tr, X_va, X_te, y_tr, y_va, y_te = wine_split(feats)
    model = PyRuleAnalyzer.new_model("Decision Tree", {"random_state": 0})
    model.fit(X_tr, y_tr)
    initial = list(model.classifier.initial_rules)
    with contextlib.redirect_stdout(io.StringIO()):
        stats = model.execute_rule_refinement(
            X=X_va, y=y_va, remove_below_n_classifications=1)
    final = list(model.classifier.final_rules)
    acc_i = float((model.predict(X_te, use_refined=False) == y_te).mean())
    acc_f = float((model.predict(X_te) == y_te).mean())
    limits = [(X_tr[feats[0]].min() - 40, X_tr[feats[0]].max() + 40),
              (X_tr[feats[1]].min() - 0.2, X_tr[feats[1]].max() + 0.2)]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6),
                             gridspec_kw={"wspace": 0.12})
    for ax, rule_set, title, acc in (
            (axes[0], initial, "Initial", acc_i),
            (axes[1], final, "Refined", acc_f)):
        boxes = [rule_box(r, feats, limits) for r in rule_set]
        for r, b in zip(rule_set, boxes):
            cls = int(r.class_)
            ax.add_patch(Rectangle((b[0][0], b[1][0]), b[0][1] - b[0][0],
                                   b[1][1] - b[1][0],
                                   facecolor=CLASS_COLORS[cls], alpha=0.22,
                                   edgecolor=CLASS_COLORS[cls], linewidth=1.3))
        for c in range(3):
            m = np.asarray(y_te) == c
            ax.scatter(X_te[feats[0]][m], X_te[feats[1]][m], s=14,
                       color=CLASS_COLORS[c], edgecolor="white", linewidth=0.5,
                       zorder=3, label=f"class {c}")
        # Rules of one tree must partition the space: no two boxes overlap.
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                ox = min(boxes[i][0][1], boxes[j][0][1]) - max(boxes[i][0][0], boxes[j][0][0])
                oy = min(boxes[i][1][1], boxes[j][1][1]) - max(boxes[i][1][0], boxes[j][1][0])
                assert not (ox > 1e-9 and oy > 1e-9), "rule boxes overlap"
        ax.set_xlim(*limits[0])
        ax.set_ylim(*limits[1])
        ax.set_xlabel(feats[0])
        ax.set_title(f"{title}: {len(rule_set)} rules", loc="left",
                     fontsize=11.5, fontweight="bold")
        ax.text(0.99, 0.02, f"test accuracy {acc:.3f}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9, color=INK,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=2))
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel(feats[1])
    axes[1].set_yticklabels([])
    axes[0].legend(loc="upper left", frameon=False, fontsize=8.3)
    fig.suptitle("Refinement: fewer, larger boxes; the boxes still never "
                 "overlap", x=0.01, y=1.02, ha="left", fontsize=12.5,
                 fontweight="bold")
    save(fig, "refinement.png")
    return {"rules_before": stats["rules_before"],
            "rules_after": stats["rules_after"], "acc_initial": acc_i,
            "acc_final": acc_f}


# ---------------------------------------------------------------------------
# 4. How each model type decides
# ---------------------------------------------------------------------------

# Function to match a sample against the rules of a tree.
def matching_rule(rule_list, sample):
    """The rule whose conditions the sample satisfies.

    Args:
        rule_list (list): Rules of one tree.
        sample (dict): Feature values.

    Returns:
        Rule: The matching rule.
    """
    for r in rule_list:
        if all((sample[f] <= t) if op == "<=" else (sample[f] > t)
               for f, op, t in r.parsed_conditions):
            return r
    raise AssertionError("no rule matches")


# Function to draw the decision of a forest and a boosting model on one sample.
def strategies() -> None:
    """One real sample through a tree, a forest and a boosting model."""
    fig = plt.figure(figsize=(12.6, 4.6))
    # ---- Decision tree
    data = load_iris(as_frame=True)
    X, y = data.data, data.target
    X.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    dt = PyRuleAnalyzer.new_model("Decision Tree",
                                  {"max_depth": 3, "random_state": 0}).fit(X, y)
    sample = X.iloc[[100]]
    s = sample.iloc[0].to_dict()
    r = matching_rule(dt.classifier.initial_rules, s)
    assert int(r.class_) == int(dt.predict(sample)[0])
    ax = fig.add_axes([0.0, 0.0, 0.31, 1.0])
    ax.axis("off")
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4.6)
    ax.text(0.0, 4.55, "Decision Tree: first match", fontsize=11.5,
            fontweight="bold", va="top")
    ax.text(0.0, 4.15, "one tree, one rule fires", fontsize=8.5, color=INK_2,
            va="top")
    box(ax, 2.0, 3.0, 3.6, 0.9, "sample",
        "  ".join(f"{k.split('_')[0][0]}{k.split('_')[1][0]}={v:.1f}"
                  for k, v in s.items()), detail_size=8.0)
    arrow(ax, (2.0, 2.5), (2.0, 2.0), color=BLUE, lw=1.8)
    box(ax, 2.0, 1.45, 3.6, 1.0, "the one rule it satisfies",
        "\n".join(c if len(c) < 30 else c[:30] for c in
                  [f"{f} {op} {t:.2f}" for f, op, t in r.parsed_conditions]),
        detail_size=8.0)
    arrow(ax, (2.0, 0.92), (2.0, 0.5), color=BLUE, lw=1.8)
    ax.text(2.0, 0.28, f"class {int(r.class_)}", ha="center", va="center",
            fontsize=12, fontweight="bold", color=CLASS_COLORS[int(r.class_)])

    # ---- Random forest
    rf = PyRuleAnalyzer.new_model("Random Forest",
                                  {"n_estimators": 3, "max_depth": 2,
                                   "random_state": 0}).fit(X, y)
    by_tree = {}
    for rule in rf.classifier.initial_rules:
        by_tree.setdefault(rule.name.split("_")[0], []).append(rule)
    dists = []
    for tree_rules in by_tree.values():
        rr = matching_rule(tree_rules, s)
        d = np.asarray(rr.class_distribution, dtype=float)
        dists.append(d / d.sum())
    total = np.sum(dists, axis=0)
    assert int(np.argmax(total)) == int(rf.predict(sample)[0])
    ax = fig.add_axes([0.345, 0.0, 0.31, 1.0])
    ax.axis("off")
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4.6)
    ax.text(0.0, 4.55, "Random Forest: soft vote", fontsize=11.5,
            fontweight="bold", va="top")
    ax.text(0.0, 4.15, "every tree answers with a distribution", fontsize=8.5,
            color=INK_2, va="top")
    for k, d in enumerate(dists):
        x0 = 0.15 + k * 1.3
        ax.text(x0 + 0.5, 3.65, f"tree {k}", ha="center", fontsize=8.5,
                fontweight="bold")
        for c in range(3):
            ax.add_patch(Rectangle((x0 + c * 0.34, 2.75),
                                   0.28, 0.75 * float(d[c]) + 0.001,
                                   facecolor=CLASS_COLORS[c], edgecolor="none"))
        ax.plot([x0 - 0.04, x0 + 1.02], [2.75, 2.75], color=INK_3, linewidth=0.8)
        arrow(ax, (x0 + 0.5, 2.6), (2.0, 2.0), color=INK_3, lw=1.0)
    box(ax, 2.0, 1.55, 3.4, 0.8, "sum the distributions",
        "  ".join(f"c{c}: {total[c]:.2f}" for c in range(3)), detail_size=8.3)
    arrow(ax, (2.0, 1.12), (2.0, 0.5), color=BLUE, lw=1.8)
    ax.text(2.0, 0.28, f"argmax: class {int(np.argmax(total))}", ha="center",
            va="center", fontsize=12, fontweight="bold",
            color=CLASS_COLORS[int(np.argmax(total))])

    # ---- GBDT
    bc = load_breast_cancer(as_frame=True)
    gb = PyRuleAnalyzer.new_model("Gradient Boosting Decision Trees",
                                  {"n_estimators": 3, "max_depth": 2,
                                   "random_state": 0}).fit(bc.data, bc.target)
    gs = bc.data.iloc[[0]]
    sg = gs.iloc[0].to_dict()
    init = list(gb.classifier._gbdt_init_scores.values())[0]
    parts = []
    for m in (1, 2, 3):
        tr = [r for r in gb.classifier.initial_rules
              if r.name.startswith(f"GBDT1T{m}_")]
        parts.append(float(matching_rule(tr, sg).contribution))
    score = init + sum(parts)
    assert int(score >= 0) == int(gb.predict(gs)[0])
    ax = fig.add_axes([0.69, 0.0, 0.31, 1.0])
    ax.axis("off")
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4.6)
    ax.text(0.0, 4.55, "GBDT: additive score", fontsize=11.5,
            fontweight="bold", va="top")
    ax.text(0.0, 4.15, "each stage adds its leaf's contribution", fontsize=8.5,
            color=INK_2, va="top")
    items = [("init", init)] + [(f"stage {m}", p)
                                for m, p in zip((1, 2, 3), parts)]
    for k, (label, v) in enumerate(items):
        yy = 3.55 - k * 0.5
        ax.text(0.2, yy, label, fontsize=9, va="center", color=INK_2)
        width = min(abs(v) * 3.0, 1.9)
        x0 = 2.3
        ax.add_patch(Rectangle((x0 if v >= 0 else x0 - width, yy - 0.14),
                               width, 0.28, facecolor=BLUE if v >= 0 else ORANGE,
                               edgecolor="none"))
        ax.text(x0 + (width + 0.08 if v >= 0 else -width - 0.08), yy,
                f"{v:+.2f}", fontsize=8.5, va="center",
                ha="left" if v >= 0 else "right")
    ax.plot([2.3, 2.3], [1.65, 3.8], color=INK_3, linewidth=0.8)
    box(ax, 2.0, 1.25, 3.4, 0.7, f"score = {score:+.2f}",
        "binary: class 1 when score >= 0", detail_size=8.3)
    arrow(ax, (2.0, 0.9), (2.0, 0.5), color=BLUE, lw=1.8)
    cls = int(score >= 0)
    ax.text(2.0, 0.28, f"class {cls}", ha="center", va="center", fontsize=12,
            fontweight="bold", color=CLASS_COLORS[cls])
    save(fig, "strategies.png")


# ---------------------------------------------------------------------------
# 5. Speed and size
# ---------------------------------------------------------------------------

# Function to time a callable, best of several runs.
def best_of(fn, repeat=5):
    """The fastest of ``repeat`` runs, in seconds.

    Args:
        fn (callable): What to time.
        repeat (int): Runs.

    Returns:
        float: Seconds.
    """
    fn()
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


# Function to measure inference speed and file size of the engines.
def measure_benchmark():
    """Throughput of each engine and size of each file, on one forest.

    Returns:
        tuple: ``(throughput rows, size rows, description)``.
    """
    X, y = make_classification(n_samples=30000, n_features=20, n_informative=10,
                               n_classes=3, random_state=0)
    X_tr, X_te, y_tr, _ = train_test_split(X, y, test_size=1 / 3,
                                           random_state=0)
    X_tr = X_tr.astype(np.float32)
    X_te = X_te.astype(np.float32)
    est = RandomForestClassifier(n_estimators=100, max_depth=10,
                                 random_state=0, n_jobs=1).fit(X_tr, y_tr)
    model = PyRuleAnalyzer.from_sklearn(est)
    n = len(X_te)
    assert (model.predict(X_te) == est.predict(X_te)).all()
    model.predict(X_te)
    rows = [("scikit-learn predict", n / best_of(lambda: est.predict(X_te)),
             NEUTRAL)]
    rows.append(("pyRuleAnalyzer, C extension",
                 n / best_of(lambda: model.predict(X_te)), BLUE))
    had_c = _accel.HAS_C_EXTENSION
    _accel.HAS_C_EXTENSION = False
    try:
        rows.append(("pyRuleAnalyzer, numpy fallback",
                     n / best_of(lambda: model.predict(X_te), repeat=3),
                     BLUE_LIGHT))
    finally:
        _accel.HAS_C_EXTENSION = had_c
    out = tempfile.mkdtemp()
    names = [f"f{i}" for i in range(X_te.shape[1])]
    with contextlib.redirect_stdout(io.StringIO()):
        files = model.export(os.path.join(out, "m"),
                             formats=["python", "binary", "c"])
    ns = {}
    with open(files["python"], encoding="utf-8") as fh:
        exec(compile(fh.read(), files["python"], "exec"), ns)
    sample_rows = [dict(zip(model.feature_names, map(float, r)))
                   for r in X_te[:500]]
    py_predict = ns["predict"]
    rows.append(("exported .py, one sample at a time",
                 len(sample_rows) / best_of(
                     lambda: [py_predict(r) for r in sample_rows], repeat=3),
                 AQUA))
    sizes = [("scikit-learn pickle", len(pickle.dumps(est)) / 1024, NEUTRAL),
             ("exported .py", os.path.getsize(files["python"]) / 1024, AQUA),
             ("binary .bin", os.path.getsize(files["binary"]) / 1024, BLUE),
             ("C header .h", os.path.getsize(files["c"]) / 1024, BLUE_LIGHT)]
    desc = (f"Random Forest, {est.n_estimators} trees, depth <= 10, "
            f"{X_tr.shape[1]} features, {n:,} test samples")
    return rows, sizes, desc


# Function to draw the speed and size of the engines.
def benchmark() -> None:
    """Throughput and size, measured on this machine."""
    rows, sizes, desc = measure_benchmark()
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.7),
                             gridspec_kw={"wspace": 0.75})
    for ax, data, title, unit, fmt in (
            (axes[0], rows, "Speed: samples per second", "samples / s",
             lambda v: f"{v:,.0f}".replace(",", " ")),
            (axes[1], sizes, "Size on disk", "KiB",
             lambda v: f"{v:,.0f} KiB".replace(",", " "))):
        labels = [d[0] for d in data][::-1]
        values = [d[1] for d in data][::-1]
        colors = [d[2] for d in data][::-1]
        ys = range(len(data))
        ax.barh(list(ys), values, color=colors, height=0.62,
                edgecolor=SURFACE, linewidth=2)
        top = max(values)
        for yy, v in zip(ys, values):
            ax.text(v + top * 0.015, yy, fmt(v), va="center", fontsize=8.6,
                    color=INK)
        ax.set_yticks(list(ys))
        ax.set_yticklabels(labels, fontsize=8.6)
        ax.set_xlim(0, top * 1.3)
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold")
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.grid(True, color="#e8e6e1", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel(unit, fontsize=8.6)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _: f"{v / 1000:g}k" if v >= 1000 else f"{v:g}"))
    fig.suptitle("The same forest, run and stored in different ways", x=0.01,
                 y=1.12, ha="left", fontsize=12.5, fontweight="bold")
    fig.text(0.01, 1.04, desc, fontsize=8.5, color=INK_2, ha="left")
    save(fig, "benchmark.png")
    return rows, sizes


# ---------------------------------------------------------------------------
# 6. The Petri net of a model
# ---------------------------------------------------------------------------

# Function to draw how each model family becomes a net.
def hcpn() -> None:
    """Names and shapes of the nets the exporter really writes."""
    fig, axes = plt.subplots(3, 1, figsize=(12.2, 9.6),
                             gridspec_kw={"hspace": 0.12})
    # ---- Decision tree: one transition per rule
    ax = axes[0]
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0, 3.1)
    ax.axis("off")
    data = load_iris(as_frame=True)
    X, y = data.data, data.target
    X.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    dt = PyRuleAnalyzer.new_model("Decision Tree",
                                  {"max_depth": 2, "random_state": 0}).fit(X, y)
    rl = dt.classifier.initial_rules
    ax.text(0.0, 3.05, "Decision Tree: one page, one transition per rule",
            fontsize=11.5, fontweight="bold", va="top")
    place(ax, 1.3, 1.35, "Input")
    place(ax, 10.9, 1.35, "Prediction")
    ys = [2.3, 1.35, 0.4]
    for yy, r in zip(ys, rl):
        guard = " and ".join(
            f"{f.replace('petal_', 'p_')} {op} {t:.2f}"
            for f, op, t in r.parsed_conditions)
        transition(ax, 6.1, yy, r.name, w=2.5, h=0.38)
        ax.text(6.1, yy + 0.3, f"[{guard}]", ha="center", fontsize=7.6,
                color=INK_2)
        arrow(ax, (1.6, 1.4), (4.82, yy), color=INK_3)
        arrow(ax, (7.38, yy), (10.6, 1.4), color=INK_3,
              label=f"class {int(r.class_)}", label_offset=(0.2, 0.0))
    # ---- Random forest
    ax = axes[1]
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0, 3.3)
    ax.axis("off")
    ax.text(0.0, 3.25, "Random Forest: a page per tree, then a vote",
            fontsize=11.5, fontweight="bold", va="top")
    place(ax, 0.9, 1.45, "Input")
    transition(ax, 2.5, 1.45, "Distribute", w=1.2)
    arrow(ax, (1.2, 1.45), (1.9, 1.45), color=INK_3)
    trees = [2.4, 1.45, 0.5]
    labels = ["Tree_0", "Tree_1", "Tree_n"]
    for yy, name, k in zip(trees, labels, ("0", "1", "n")):
        place(ax, 4.5, yy, f"TIn_{k}", r=0.26, label_dy=-0.1, size=7.6)
        transition(ax, 6.6, yy, name, w=1.2, sub=True)
        place(ax, 8.7, yy, f"TProb_{k}", r=0.26, label_dy=-0.1, size=7.6)
        arrow(ax, (3.1, 1.45), (4.24, yy), color=INK_3)
        arrow(ax, (4.76, yy), (6.0, yy), color=INK_3)
        arrow(ax, (7.2, yy), (8.44, yy), color=INK_3)
        arrow(ax, (8.96, yy), (10.0, 1.5), color=INK_3)
    ax.text(6.6, 1.0, "...", ha="center", fontsize=14, color=INK_3)
    transition(ax, 10.4, 1.45, "Vote", w=0.9)
    place(ax, 11.8, 1.45, "Prediction", r=0.26, label_dy=-0.1, size=7.6)
    arrow(ax, (10.85, 1.45), (11.54, 1.45), color=INK_3)
    ax.text(12.2, 3.25, "each Tree_i page is a Decision Tree page",
            fontsize=8, color=INK_2, ha="right", va="top")
    # ---- GBDT
    ax = axes[2]
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0, 3.1)
    ax.axis("off")
    ax.text(0.0, 3.05, "GBDT: a page per class, a stage per tree, then a "
            "decision", fontsize=11.5, fontweight="bold", va="top")
    place(ax, 0.8, 1.5, "Input")
    transition(ax, 2.2, 1.5, "Distribute", w=1.1)
    transition(ax, 3.9, 1.5, "Channel_k", w=1.2, sub=True)
    arrow(ax, (1.1, 1.5), (1.65, 1.5), color=INK_3)
    arrow(ax, (2.75, 1.5), (3.3, 1.5), color=INK_3, label="ChIn",
          label_offset=(0, 0.17))
    ax.add_patch(FancyBboxPatch((4.75, 0.55), 5.4, 2.0,
                                boxstyle="round,pad=0.02,rounding_size=0.1",
                                facecolor="none", edgecolor=BLUE, linewidth=1.5,
                                linestyle="--"))
    ax.text(4.9, 2.47, "inside Channel_k: s0 -> s1 -> ... -> sM", fontsize=8,
            color=BLUE, va="top")
    place(ax, 5.2, 1.5, "Acc0", r=0.25, label_dy=-0.1, size=7.6)
    transition(ax, 6.35, 1.5, "Stage_1", w=0.95, sub=True, size=7.4)
    place(ax, 7.5, 1.5, "Acc1", r=0.25, label_dy=-0.1, size=7.6)
    transition(ax, 8.65, 1.5, "Stage_2", w=0.95, sub=True, size=7.4)
    place(ax, 9.8, 1.5, "Acc2", r=0.25, label_dy=-0.1, size=7.6)
    for x1, x2 in ((5.47, 5.8), (6.9, 7.23), (8.05, 8.4), (9.25, 9.53)):
        arrow(ax, (x1, 1.5), (x2, 1.5), color=INK_3)
    place(ax, 10.9, 1.5, "ChScore", r=0.28, label_dy=-0.1, size=7.8)
    arrow(ax, (10.15, 1.5), (10.62, 1.5), color=INK_3)
    transition(ax, 11.55, 2.1, "Decide", w=0.9, size=7.6)
    place(ax, 11.7, 0.7, "Prediction", r=0.26, label_dy=-0.1, size=7.6)
    arrow(ax, (11.1, 1.7), (11.3, 1.95), color=INK_3)
    arrow(ax, (11.6, 1.9), (11.68, 0.97), color=INK_3)
    ax.text(0.0, -0.02, "Names are the ones the exporter writes. A double "
            "border marks a substitution transition: a page lives behind it.",
            fontsize=8.3, color=INK_2, va="top")
    save(fig, "hcpn.png")


# ---------------------------------------------------------------------------
# 7. What the checker prints
# ---------------------------------------------------------------------------

# Function to run the verification on a small boosting model.
def verification():
    """Check a GBDT before and after refinement.

    Returns:
        tuple: ``(report of the refined model, comparison table)`` as text.
    """
    bc = load_breast_cancer(as_frame=True)
    X_tr, X_te, y_tr, y_te = train_test_split(bc.data, bc.target,
                                              test_size=0.3, random_state=0,
                                              stratify=bc.target)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.3,
                                              random_state=0, stratify=y_tr)
    model = PyRuleAnalyzer.new_model("Gradient Boosting Decision Trees",
                                     {"n_estimators": 3, "max_depth": 2,
                                      "random_state": 0})
    model.fit(X_tr, y_tr)
    model.execute_rule_refinement(X=X_va, y=y_va,
                                  remove_below_n_classifications=1,
                                  refine_between_trees=True)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        results = model.model_check(which="both", samples=X_te.iloc[:1],
                                    test_samples=X_te,
                                    base_name=os.path.join(
                                        tempfile.mkdtemp(), "gbdt"))
    return buf.getvalue(), results


# Function to render real output as a terminal.
def render_terminal(lines, name, colored=("FAIL", "PASS", "VERIFIED", "VIOLATION")):
    """Draw ``lines`` as terminal text on a dark background.

    Args:
        lines (list): The text, line by line.
        name (str): File name.
        colored (tuple): Words painted green or orange.
    """
    height = 0.205 * len(lines) + 0.5
    fig = plt.figure(figsize=(11, height))
    fig.patch.set_facecolor("#1a1a19")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_facecolor("#1a1a19")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(lines) + 1)
    for i, ln in enumerate(lines):
        yy = len(lines) - i
        ax.text(0.015, yy, ln, family="DejaVu Sans Mono", fontsize=8.4,
                color="#c3c2b7" if ln.startswith(">>>")
                else "#f0efec", va="center", ha="left")
        for word, c in (("FAIL", "#ff8a5b"), ("PASS", "#3fd39b"),
                        ("VERIFIED", "#3fd39b"), ("VIOLATION", "#ff8a5b")):
            if word in colored and word in ln and not ln.startswith(">>>"):
                col = ln.rindex(word)
                ax.text(0.015, yy, " " * col + word, family="DejaVu Sans Mono",
                        fontsize=8.4, color=c, va="center", ha="left",
                        fontweight="bold")
    save(fig, name)


# Function to draw the checker's report for the refined model.
def terminal() -> None:
    """``model_check`` on a refined GBDT, as printed."""
    text, _ = verification()
    parts = text.split("Model checking: ")
    final = [p for p in parts if p.startswith("gbdt_final")][0]
    final = final.split("Initial vs. refined")[0].rstrip()
    block = ["Model checking: " + ln if i == 0 else ln
             for i, ln in enumerate(final.splitlines())]
    shown = [">>> model.model_check(which='final', test_samples=X_test)", ""]
    shown += [ln[:104] for ln in block]
    render_terminal(shown, "terminal.png")


# Function to draw what the gate says about a broken refinement.
def violation() -> None:
    """A rule loses its last condition: tested inputs pass, B4 fails."""
    data = load_wine(as_frame=True)
    X_tr, X_te, y_tr, y_te = train_test_split(
        data.data, data.target, test_size=0.3, random_state=0,
        stratify=data.target)
    model = PyRuleAnalyzer.new_model("Decision Tree",
                                     {"max_depth": 3, "random_state": 0})
    model.fit(X_tr, y_tr)
    with contextlib.redirect_stdout(io.StringIO()):
        model.execute_rule_refinement(X=X_te, y=y_te,
                                      remove_below_n_classifications=-1)
    rules_ = model.classifier.final_rules
    victim = max(rules_, key=lambda r: len(r.parsed_conditions))
    victim.parsed_conditions = victim.parsed_conditions[:-1]
    victim.conditions = victim.conditions[:-1]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        model.model_check(which="final", samples=X_te.iloc[:1],
                          test_samples=X_te, base_name=os.path.join(
                              tempfile.mkdtemp(), "wine"))
    lines = buf.getvalue().splitlines()
    start = [i for i, ln in enumerate(lines)
             if ln.startswith("Model checking:")][0]
    keep = [ln for ln in lines[start:] if ln.strip()]
    pick = [ln for ln in keep if ln.lstrip().startswith(
        ("B1 ", "B2 ", "B3 ", "B4 ", "-> ", "PC ", "11 passed")) or "VIOLATION" in ln]
    shown = [">>> victim.parsed_conditions = victim.parsed_conditions[:-1]"
             "   # a refinement bug", ">>> model.model_check(which='final')",
             "", "  ...  (A1-A8 and B1-B3 pass on the tested input)", ""]
    shown += [ln[:150] for ln in pick if ln.lstrip().startswith(
        ("B1 ", "B2 ", "B3 ", "B4 ", "->"))]
    shown += [ln[:150] for ln in pick if ln.lstrip().startswith(("PC ",))]
    shown += [""] + [ln for ln in pick if "VIOLATION" in ln]
    # Keep the long detail lines readable.
    shown = [ln if len(ln) < 112 else ln[:109] + "..." for ln in shown]
    render_terminal(shown, "violation.png")


# Function to draw the comparison of the two stages.
def comparison() -> None:
    """The initial-versus-refined table printed by ``which='both'``."""
    text, _ = verification()
    lines = text.splitlines()
    start = [i for i, ln in enumerate(lines)
             if ln.startswith("Initial vs. refined")][0]
    block = [ln for ln in lines[start:] if ln.strip()]
    shown = [">>> model.model_check(which='both')   # the acceptance test", ""]
    shown += [ln[:104] for ln in block]
    render_terminal(shown, "comparison.png", colored=("PASS", "FAIL"))


# ---------------------------------------------------------------------------
# 8. The verified pipeline and its gate
# ---------------------------------------------------------------------------

# Function to draw the verified pipeline.
def gate() -> None:
    """Train, verify, refine, verify again; the second check is the gate."""
    fig, ax = plt.subplots(figsize=(12.5, 4.5))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(-0.5, 4.5)
    ax.axis("off")
    y = 3.1
    top = [
        (1.2, "Train", "scikit-learn model,\nor one you bring"),
        (3.55, "Extract + HCPN", "rules -> Petri net\n(initial)"),
        (5.9, "Check", "baseline: what the\nunrefined net satisfies"),
        (8.25, "Refine", "merge siblings,\ndrop low-usage rules"),
        (10.6, "Check again", "HCPN (refined),\nsame properties"),
    ]
    for x, title, detail in top:
        box(ax, x, y, 2.05, 1.25, title, detail)
    for (x1, *_), (x2, *_) in zip(top, top[1:]):
        arrow(ax, (x1 + 1.07, y), (x2 - 1.07, y), color=BLUE, lw=1.8)
    # The gate.
    gx, gy = 10.6, 1.35
    ax.add_patch(plt.Polygon([(gx, gy + 0.55), (gx + 0.9, gy),
                              (gx, gy - 0.55), (gx - 0.9, gy)],
                             facecolor="white", edgecolor=INK, linewidth=1.8))
    ax.text(gx, gy, "same\nverdicts?", ha="center", va="center", fontsize=8.6,
            fontweight="bold")
    arrow(ax, (10.6, y - 0.66), (10.6, gy + 0.58), color=BLUE, lw=1.8)
    box(ax, 6.6, 1.35, 3.0, 1.0, "Export + deploy",
        ".py  .bin  .h  .ino -> arduino-cli", edge=AQUA)
    arrow(ax, (gx - 0.92, gy), (8.12, gy), color=AQUA, lw=1.8, label="yes",
          label_offset=(0, 0.17))
    box(ax, 10.6, 0.0, 3.2, 0.7, "VerificationError",
        "nothing is exported", edge=ORANGE, title_size=9.5, detail_size=7.8)
    arrow(ax, (gx, gy - 0.57), (gx, 0.38), color=ORANGE, lw=1.8, label="no",
          label_offset=(0.25, 0.0))
    ax.text(0.05, 4.45, "verified_pipeline(): the verification is a gate, "
            "not a report", fontsize=13, fontweight="bold", va="top")
    ax.text(0.05, 0.35, "A refinement can leave accuracy untouched and still "
            "break the model:\nthe check catches what accuracy cannot.",
            fontsize=8.8, color=INK_2, va="center")
    save(fig, "gate.png")


# Function to draw every figure.
def main() -> None:
    """Draw them all."""
    pipeline()
    rules()
    numbers = refinement()
    print("refinement", numbers)
    strategies()
    rows, sizes = benchmark()
    print("throughput", [(r[0], round(r[1])) for r in rows])
    print("sizes", [(s[0], s[1]) for s in sizes])
    hcpn()
    terminal()
    comparison()
    violation()
    gate()


if __name__ == "__main__":
    main()
