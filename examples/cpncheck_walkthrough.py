"""A guided run of CPNCheck, one layer at a time.

The README states what CPNCheck does; this script *shows* it, on the nets under
``tests/cpn_nets``, printing what each layer produced before handing it to the next
one. The order is the order of the pipeline:

    the .cpn file
        -> CPNNet          the XML, as pages, places, transitions      (step 1)
        -> ColouredNet     guards and inscriptions compiled, flattened (step 2)
        -> Profile         which places play which role                (step 3)
        -> StateSpace      the occurrence graph and its SCCs           (step 4)
        -> one run         markings, occurrence by occurrence          (step 5)
        -> CTL             a verdict per property                      (step 6)

and then the four things a verdict is worth only because of:

    a violation is evidenced, and the evidence replays                 (step 7)
    a structural finding names an input, and that input reproduces it  (step 8)
    the net is compared with a reference on inputs it never saw        (step 9)
    the same questions are exported for CPN Tools to answer            (step 10)
    every property can be made to fail                                 (step 11)

Run it::

    python examples/cpncheck_walkthrough.py
    python examples/cpncheck_walkthrough.py --net tests/cpn_nets/rf_multiclass.cpn --labels 0,1,2
    python examples/cpncheck_walkthrough.py --only 4,5,6
    python examples/cpncheck_walkthrough.py --keep build/walkthrough

Nothing here needs anything outside the standard library, and nothing is
written unless ``--keep`` asks for it.
"""

import argparse
import collections
import json
import os
import random
import shutil
import sys
import tempfile
import textwrap

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cpncheck import METHODS                                      # noqa: E402
from pyruleanalyzer.model_checker import CPNModelChecker, check_cpn  # noqa: E402
from cpncheck.conformance import (check_conformance,               # noqa: E402
                                  projection, terminal_marking)
from cpncheck.counterexample import Counterexample                 # noqa: E402
from pyruleanalyzer.cpn_mutants import make_mutants                # noqa: E402

#: Width of the printed report; the report() of cpncheck.result fits in it.
WIDTH = 82

#: The net used when ``--net`` is not given, and its label domain.
DEFAULT_NET = os.path.join(os.path.dirname(__file__), "..", "tests", "cpn_nets",
                           "gbdt_binary.cpn")
DEFAULT_LABELS = [0, 1]

#: Inputs drawn for the conformance step; the seed keeps the run reproducible.
SEED = 20250921
N_SAMPLES = 12


# ---------------------------------------------------------------------------
# Printing
#
# Everything goes through emit(), which collapses runs of blank lines, so a
# step never has to know what the step before it printed last.
# ---------------------------------------------------------------------------

#: Whether the last line printed was blank.
_blank = True


# Function to print one line, collapsing consecutive blank ones.
def emit(text=""):
    """Print a line; a blank line after a blank line is dropped.

    Args:
        text (str): The line.
    """
    global _blank
    if not text.strip():
        if not _blank:
            print()
            _blank = True
        return
    print(text.rstrip())
    _blank = False


# Function to print a rule across the report.
def rule(char="-"):
    """Print a horizontal rule.

    Args:
        char (str): The character to repeat.
    """
    emit(char * WIDTH)


# Function to print the banner of the whole run.
def banner(lines):
    """Print the title block.

    Args:
        lines (list): The lines of the block.
    """
    rule("=")
    for line in lines:
        emit(f"  {line}")
    rule("=")


# Function to print the header of one step.
def step_header(number, total, title, module):
    """Announce a step and the module that does the work.

    Args:
        number (int): Step number.
        total (int): Number of steps in the run.
        title (str): What the step shows.
        module (str): The module responsible.
    """
    emit()
    rule()
    left = f" STEP {number}/{total}  {title}"
    emit(f"{left}{module:>{max(1, WIDTH - len(left))}}")
    rule()


# Function to print the sentence explaining why a step exists.
def why(text):
    """Print the rationale of a step, wrapped to the report width.

    Args:
        text (str): The rationale.
    """
    wrapped(text, indent=2, hang=0)
    emit()


# Function to print one labelled value.
def kv(key, value, indent=2):
    """Print ``key : value`` in the report's column layout.

    Args:
        key (str): The label.
        value: The value.
        indent (int): Leading spaces.
    """
    emit(f"{' ' * indent}{key:<18}{value}")


# Function to print a bullet.
def item(text, indent=6):
    """Print a bullet line.

    Args:
        text (str): The text.
        indent (int): Leading spaces.
    """
    emit(f"{' ' * indent}{text}")


# Function to print a line of text, wrapped to the report width.
def wrapped(text, indent=6, hang=4):
    """Print text folded to the report width, with a hanging indent.

    Args:
        text (str): The text.
        indent (int): Leading spaces of the first line.
        hang (int): Extra leading spaces of the continuation lines.
    """
    for line in textwrap.wrap(str(text), WIDTH - indent,
                              subsequent_indent=" " * hang):
        emit(f"{' ' * indent}{line}")


# Function to print a sub-heading inside a step.
def block(title):
    """Print a labelled block inside a step.

    Args:
        title (str): The heading.
    """
    emit()
    emit(f"  {title}")
    emit("  " + "." * (WIDTH - 4))


# Function to shorten a value to one line.
def short(value, width=52):
    """Render a token or input compactly.

    Args:
        value: The value.
        width (int): Maximum length.

    Returns:
        str: The rendering, truncated with an ellipsis when too long.
    """
    if isinstance(value, tuple):
        text = "(" + ", ".join(short(v, width) for v in value) + ")"
    elif isinstance(value, float):
        text = f"{value:g}"
    else:
        text = str(value)
    return text if len(text) <= width else text[:width - 3] + "..."


# Function to render the marked places of a marking.
def marking_line(cnet, marking, width=48):
    """The non-empty places of a marking, as ``Place=tokens``.

    A marking is a tuple indexed by place, so the empty places carry no
    information and are left out.

    Args:
        cnet (ColouredNet): The flattened net.
        marking (tuple): The marking.
        width (int): Maximum length of the rendering.

    Returns:
        str: The marked places, or ``(empty)``.
    """
    parts = []
    for i, tokens in enumerate(marking):
        if not tokens:
            continue
        name = cnet.places[i]["name"]
        if len(tokens) != 1:
            parts.append(f"{name}={len(tokens)} tokens")
            continue
        # A record, or any token too wide for the column, says nothing useful
        # half-printed; that a token is *there* is what the table is for.
        text = short(tokens[0], 10 ** 6)
        parts.append(f"{name}={text if len(text) <= 16 else '<...>'}")
    return short(", ".join(parts), width) if parts else "(empty)"


# Function to render a path the shorter of the two ways.
def where(path):
    """A path relative to the working directory, when that is the shorter form.

    Args:
        path (str): The path.

    Returns:
        str: The rendering, truncated to the report width.
    """
    rel = os.path.relpath(path)
    return short(rel if not rel.startswith("..") else path, WIDTH - 22)


# Function to render the arcs of a transition as the file writes them.
def arcs(raw):
    """Arcs as ``place <inscription>``, from their unflattened form.

    Args:
        raw (list): ``(place gid, inscription)`` pairs.

    Returns:
        str: The arcs, or ``(none)``.
    """
    return ", ".join(f"{gid.rsplit('/', 1)[-1]} <{expr}>"
                     for gid, expr in raw) or "(none)"


# ---------------------------------------------------------------------------
# The steps
# ---------------------------------------------------------------------------

# Function to show what the XML parser read.
def step_parse(ctx):
    """Step 1: the ``.cpn`` file as pages, places and transitions.

    Args:
        ctx (dict): The shared context of the run.
    """
    why("CPNCheck reads the file CPN Tools itself would open. Nothing is "
        "taken from whatever generated the net, so a disagreement between "
        "the net and its generator is a real disagreement.")
    net, cnet = ctx["mc"].net, ctx["mc"].cnet
    kv("file", where(ctx["path"]))
    kv("top page", cnet.top["name"])
    kv("pages", f"{len(net.pages)}   " + ", ".join(net.pages))
    kv("places", f"{len(cnet.places)} (after flattening the hierarchy)")
    kv("transitions", len(cnet.transitions))

    block("the place that carries the input, as the file declares it")
    place = cnet.places[ctx["mc"].input_places[0]]
    item(f"name      {place['name']}   (page {place['page']})")
    item(f"initmark  {short(place['initmark'], 60)}")

    block("a transition, as the file declares it")
    t = cnet.transitions[ctx["leaf"]]
    item(f"name      {t['name']}   (page {t['page']})")
    item(f"guard     {short(t['guard_text'], 60)}")
    item("in arcs   " + arcs(t["in_raw"]))
    item("out arcs  " + arcs(t["out_raw"]))


# Function to show what the CPN ML front end made of the inscriptions.
def step_compile(ctx):
    """Step 2: guards and inscriptions compiled into closures.

    Args:
        ctx (dict): The shared context of the run.
    """
    why("Guards and arc inscriptions are parsed as CPN ML, not pattern "
        "matched, and compiled to closures, so firing stays fast. The net's "
        "own val and fun declarations are read and used: a net describes "
        "itself, and nothing about what it computes is built into the "
        "checker.")
    cnet = ctx["mc"].cnet
    decl = cnet.declarations
    kv("colour sets", ", ".join(decl.colsets))
    kv("variables", ", ".join(f"{v}:{cs}" for v, cs in decl.variables.items()))
    kv("record fields", ", ".join(decl.fields) if decl.fields else "(none)")

    block("declarations the net makes in its own <ml> blocks")
    shown = 0
    for element in ctx["mc"].net.cpnet.iter("ml"):
        item(short(" ".join((element.text or "").split()), 66))
        shown += 1
        if shown == 8:
            item("...")
            break
    if not shown:
        item("(this net declares none; pass prelude={...} to supply them)")

    block("the guards of one tree, compiled, on the file's own input")
    sample = ctx["mc"].own_sample
    item(f"input   {short(sample, 62)}", indent=4)
    emit()
    page, info = next(iter(ctx["mc"].structure.trees.items()))
    for ti in info["leaves"]:
        t = cnet.transitions[ti]
        holds = t["guard"]({"x": sample}) if t["guard_text"] else True
        item(f"{'-> TRUE ' if holds else '   false':<10}"
             f"{short(t['guard_text'] or '(no guard)', WIDTH - 20)}", indent=4)
    item(f"exactly one of the {len(info['leaves'])} leaves of {page} holds -- "
         f"which is", indent=4)
    item("property B1, and the structural check of step 8 proves it for "
         "every input", indent=4)

    block("the binding elements enabled in the initial marking")
    for ti, _key, env in cnet.enabled(cnet.initial):
        bind = ", ".join(f"{k} = {short(v, 28)}" for k, v in sorted(env.items()))
        item(f"{cnet.transitions[ti]['name']:<24}{bind}")


# Function to show which profile recognised the net and what it names.
def step_profile(ctx):
    """Step 3: the profile that gives the net's places their meaning.

    Args:
        ctx (dict): The shared context of the run.
    """
    why("The checker knows only markings, arcs and formulas. Which places and "
        "transitions play which role, and which properties apply, come from a "
        "profile -- so verifying your own nets is writing a profile, not "
        "patching the checker.")
    mc = ctx["mc"]
    kv("profile", type(mc.profile).__name__ + f"  ({mc.profile.name})")
    kv("family", mc.family)
    kv("label domain", f"{mc.labels}   (source: {mc.label_source})")
    st = mc.structure
    kv("input place", mc.cnet.places[st.input]["name"])
    kv("output place", mc.cnet.places[st.pred]["name"])

    block("groups the profile resolved")
    for name, members in mc.groups.items():
        item(f"{name:<12}{len(members)}  " + short(", ".join(map(str, members)), 50))

    block("the properties the profile selected for this net, by technique")
    selected = ctx["result"].properties
    for method, title in METHODS.items():
        ids = [pid for pid, r in selected.items() if r.spec.method == method]
        item(f"{method:<12}"
             + (", ".join(ids) if ids
                else "PC -- only when a reference is given; see step 9"))

    block("in the profile's catalog, but not for this family")
    others = [pid for pid in mc.catalog if pid not in selected and pid != "PC"]
    item(", ".join(others) if others else "(none: this family uses them all)")


# Function to show the occurrence graph and its SCC decomposition.
def step_state_space(ctx):
    """Step 4: every reachable marking, and what the SCCs say about them.

    Args:
        ctx (dict): The shared context of the run.
    """
    why("The occurrence graph is every marking reachable from the initial "
        "one, with real token values and no abstraction. Its SCC "
        "decomposition answers the questions CPN Tools answers with "
        "ListDeadMarkings and ListHomeMarkings: a home marking exists exactly "
        "when there is a single terminal component.")
    ss = ctx["ss"]
    kv("occurrence graph", f"{ss.n_nodes} nodes, {ss.n_arcs} arcs"
                           f"{'' if ss.complete else '  (INCOMPLETE)'}")
    kv("SCC graph", f"{ss.n_sccs} nodes, {ss.n_scc_arcs} arcs")
    kv("dead markings", f"{len(ss.dead)}  {ss.dead}")
    kv("home markings", f"{len(ss.home)}  {ss.home}")
    kv("built in", f"{ss.elapsed:.3f}s")

    block("the shortest occurrence sequence to the dead marking")
    for line in textwrap.wrap(ss.trace(ss.dead[0], limit=99), WIDTH - 8,
                              subsequent_indent="   "):
        item(line, indent=4)


# Function to replay one run occurrence by occurrence.
def step_firing(ctx):
    """Step 5: a single run, marking by marking.

    Args:
        ctx (dict): The shared context of the run.
    """
    why("This is the firing rule the whole checker rests on, unrolled: at "
        "each marking, the enabled binding elements are enumerated, one "
        "occurs, and tokens move. A counterexample is nothing but such a "
        "sequence, recorded.")
    mc, ss = ctx["mc"], ctx["ss"]
    target = ss.dead[0]
    emit(f"  {'#':<4}{'transition':<26}"
          f"{'marking after it   (<...> is a record token)'}")
    emit("  " + "-" * (WIDTH - 4))
    emit(f"  {'0':<4}{'(initial marking)':<26}"
          f"{marking_line(mc.cnet, ss.markings[0])}")
    for n, (ti, _key, dst) in enumerate(ss.path(target), start=1):
        name = mc.cnet.transitions[ti]["name"]
        emit(f"  {n:<4}{short(name, 24):<26}"
              f"{marking_line(mc.cnet, ss.markings[dst])}")
    emit()
    kv("ends in", f"node {target}: nothing is enabled, the net has stopped")
    answer = projection(ss.markings[target], (mc.structure.pred,))
    kv("its answer", f"{mc.cnet.places[mc.structure.pred]['name']} holds "
                     f"{', '.join(short(v) for v in answer) or 'nothing'}")


# Function to print the verdicts of the correct net.
def step_verdicts(ctx):
    """Step 6: the report -- one verdict per property, grouped by technique.

    Args:
        ctx (dict): The shared context of the run.
    """
    why("Each formula is evaluated at the initial marking by the CTL "
        "fixpoints of cpncheck.ctl. Dead markings are given a self-loop, so a "
        "maximal path that ends before phi really is a counterexample to "
        "AF phi -- the textbook reading, which plain ASK-CTL EV does not "
        "give.")
    for line in ctx["result"].report().splitlines():
        emit(line)


# Function to show that a violation comes with evidence that replays.
def step_counterexamples(ctx):
    """Step 7: mutants, their counterexamples, and the replay certificate.

    Args:
        ctx (dict): The shared context of the run.
    """
    why("Agreement on correct nets alone would also be reached by two "
        "checkers that always answer true. So each property gets a mutant "
        "built to falsify it -- and each violation hands back evidence: the "
        "occurrence sequence, the binding of every step, and the marking it "
        "ends in.")
    labels = ctx["labels"]
    shown = [("cycle", "A1", "a run that never terminates: no finite witness "
                             "exists, so the evidence is a lasso"),
             ("deadlock", "A3", "a finite sequence into a marking where the "
                                "prediction can no longer happen"),
             ("bad_label", "A6", "the net answers, but with a value outside "
                                 "the declared label domain")]
    for name, pid, note in shown:
        mutant = ctx["mutants"].get(name)
        if mutant is None:
            continue
        mc = CPNModelChecker(mutant["path"], class_labels=mutant["labels"] or labels)
        r = mc.check().properties[pid]
        block(f"{name}  ({mutant['description']})")
        wrapped(note, indent=4, hang=0)
        emit()
        item(f"{pid:<10}{r.status}   {mc.catalog[pid].name} -- "
             f"{mc.catalog[pid].ctl}")
        item(f"{'kind':<10}{r.cex.kind}")
        wrapped(f"{'says':<10}{r.cex.note}", indent=6, hang=10)
        item(f"{'length':<10}{r.cex.length} occurrence(s)")
        wrapped(f"{'sequence':<10}{r.cex.sequence(limit=8)}",
                indent=6, hang=10)
        item(f"{'replays':<10}{r.cex.replay(mc.cnet)}"
             "   <- re-fired with the net's own firing rule")

    block("the certificate has to be able to fail, or it certifies nothing")
    mutant = ctx["mutants"]["deadlock"]
    mc = CPNModelChecker(mutant["path"], class_labels=mutant["labels"] or labels)
    cex = next(r.cex for r in mc.check().failures if r.cex.kind == "finite")
    tampered = [
        ("as reported", cex),
        ("with its steps reversed", Counterexample(
            cex.kind, note=cex.note, prefix=list(reversed(cex.prefix)),
            final=cex.final, input=cex.input, data=cex.data)),
        ("with the last step removed", Counterexample(
            cex.kind, note=cex.note, prefix=cex.prefix[:-1], final=cex.final,
            input=cex.input, data=cex.data)),
    ]
    for label, candidate in tampered:
        item(f"{label:<30}replay() -> {candidate.replay(mc.cnet)}")


# Function to show a structural finding turned back into a run.
def step_structural(ctx):
    """Step 8: a finding true of every input, and the input that shows it.

    Args:
        ctx (dict): The shared context of the run.
    """
    why("Guard disjointness holds or fails for every input, which no single "
        "state space can show, so a failure names no particular run. But the "
        "intersection of two guards that are conjunctions of bounds is a box, "
        "and a point in it is an input on which the overlap really happens. "
        "Fed back, the per-input property must fail too -- on the same pair.")
    mutant = ctx["mutants"].get("hidden_overlap")
    if mutant is None:
        item("(this net has no overlap mutant)")
        return
    labels = mutant["labels"] or ctx["labels"]

    block("1. the structural check, over every input at once")
    first = check_cpn(mutant["path"], class_labels=labels)
    wrapped(f"B4  {first.properties['B4'].status}   "
            f"{first.properties['B4'].cex.note}", indent=4, hang=11)
    wrapped(f"B1  {first.properties['B1'].status}   "
            f"<- no overlap on the input the file happens to carry",
            indent=4, hang=11)

    witness = first.properties["B4"].cex.input
    block("2. the witness the structural check extracted")
    item(short(witness, WIDTH - 8), indent=4)

    block("3. the same net, given that witness as its input")
    again = check_cpn(mutant["path"], samples=[witness], class_labels=labels)
    wrapped(f"B1  {again.properties['B1'].status}   "
            f"{again.properties['B1'].cex.note}", indent=4, hang=11)
    mc = CPNModelChecker(mutant["path"], class_labels=labels)
    item(f"    replays -> {again.properties['B1'].cex.replay(mc.cnet)}", indent=4)


# Function to compare a net with a reference implementation.
def step_conformance(ctx):
    """Step 9: does the net compute the right thing, on inputs it never saw?

    The reference here is the correct net itself, run to its terminal marking:
    it needs no dependency, and it makes the point exactly -- the temporal
    properties say how a net behaves, never that its answer is right.

    Args:
        ctx (dict): The shared context of the run.
    """
    why("None of the temporal properties says the answer is correct; that is "
        "a different question, answered by running the net on concrete inputs "
        "and comparing. One maximal occurrence sequence is followed per "
        "input, which is cheap -- and that this single run gives *the* answer "
        "of the net is what the model-checking step established.")
    good = ctx["mc"]
    rng = random.Random(SEED)
    inputs = [tuple(round(rng.uniform(-2.5, 2.5), 4) for _ in range(good.width))
              for _ in range(N_SAMPLES)]

    # The reference: the verified net, run to the marking where nothing is
    # enabled, and read out of its output place.
    def reference(batch):
        """Label every input by running the correct net on it.

        Args:
            batch (list): The inputs.

        Returns:
            list: One label per input.
        """
        out = []
        for x in batch:
            marking = terminal_marking(good.cnet, x, good.input_places)
            tokens = projection(marking, (good.structure.pred,))
            out.append(int(tokens[0]) if len(tokens) == 1 else None)
        return out

    kv("inputs", f"{N_SAMPLES} drawn with seed {SEED}, none of them the "
                 f"file's own")
    kv("reference", "the verified net itself, run to its terminal marking")
    kv("expected", reference(inputs))

    for name, path in [("the verified net", ctx["path"]),
                       ("mutant wrong_value",
                        ctx["mutants"]["wrong_value"]["path"])]:
        mc = CPNModelChecker(path, class_labels=ctx["labels"])
        r = check_conformance(mc.cnet, inputs, reference,
                              input_places=(mc.structure.input,),
                              output_places=(mc.structure.pred,),
                              pid="PC", spec=mc.catalog.get("PC"))
        block(name)
        wrapped(f"PC  {r.status}   {r.detail}", indent=4, hang=11)


# Function to show the artefacts a run can write.
def step_exports(ctx):
    """Step 10: the same questions, for CPN Tools, for CI and for a paper.

    Args:
        ctx (dict): The shared context of the run.
    """
    why("A verdict is only worth what can be done with it. The ASK-CTL script "
        "asks CPN Tools the same questions -- generated from the same "
        "formulas, so the two checks are the same question by construction -- "
        "and cpncheck oracle runs CPN Tools headless and compares every "
        "statistic and every verdict.")
    out = ctx["out_dir"]
    script = ctx["mc"].export_askctl(os.path.join(out, "askctl.sml"))

    block("the ASK-CTL script, as CPN Tools would be given it")
    with open(script, encoding="utf-8") as fh:
        lines = [line.rstrip() for line in fh]
    head = [line for line in lines if line.strip()][:14]
    for line in head:
        item(short(line, WIDTH - 8), indent=4)
    item(f"... {len(lines) - len(head)} more lines", indent=4)
    emit()
    kv("written to", where(script))

    payload = os.path.join(out, "verdicts.json")
    with open(payload, "w", encoding="utf-8") as fh:
        json.dump(ctx["result"].to_dict(), fh, indent=2)
    kv("verdicts, JSON", where(payload))

    block("the same verdicts as LaTeX rows (cpncheck check --latex)")
    for line in ctx["result"].latex_rows().splitlines()[:3]:
        item(short(line, WIDTH - 8), indent=4)
    item("...", indent=4)

    block("and, where CPN Tools 4.0.1 is installed")
    item("cpncheck oracle " + where(ctx["path"]), indent=4)
    item("-> every statistic and every verdict, ours beside theirs", indent=4)


# Function to run every mutant and tabulate the outcome.
def step_sensitivity(ctx):
    """Step 11: every property, made to fail by a mutant built for it.

    Args:
        ctx (dict): The shared context of the run.
    """
    why("One mutant per property, each a small edit of a guard, an "
        "inscription or an arc that makes exactly that property false. The "
        "test suite requires the targeted property to fail on every one of "
        "them; this is that run, printed.")
    emit(f"  {'mutant':<17}{'targets':<12}{'verdict':<12}{'outcome'}")
    emit("  " + "-" * (WIDTH - 4))
    ok = True
    for name, mutant in ctx["mutants"].items():
        result = check_cpn(mutant["path"],
                           class_labels=mutant["labels"] or ctx["labels"])
        targets = [pid for pid in mutant["targets"] if pid != "PC"]
        if not targets:                      # PC needs the reference; step 9
            emit(f"  {name:<17}{','.join(mutant['targets']):<12}"
                  f"{'--':<12}shown in step 9")
            continue
        verdicts = [result.properties[pid].status for pid in targets]
        good = all(v == "FAIL" for v in verdicts)
        ok = ok and good
        emit(f"  {name:<17}{','.join(targets):<12}"
              f"{','.join(verdicts):<12}{'as designed' if good else 'MISSED'}")
    emit()
    kv("outcome", "every mutant falsified the property it was built to "
                  "violate" if ok else "a mutant went undetected")


#: The run, in order: ``(title, module, function)``.
STEPS = [
    ("Reading the .cpn file", "cpncheck.net", step_parse),
    ("Compiling the CPN ML", "cpncheck.coloured", step_compile),
    ("Recognising the net", "cpncheck.profile", step_profile),
    ("Building the occurrence graph", "cpncheck.statespace", step_state_space),
    ("One run, occurrence by occurrence", "cpncheck.coloured", step_firing),
    ("Verifying the properties", "cpncheck.ctl", step_verdicts),
    ("Evidence, and its certificate", "cpncheck.counterexample",
     step_counterexamples),
    ("A finding about every input", "cpncheck.structural", step_structural),
    ("Conformance with a reference", "cpncheck.conformance", step_conformance),
    ("Exporting the same questions", "cpncheck.askctl", step_exports),
    ("Sensitivity: every property can fail", "cpncheck.mutation",
     step_sensitivity),
]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

# Function to parse a comma-separated list of integers.
def parse_labels(text):
    """Parse ``--labels 0,1,2``.

    Args:
        text (str): The option value, or None.

    Returns:
        list|None: The labels.
    """
    return None if not text else [int(v) for v in text.split(",")]


# Function to parse a comma-separated list of step numbers.
def parse_only(text):
    """Parse ``--only 4,5,6``.

    Args:
        text (str): The option value, or None.

    Returns:
        set|None: Step numbers, one-based.
    """
    return None if not text else {int(v) for v in text.split(",")}


# Function to run the walkthrough.
def main(argv=None):
    """Run the guided walkthrough.

    Args:
        argv (list, optional): Arguments; ``sys.argv[1:]`` when omitted.

    Returns:
        int: Process exit status.
    """
    parser = argparse.ArgumentParser(
        description="A guided run of CPNCheck, one layer at a time.")
    parser.add_argument("--net", default=os.path.normpath(DEFAULT_NET),
                        help="the .cpn file to walk through")
    parser.add_argument("--labels", help="label domain, e.g. 0,1,2")
    parser.add_argument("--only", help="run only these steps, e.g. 4,5,6")
    parser.add_argument("--keep", metavar="DIR",
                        help="keep the mutants and the exports in DIR")
    args = parser.parse_args(argv)

    labels = parse_labels(args.labels)
    if labels is None and os.path.abspath(args.net) == os.path.abspath(
            DEFAULT_NET):
        labels = DEFAULT_LABELS
    wanted = parse_only(args.only)

    out_dir = args.keep or tempfile.mkdtemp(prefix="cpncheck-walkthrough-")
    os.makedirs(out_dir, exist_ok=True)
    try:
        mc = CPNModelChecker(args.net, class_labels=labels)
        # Steps 3, 8 and 9 read the profile's structure, and 7, 8 and 11 its
        # mutation operators; a profile that supplies neither still gets the
        # layers that only need the net itself.
        trees = getattr(mc.structure, "trees", None) or {}
        leaves = next(iter(trees.values()))["leaves"] if trees else None
        if leaves is None:
            leaves = [i for i, t in enumerate(mc.cnet.transitions)
                      if t["guard_text"]] or [0]
        ctx = {"path": args.net, "labels": labels, "mc": mc,
               "out_dir": out_dir, "leaf": leaves[0],
               "ss": mc.state_space(),
               "result": mc.check(),
               "mutants": collections.OrderedDict(
                   (m["name"], m) for m in make_mutants(args.net, out_dir))}

        banner([
            "CPNCheck -- a guided run",
            "",
            f"net      {where(args.net)}",
            f"family   {mc.family}   (profile: {mc.profile.name})",
            f"labels   {mc.labels}",
            f"workdir  {'' if args.keep else '(temporary)  '}"
            f"{short(where(out_dir), WIDTH - 36)}",
        ])
        for number, (title, module, func) in enumerate(STEPS, start=1):
            if wanted and number not in wanted:
                continue
            step_header(number, len(STEPS), title, module)
            try:
                func(ctx)
            except Exception as exc:                 # a profile without a hook
                emit()
                wrapped(f"[skipped] this step needs something the profile does "
                        f"not supply: {exc}", indent=2, hang=2)

        emit()
        banner([f"{'VERIFIED' if ctx['result'].passed else 'VIOLATION FOUND'}"
                f"  --  {len(ctx['result'].properties)} properties, "
                f"{len(ctx['mutants'])} mutants"])
        return 0 if ctx["result"].passed else 1
    finally:
        if not args.keep:
            shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
