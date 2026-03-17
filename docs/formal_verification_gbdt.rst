GBDT
====

In Gradient Boosting, classification is performed by sequential accumulation of real-valued scores. The implementation follows scikit-learn's One-vs-Rest (OVR) scheme: for a problem with :math:`|C|` classes, there are :math:`|C|` independent score channels.

HCPN Formalization
-------------------

The GBDT model is represented as a 3-level HCPN:

.. math::

   HCPN_{GBDT} = (S_{GBDT}, \; SM_{GBDT}, \; PS_{GBDT}, \; FS_{GBDT})

**Modules** :math:`S_{GBDT} = \{s_{GBDT}, \; s_{channel}, \; s_{DT}\}`:

* :math:`s_{GBDT}` — the **top-level module** with class-channel substitution transitions and the activation transition.
* :math:`s_{channel}` — the **mid-level module** representing a single class channel's sequential boosting loop.
* :math:`s_{DT}` — the **leaf module** (see :doc:`formal_verification_dt`), reused :math:`|C| \times K` times (once per boosting stage per class).

Top-Level Module :math:`s_{GBDT}`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::

   s_{GBDT} = (CPN_{GBDT}, \; T_{sub} = \{{t_{ch}}^{(1)}, \dots, {t_{ch}}^{(|C|)}\}, \; P_{port} = \{P_{in}, P_{out}\}, \; PT)

with:

* :math:`P = \{P_{in}, \; {P_{score}}^{(1)}, \dots, {P_{score}}^{(|C|)}, \; P_{out}\}`.
* :math:`C(P_{in}) = \text{VEC}`, :math:`C({P_{score}}^{(c)}) = \text{SCORE}`, :math:`C(P_{out}) = \text{LABEL}`.
* :math:`T = \{{t_{ch}}^{(1)}, \dots, {t_{ch}}^{(|C|)}, \; t_{act}\}` — :math:`|C|` substitution transitions (one per class channel) plus one regular activation transition.

**Submodule function:**

.. math::

   SM_{GBDT}({t_{ch}}^{(c)}) = s_{channel} \quad \text{for } c = 1, \dots, |C|

**Port-socket bindings** for each :math:`{t_{ch}}^{(c)}`:

* :math:`P_{in}` (In port of :math:`s_{channel}`) :math:`\leftrightarrow` :math:`P_{in}` (socket in :math:`s_{GBDT}`).
* :math:`P_{result}` (Out port of :math:`s_{channel}`) :math:`\leftrightarrow` :math:`{P_{score}}^{(c)}` (socket in :math:`s_{GBDT}`).

Mid-Level Module :math:`s_{channel}` (Boosting Loop)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each class channel is a module that sequentially accumulates contributions from :math:`K` boosting stages:

.. math::

   s_{channel} = (CPN_{ch}, \; T_{sub} = \{{t_{stage}}^{(1)}, \dots, {t_{stage}}^{(K)}\}, \; P_{port} = \{P_{in}, P_{result}\}, \; PT)

with:

* :math:`P = \{P_{in}, \; P_{accum}, \; P_{stage}, \; P_{result}\}`.
* :math:`C(P_{in}) = \text{VEC}`, :math:`C(P_{accum}) = \text{SCORE}`, :math:`C(P_{stage}) = \text{STAGE}`, :math:`C(P_{result}) = \text{SCORE}`.
* :math:`PT(P_{in}) = In`, :math:`PT(P_{result}) = Out`.
* :math:`T = \{{t_{stage}}^{(1)}, \dots, {t_{stage}}^{(K)}, \; t_{finalize}\}` — :math:`K` substitution transitions (one per boosting stage) plus a finalize transition.

**Initial marking:**

* :math:`I(P_{accum}) = \{s_0\}` — the initial score (bias) derived from scikit-learn's prior estimator (``DummyClassifier``):

.. math::

   {s_0}^{(\text{binary})} = \ln\!\frac{p}{1-p}, \qquad {s_{0,c}}^{(\text{multi})} = \ln p_c - \frac{1}{|C|}\sum_j \ln p_j

* :math:`I(P_{stage}) = \{1\}` — stage counter starts at 1.

**Submodule function:**

.. math::

   SM_{ch}({t_{stage}}^{(k)}) = s_{DT} \quad \text{for } k = 1, \dots, K

Each substitution transition :math:`{t_{stage}}^{(k)}` is refined into a DT leaf module containing the rules of the :math:`k`-th boosting tree for this class channel.

**Port-socket bindings** for each :math:`{t_{stage}}^{(k)}`:

* :math:`P_{in}` (In port of :math:`s_{DT}`) :math:`\leftrightarrow` :math:`P_{in}` (socket in :math:`s_{channel}`).
* :math:`P_{out}` (Out port of :math:`s_{DT}`) :math:`\leftrightarrow` :math:`P_{accum}` (socket in :math:`s_{channel}`).

**Sequential accumulation logic:**

The stage counter :math:`\kappa \in P_{stage}` enforces sequential firing. For each stage :math:`k`:

* **Guard on** :math:`{t_{stage}}^{(k)}`: :math:`[\kappa = k]` — the substitution transition is only enabled at the correct stage.
* After the DT sub-module fires (exactly one rule matches, producing :math:`\eta \cdot {v_k}^{(c)}`), the channel module's arc expressions update:

  - :math:`P_{accum} \leftarrow s + \eta \cdot {v_k}^{(c)}` (accumulated score).
  - :math:`P_{stage} \leftarrow \kappa + 1` (advance counter).

**Finalize transition** :math:`t_{finalize}`:

Guarded by :math:`[\kappa = K{+}1]`, this regular transition consumes the final accumulated score from :math:`P_{accum}` and deposits it in the output port :math:`P_{result}`.

.. note::

   Within each boosting stage, the DT sub-module's internal mutual exclusivity guarantees that exactly one leaf fires. The stage counter ensures strict sequential ordering across stages, so at most one substitution transition is enabled at any time.

Activation Transition :math:`t_{act}`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After all :math:`|C|` channel modules complete (each depositing a terminal score in :math:`{P_{score}}^{(c)}`), the activation transition :math:`t_{act}` in the top-level module :math:`s_{GBDT}` fires. It is a **synchronization transition** with :math:`|C|` input arcs — one from each :math:`{P_{score}}^{(c)}` — collecting the terminal scores :math:`{s_K}^{(1)}, {s_K}^{(2)}, \dots, {s_K}^{(|C|)}` in a single atomic step. The activation is a two-step process:

1. **Probability computation:**

.. math::

   p^{(\text{binary})} = \sigma(s_K) = \frac{1}{1 + e^{-s_K}}, \qquad {\mathbf{p}_c}^{(\text{multi})} = \frac{e^{{s_K}^{(c)}}}{\sum_j e^{{s_K}^{(j)}}} \;\text{(softmax)}

2. **Decision:**

.. math::

   \hat{y} = \begin{cases}
   1 & \text{if } p^{(\text{binary})} \geq 0.5 \quad \text{(binary)} \\[4pt]
   \arg\max_c \; {s_K}^{(c)} & \text{(multiclass)}
   \end{cases}

.. note::

   In the multiclass case, because softmax is a monotonic transformation, :math:`\arg\max` can be applied directly to the raw scores :math:`{s_K}^{(c)}` without computing the softmax probabilities, which is what the implementation does.

**Fusion set** :math:`FS_{GBDT}`:

.. math::

   FS_{GBDT} = \{\{P_{in}\}\}

The input place :math:`P_{in}` is fused across the top module, all channel sub-modules, and all DT sub-module instances. Every tree in every channel evaluates the same input vector :math:`x`.

**Summary of the OVR score accumulation:**

.. math::

   {s_K}^{(c)} = {s_0}^{(c)} + \sum_{k=1}^{K} \eta \cdot {v_k}^{(c)}

where :math:`K` is the number of boosting stages, :math:`\eta` is the learning rate, and :math:`{v_k}^{(c)}` is the leaf value of the matched rule in tree :math:`k` for class :math:`c`.

The figure below depicts the additive composition for a single class channel. The place :math:`P_{initial\_score}` holds the bias token :math:`s_0`. Inside the additive boosting cycle, each gradient step transition :math:`T_{tree\_k}` adds :math:`\eta \cdot v_k` to the accumulator. After the final iteration, the activation transition :math:`T_{activation}` (sigmoid or softmax) converts the raw score into a class probability and deposits the result in :math:`P_{final\_prediction}`. For multiclass problems, :math:`|C|` such channels operate in parallel, and the activation transition collects all channel scores before producing the final label.

.. figure:: _static/images/gbdt_to_cpn.png
   :align: center
   :width: 85%
   :alt: GBDT HCPN diagram

   HCPN for GBDT (single class channel view): initial bias score, sequential accumulation via substitution transitions (each refining to a DT module), and a final activation transition (sigmoid/softmax).


Numerical Induction Example
-----------------------------

The following example demonstrates that the HCPN construction correctly accumulates scores for a channel with :math:`K` boosting stages and remains valid when extended to :math:`K+1` stages. A binary classification problem is used for clarity (single channel, sigmoid activation).

**Setup:** Feature space :math:`(x_1, x_2)`, classes :math:`\{0, 1\}`, learning rate :math:`\eta = 0.1`. Training set has 60% positive samples, so the initial score is:

.. math::

   s_0 = \ln\frac{0.6}{0.4} = \ln 1.5 \approx 0.4055

**Base case** :math:`(K = 2)`:

Two boosting stages, each with a small decision tree:

.. list-table::
   :header-rows: 1
   :widths: 10 10 35 15

   * - Stage
     - Rule
     - Guard
     - Leaf value :math:`v`
   * - :math:`k = 1`
     - :math:`r_{1a}`
     - :math:`x_1 \leq 3.0`
     - :math:`+2.4`
   * - :math:`k = 1`
     - :math:`r_{1b}`
     - :math:`x_1 > 3.0`
     - :math:`-1.8`
   * - :math:`k = 2`
     - :math:`r_{2a}`
     - :math:`x_2 \leq 5.0`
     - :math:`+1.5`
   * - :math:`k = 2`
     - :math:`r_{2b}`
     - :math:`x_2 > 5.0`
     - :math:`-0.9`

**HCPN state trace for input** :math:`x = (2.0, \; 7.0)`:

*Initial state:* :math:`P_{accum} = 0.4055`, :math:`P_{stage} = 1`.

*Stage 1* (:math:`\kappa = 1`):

* Guard :math:`[\kappa = 1]` is true, so :math:`{t_{stage}}^{(1)}` is enabled.
* DT sub-module fires: :math:`G(r_{1a}) = [2.0 \leq 3.0] = \text{true}`, matched leaf value :math:`v_1 = +2.4`.
* :math:`P_{accum} \leftarrow 0.4055 + 0.1 \times 2.4 = 0.4055 + 0.24 = 0.6455`.
* :math:`P_{stage} \leftarrow 2`.

*Stage 2* (:math:`\kappa = 2`):

* Guard :math:`[\kappa = 2]` is true, so :math:`{t_{stage}}^{(2)}` is enabled.
* DT sub-module fires: :math:`G(r_{2b}) = [7.0 > 5.0] = \text{true}`, matched leaf value :math:`v_2 = -0.9`.
* :math:`P_{accum} \leftarrow 0.6455 + 0.1 \times (-0.9) = 0.6455 - 0.09 = 0.5555`.
* :math:`P_{stage} \leftarrow 3`.

*Finalize* (:math:`\kappa = 3 = K{+}1`):

* :math:`t_{finalize}` fires, deposits :math:`s_2 = 0.5555` in :math:`P_{result}`.

*Activation:*

.. math::

   p = \sigma(0.5555) = \frac{1}{1 + e^{-0.5555}} = \frac{1}{1 + 0.5736} \approx 0.6353

.. math::

   \hat{y} = 1 \quad (\text{since } 0.6353 \geq 0.5)

Accumulated score matches the formula:

.. math::

   s_2 = s_0 + \eta v_1 + \eta v_2 = 0.4055 + 0.24 + (-0.09) = 0.5555 \quad \checkmark

**Inductive step** :math:`(K = 2 \to K + 1 = 3)`:

A third boosting stage is added with a new decision tree:

.. list-table::
   :header-rows: 1
   :widths: 10 10 35 15

   * - Stage
     - Rule
     - Guard
     - Leaf value :math:`v`
   * - :math:`k = 3`
     - :math:`r_{3a}`
     - :math:`x_1 \leq 4.0 \;\wedge\; x_2 > 6.0`
     - :math:`+1.2`
   * - :math:`k = 3`
     - :math:`r_{3b}`
     - :math:`x_1 \leq 4.0 \;\wedge\; x_2 \leq 6.0`
     - :math:`-0.5`
   * - :math:`k = 3`
     - :math:`r_{3c}`
     - :math:`x_1 > 4.0`
     - :math:`-1.0`

The HCPN is extended by:

* Adding a substitution transition :math:`{t_{stage}}^{(3)}` to :math:`T_{sub}` in :math:`s_{channel}`.
* Mapping :math:`SM_{ch}({t_{stage}}^{(3)}) = s_{DT}`.
* Port-socket bindings identical to existing stages.
* Updating :math:`t_{finalize}`'s guard from :math:`[\kappa = 3]` to :math:`[\kappa = 4]`.
* Updating :math:`\text{STAGE}` colour set from :math:`\{1,2,3\}` to :math:`\{1,2,3,4\}`.

**HCPN state trace for input** :math:`x = (2.0, \; 7.0)` **(continued from** :math:`K = 2` **):**

After stages 1 and 2, the state is identical: :math:`P_{accum} = 0.5555`, :math:`P_{stage} = 3`. The new stage 3 fires:

*Stage 3* (:math:`\kappa = 3`):

* Guard :math:`[\kappa = 3]` is true, so :math:`{t_{stage}}^{(3)}` is enabled.
* DT sub-module fires: :math:`G(r_{3a}) = [2.0 \leq 4.0] \wedge [7.0 > 6.0] = \text{true}`, matched leaf value :math:`v_3 = +1.2`.
* :math:`P_{accum} \leftarrow 0.5555 + 0.1 \times 1.2 = 0.5555 + 0.12 = 0.6755`.
* :math:`P_{stage} \leftarrow 4`.

*Finalize* (:math:`\kappa = 4 = K{+}1`):

* :math:`t_{finalize}` fires, deposits :math:`s_3 = 0.6755` in :math:`P_{result}`.

*Activation:*

.. math::

   p = \sigma(0.6755) = \frac{1}{1 + e^{-0.6755}} = \frac{1}{1 + 0.5090} \approx 0.6626

.. math::

   \hat{y} = 1 \quad (\text{since } 0.6626 \geq 0.5)

Accumulated score matches the extended formula:

.. math::

   s_3 = s_0 + \eta v_1 + \eta v_2 + \eta v_3 = 0.4055 + 0.24 + (-0.09) + 0.12 = 0.6755 \quad \checkmark

**Why the invariant is preserved:** Adding stage :math:`K+1` to a valid :math:`K`-stage channel requires only:

1. One new substitution transition :math:`{t_{stage}}^{(K+1)}` with guard :math:`[\kappa = K{+}1]`.
2. :math:`SM_{ch}({t_{stage}}^{(K+1)}) = s_{DT}` — same leaf module type, new rule set.
3. Port-socket bindings identical to all existing stages.
4. Update :math:`t_{finalize}`'s guard from :math:`[\kappa = K{+}1]` to :math:`[\kappa = K{+}2]`.

The stage counter ensures that stages 1 through :math:`K` execute identically to the :math:`K`-stage case (their guards and rule sets are unchanged). Stage :math:`K+1` fires only after :math:`P_{stage} = K{+}1`, extending the accumulation by exactly :math:`\eta \cdot v_{K+1}`:

.. math::

   {s_{K+1}}^{(c)} = {s_K}^{(c)} + \eta \cdot {v_{K+1}}^{(c)}

The finalize transition then deposits the updated score. The DT invariant guarantees exactly one rule fires per stage, the stage counter guarantees sequential order, and the additive formula generalises directly.

**Conclusion:** By induction on :math:`K`, the GBDT HCPN construction correctly accumulates scores for any number of boosting stages. Each additional stage adds one substitution transition and one additive contribution, without affecting the existing stages or their determinism guarantees.
