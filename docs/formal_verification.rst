Formal Verification
===================

Formalization of the Conversion: Tree Models to Coloured Petri Nets (CPN)
--------------------------------------------------------------------------

This section documents the mathematical foundation used by pyRuleAnalyzer to translate machine learning models (DT, RF, GBDT) into formal representations amenable to verification and structural analysis.


1. Formal Space Definition
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The formalization is based on the definition of a Coloured Petri Net (CPN), denoted by the tuple:

.. math::

   CPN = (\Sigma, P, T, A, N, C, G, E, I)

The following table describes the role of each element in the context of a tree-based classifier:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Element
     - Description
   * - :math:`\Sigma`
     - **Colour sets.** The data types carried by tokens: :math:`\text{VEC} = \mathbb{R}^d` (feature vectors), :math:`\text{LABEL}` (finite set of class labels), :math:`\text{PROB} = [0,1]^{|C|}` (probability vectors, used by RF), :math:`\text{SCORE} = \mathbb{R}` (real-valued scores, used by GBDT), and :math:`\text{STAGE} = \{1, 2, \dots, K{+}1\} \subset \mathbb{N}` (stage counter, used by GBDT to enforce sequential accumulation).
   * - :math:`P`
     - **Places.** :math:`P = \{P_{in}, P_{out}\}` at minimum; ensemble models add intermediate places (e.g. :math:`P_{collect}` for RF, :math:`P_{accum}` and :math:`P_{stage}` for GBDT).
   * - :math:`T`
     - **Transitions.** Each extracted rule :math:`r_i` becomes a transition :math:`t_i \in T`.
   * - :math:`A`
     - **Arcs.** Directed connections between places and transitions: :math:`(P_{in}, t_i)` and :math:`(t_i, P_{out})` for every transition.
   * - :math:`N`
     - **Node function.** Maps each arc to its source and destination nodes, defining the net topology.
   * - :math:`C`
     - **Colour function.** Assigns colour sets to places: :math:`C(P_{in}) = \text{VEC}`, :math:`C(P_{out}) = \text{LABEL}`. For ensemble models: :math:`C(P_{collect}) = \text{PROB}` (RF), :math:`C(P_{accum}) = \text{SCORE}` and :math:`C(P_{stage}) = \text{STAGE}` (GBDT).
   * - :math:`G`
     - **Guards.** Boolean expressions on transitions: :math:`G(t_i) = \bigwedge_{j \in C_i} (x_{feat} \text{ op } \theta_j)`.
   * - :math:`E`
     - **Arc expressions.** Define how tokens flow through the net. Input arcs: :math:`E(P_{in} \to t_i) = x` (the transition binds the token to variable :math:`x`). Output arcs: :math:`E(t_i \to P_{out}) = v_i` (the transition produces the class label).
   * - :math:`I`
     - **Initialization.** The initial marking: :math:`I(P_{in}) = \{x\}` (one token with the input vector), :math:`I(P_{out}) = \emptyset`.


2. Fundamental Conversion Theorem (Decision Tree)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Definition:** A Decision Tree :math:`DT` is a set of rules :math:`R = \{r_1, r_2, \dots, r_n\}` that are exhaustive and mutually exclusive. Each rule :math:`r` is defined by a pair :math:`(C, v)`, where :math:`C` is the set of split conditions and :math:`v` is the resulting class.

**Mapping** :math:`\Psi(DT \to CPN)`:

Each rule :math:`r_i = (C_i, v_i)` is converted into a guarded transition :math:`t_i`:

* **Input arc:** :math:`(P_{in}, t_i)` consumes the feature vector token :math:`x`.
* **Guard:** :math:`G(t_i) = \bigwedge_{j \in C_i} (x_{feat_j} \text{ op } \theta_j)`, the conjunction of all split conditions along the path from root to leaf.
* **Output arc:** :math:`(t_i, P_{out})` produces the class label :math:`v_i`.

The figure below shows the complete CPN structure for a Decision Tree. The input place :math:`P_{input}` (colour type :math:`\text{VEC}`) holds the feature vector token :math:`x`, which is available to all transitions via their input arcs. Each transition :math:`T\_rule\_i` has a guard composed of the conjunction of split predicates along its tree path. Since the guards are derived from complementary ``<=`` / ``>`` splits at every internal node, exactly one transition is enabled for any complete input, consuming :math:`x` and depositing the class label in :math:`P_{output}`.

.. figure:: _static/images/rule_to_cpn.png
   :align: center
   :width: 80%
   :alt: Decision Tree to CPN mapping diagram

   Mapping :math:`\Psi(DT \to CPN)`: all leaf paths of a Decision Tree become guarded transitions between :math:`P_{input}` and :math:`P_{output}`.

**Invariance Property:** Due to the mutually exclusive nature of the rules in a DT, for any initial marking :math:`M_0` containing an input token :math:`x \in \mathbb{R}^d`, exactly one transition :math:`t \in T` will be enabled, guaranteeing total determinism.

.. note::

   This property holds under the assumption that the input vector :math:`x` contains a value for every feature used by the tree. If any feature is missing, no transition may be enabled and the net falls back to the classifier's default class.


3. Variations for Ensemble Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The tool extends the base theorem to support complex models through modular compositions.

3.1 Random Forest (Parallel Composition)
"""""""""""""""""""""""""""""""""""""""""

For a forest with :math:`N` trees, the net is composed of :math:`N` sub-pages :math:`PN_{tree}`, each containing the CPN of a single Decision Tree.

* **Input Distribution:** The input place :math:`P_{in}` is modelled as a *fusion place* shared across all :math:`N` sub-pages. Each sub-page binds the same token :math:`x` through its input arcs, ensuring every tree evaluates the same instance without duplicating the token.
* **Parallelism:** Each sub-page evaluates its own rule set independently. Instead of producing a discrete label, the matched leaf's output arc expression yields the normalised class distribution at that leaf, i.e. a probability vector :math:`\mathbf{p}_k \in [0,1]^{|C|}` (colour type :math:`\text{PROB}`).
* **Synchronization:** A collector place :math:`P_{collect}` with :math:`C(P_{collect}) = \text{PROB}` accumulates the :math:`N` probability vectors from all sub-nets.
* **Soft Voting:** An aggregation transition :math:`t_{vote}` consumes all :math:`N` probability tokens from :math:`P_{collect}` via an input arc with weight :math:`N` (i.e., :math:`t_{vote}` is enabled only when :math:`P_{collect}` contains at least :math:`N` tokens, ensuring all trees have completed). The output arc expression averages the per-tree probability vectors and selects the class with the highest mean probability:

.. math::

   \bar{\mathbf{p}} = \frac{1}{N} \sum_{k=1}^{N} \mathbf{p}_k, \qquad v_{final} = \arg\max_{c} \; \bar{p}_c

This matches the default behaviour of scikit-learn's ``RandomForestClassifier``, which uses soft voting via ``predict_proba`` averaging.

The following figure shows the parallel composition. Each sub-page :math:`Tree_k` is an independent CPN whose output probability vector is collected in :math:`P_{collector}`. The aggregation transition :math:`T_{voter}` computes the averaged probabilities and deposits the winning class in :math:`P_{final\_class}`.

.. figure:: _static/images/rf_to_cpn.png
   :align: center
   :width: 80%
   :alt: Random Forest to CPN parallel composition diagram

   Parallel CPN composition for Random Forest: :math:`N` independent tree sub-nets feed into a collector place, followed by a soft-voting aggregation transition.

.. note::

   The figure labels the aggregation transition as "Mode Function" for visual simplicity. In the actual implementation, soft voting (probability averaging) is used as the primary path; hard voting (mode) exists only as a fallback when class distribution data is unavailable.

3.2 GBDT (Additive Composition)
""""""""""""""""""""""""""""""""

In Gradient Boosting, classification is performed by sequential accumulation of real-valued scores. The implementation follows scikit-learn's One-vs-Rest (OVR) scheme: for a problem with :math:`|C|` classes, there are :math:`|C|` independent score channels (for binary classification, a single channel for the positive class suffices).

**3.2.1 OVR Structure**

For each class :math:`c \in C`, a dedicated sub-net :math:`SN^{(c)}` is constructed. In the binary case only one sub-net exists (for the positive class). The sub-nets operate in parallel, each accumulating its own score independently:

.. math::

   {s_K}^{(c)} = {s_0}^{(c)} + \sum_{k=1}^{K} \eta \cdot {v_k}^{(c)}

where :math:`K` is the number of boosting stages, :math:`\eta` is the learning rate, and :math:`{v_k}^{(c)}` is the leaf value of the matched rule in tree :math:`k` for class :math:`c`.

**3.2.2 Initial Score (Bias)**

Each sub-net starts from an initial score :math:`{s_0}^{(c)}` derived from the training set prior. This score is extracted from scikit-learn's prior estimator (``DummyClassifier``) and placed as the initial token in :math:`{P_{accum}}^{(c)}`:

.. math::

   {s_0}^{(\text{binary})} = \ln\!\frac{p}{1-p}, \qquad {s_{0,c}}^{(\text{multi})} = \ln p_c - \frac{1}{|C|}\sum_j \ln p_j

**3.2.3 Sequential Accumulation Cycle**

Within each sub-net :math:`SN^{(c)}`, the :math:`K` boosting stages are modelled as a sequence of transitions :math:`{t_1}^{(c)}, {t_2}^{(c)}, \dots, {t_K}^{(c)}`. Sequencing is enforced by a *stage counter* token :math:`\kappa \in \{1, 2, \dots, K\}` held in a control place :math:`{P_{stage}}^{(c)}`:

* **Places:** :math:`{P_{accum}}^{(c)}` (colour :math:`\text{SCORE}`) holds the current cumulative score; :math:`{P_{stage}}^{(c)}` (colour :math:`\text{STAGE}`) holds the stage counter.
* **Guard:** :math:`G({t_k}^{(c)}) = [\kappa = k] \;\wedge\; G_{tree_k}(x)`, ensuring the transition fires only at the correct stage and when the tree's rule conditions are met.
* **Input arcs:** :math:`{t_k}^{(c)}` consumes the current score :math:`s` from :math:`{P_{accum}}^{(c)}` and the counter :math:`\kappa` from :math:`{P_{stage}}^{(c)}`.
* **Output arcs:** :math:`{t_k}^{(c)}` deposits :math:`s + \eta \cdot {v_k}^{(c)}` back into :math:`{P_{accum}}^{(c)}` and :math:`\kappa + 1` into :math:`{P_{stage}}^{(c)}`.

This guarantees that at most one transition per sub-net is enabled at any time, and that scores are accumulated in the correct order.

.. note::

   Each stage transition :math:`{t_k}^{(c)}` is itself an abstraction: within a single boosting tree there are multiple leaf paths (rules), each with its own guard. In a fully expanded net, :math:`{t_k}^{(c)}` would be replaced by a sub-page containing guarded transitions for every leaf, exactly as in the DT case (Section 2). The single-transition notation is used here for clarity, with the understanding that the tree's internal mutual exclusivity guarantees that exactly one leaf fires per stage.

**3.2.4 Activation Transition**

After all :math:`K` stages, a final activation transition :math:`t_{act}` produces the predicted class. In the binary case, :math:`t_{act}` has a single input arc from :math:`P_{accum}` and is guarded by :math:`[\kappa = K{+}1]`. In the multiclass case, :math:`t_{act}` is a *synchronization transition* with :math:`|C|` input arcs — one from each sub-net's accumulator :math:`{P_{accum}}^{(c)}` — and :math:`|C|` corresponding guards :math:`{\kappa}^{(c)} = K{+}1`. The transition fires only when **all** sub-nets have completed their :math:`K` stages, collecting the terminal scores :math:`{s_K}^{(1)}, {s_K}^{(2)}, \dots, {s_K}^{(|C|)}` in a single atomic step. The activation is a two-step process:

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

The figure below depicts the additive composition for a single class channel. The place :math:`P_{initial\_score}` holds the bias token :math:`s_0`. Inside the additive boosting cycle, each gradient step transition :math:`T_{tree\_k}` adds :math:`\eta \cdot v_k` to the accumulator. After the final iteration, the activation transition :math:`T_{activation}` (sigmoid or softmax) converts the raw score into a class probability and deposits the result in :math:`P_{final\_prediction}`. For multiclass problems, :math:`|C|` such channels operate in parallel, and the activation transition collects all channel scores before producing the final label.

.. figure:: _static/images/gbdt_to_cpn.png
   :align: center
   :width: 85%
   :alt: GBDT to CPN additive composition diagram

   Additive CPN composition for GBDT (single class channel): initial bias score, sequential accumulation with learning rate weighting, and a final activation transition (sigmoid/softmax).


4. Rule Extraction Algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The conversion follows the flow implemented in ``rule_classifier.py``:

1. **Prior Extraction (GBDT only):** Queries scikit-learn's prior estimator to compute the initial score :math:`s_0` (log-odds for binary, centred log-priors for multiclass). This value is encapsulated as a special :ref:`Rule<rule>` with no conditions, which becomes the initial token in :math:`P_{accum}`.
2. **Traversal:** Recursively traverses the tree structure from root to each leaf node. At every internal node, the left child receives a ``<=`` predicate and the right child receives a ``>`` predicate over the same feature and threshold.
3. **Predicate Construction:** Concatenates the accumulated split decisions along the path into a boolean conjunction that becomes the transition guard.
4. **Rule Encapsulation:** Maps the path into a :ref:`Rule<rule>` object. For DT/RF, the rule stores the class label and class distribution (sample counts at the leaf). For GBDT, the rule additionally stores the raw ``leaf_value``, the ``learning_rate``, and the ``class_group``, from which the effective contribution :math:`\eta \cdot v` is computed.

.. note::

   By converting ML models to this format, it becomes possible to apply state-space analysis methods to detect unreachable rules or logic conflicts, transforming the model's "black box" into a transparent logical structure.
