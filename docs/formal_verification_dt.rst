Decision Tree
=============

**Definition:** A Decision Tree :math:`DT` is a set of rules :math:`R = \{r_1, r_2, \dots, r_n\}` that are exhaustive and mutually exclusive. Each rule :math:`r` is defined by a pair :math:`(C, v)`, where :math:`C` is the set of split conditions and :math:`v` is the resulting output :math:`v` (e.g., a class label, probability distribution, or numerical score, valid for DT, RF, and GBDT).

CPN Module :math:`s_{DT}`
---------------------------

The Decision Tree is the **atomic leaf module** of the hierarchy — it contains no substitution transitions. Its definition is:

.. math::

   s_{DT} = (CPN_{DT}, \; T_{sub} = \emptyset, \; P_{port} = \{P_{in}, P_{out}\}, \; PT)

with:

* :math:`PT(P_{in}) = In` — the input feature vector enters through this port.
* :math:`PT(P_{out}) = Out` — the classification result exits through this port.
* :math:`T_{sub} = \emptyset` — no substitution transitions; all transitions are regular rule transitions.

**Mapping** :math:`\Psi(DT \to s_{DT})`:

Each rule :math:`r_i = (C_i, v_i)` is converted into a guarded transition :math:`t_i`:

* **Input arc:** :math:`(P_{in}, t_i)` removes the feature vector token :math:`x`.
* **Guard:** :math:`G(t_i) = \bigwedge_{j \in C_i} (x_{feat_j} \text{ op } \theta_j)`, the conjunction of all split conditions along the path from root to leaf.
* **Output arc:** :math:`(t_i, P_{out})` generates a new token representing the output :math:`v_i`.

The figure below shows the CPN structure for a Decision Tree module. The input port place :math:`P_{input}` (colour type :math:`\text{VEC}`) holds the feature vector token :math:`x`, which is available to all transitions via their input arcs. Each transition :math:`T\_rule\_i` has a guard composed of the conjunction of split predicates along its tree path. Since the guards are derived from complementary ``<=`` / ``>`` splits at every internal node, exactly one transition is enabled for any complete input, removing the token :math:`x` and generating a new token with the output value in :math:`P_{output}`.

.. figure:: _static/images/rule_to_cpn.png
   :align: center
   :width: 80%
   :alt: Decision Tree CPN module diagram

   CPN module :math:`s_{DT}`: all leaf paths of a Decision Tree become guarded transitions between port places :math:`P_{input}` (In) and :math:`P_{output}` (Out).

**Invariance Property:** Due to the mutually exclusive nature of the rules in a DT, for any initial marking :math:`M_0` containing an input token :math:`x \in \mathbb{R}^d`, exactly one transition :math:`t \in T` will be enabled, guaranteeing total determinism within the module and avoiding any structural conflict [1]_.

.. note::

   This property holds under the assumption that the input vector :math:`x` contains a value for every feature used by the tree. If any feature is missing, no transition may be enabled and the net falls back to the classifier's default class.


Numerical Induction Example
-----------------------------

The following example demonstrates that the CPN module construction preserves the determinism invariant for a tree with :math:`n` rules and remains valid when extended to :math:`n+1` rules.

**Setup:** Consider a 2-dimensional feature space :math:`(x_1, x_2)` with classes :math:`\{A, B, C\}`.

**Base case** :math:`(n = 2)`:

A minimal Decision Tree with a single split on :math:`x_1` at threshold :math:`\theta = 5.0`:

.. list-table::
   :header-rows: 1
   :widths: 15 50 15

   * - Rule
     - Guard
     - Class
   * - :math:`r_1`
     - :math:`x_1 \leq 5.0`
     - :math:`A`
   * - :math:`r_2`
     - :math:`x_1 > 5.0`
     - :math:`B`

The CPN module is:

.. math::

   s_{DT}^{(2)} = (CPN_{DT}^{(2)}, \; T_{sub} = \emptyset, \; P_{port} = \{P_{in}, P_{out}\}, \; PT)

with :math:`T = \{t_1, t_2\}`, :math:`G(t_1) = [x_1 \leq 5.0]`, :math:`G(t_2) = [x_1 > 5.0]`.

**Verification with input** :math:`x = (3.0, \; 7.0)`:

* :math:`G(t_1) = [3.0 \leq 5.0] = \text{true}` — **enabled**, generates a new token for :math:`A`.
* :math:`G(t_2) = [3.0 > 5.0] = \text{false}` — disabled.

Exactly one transition fires. :math:`\checkmark`

**Verification with input** :math:`x = (8.0, \; 2.0)`:

* :math:`G(t_1) = [8.0 \leq 5.0] = \text{false}` — disabled.
* :math:`G(t_2) = [8.0 > 5.0] = \text{true}` — **enabled**, generates a new token for :math:`B`.

Exactly one transition fires. :math:`\checkmark`

**Inductive step** :math:`(n = 2 	o n + 1 = 3)`:

The tree grows by splitting a leaf node. Scikit-learn implements Decision Trees strictly as binary trees; therefore, any split always generates exactly two children. Splitting the leaf :math:`r_2` (guard :math:`x_1 > 5.0`) on feature :math:`x_2` at threshold :math:`	heta = 4.0` replaces :math:`r_2` with two new rules, each inheriting the parent guard:

.. list-table::
   :header-rows: 1
   :widths: 15 50 15

   * - Rule
     - Guard
     - Class
   * - :math:`r_1`
     - :math:`x_1 \leq 5.0`
     - :math:`A`
   * - :math:`r_2'`
     - :math:`x_1 > 5.0 \;\wedge\; x_2 \leq 4.0`
     - :math:`B`
   * - :math:`r_3`
     - :math:`x_1 > 5.0 \;\wedge\; x_2 > 4.0`
     - :math:`C`

The updated CPN module is:

.. math::

   s_{DT}^{(3)} = (CPN_{DT}^{(3)}, \; T_{sub} = \emptyset, \; P_{port} = \{P_{in}, P_{out}\}, \; PT)

with :math:`T = \{t_1, t_2', t_3\}`.

**Why the invariant is preserved:** The split replaces one leaf with two children connected by complementary predicates (:math:`\leq` / :math:`>`). The original guard :math:`G(t_2)` is partitioned into two strictly narrower, non-overlapping regions:

.. math::

   G(t_2) = G(t_2') \;\dot\cup\; G(t_3)

where :math:`\dot\cup` denotes disjoint union. Since :math:`G(t_1)` was already disjoint from :math:`G(t_2)` by the inductive hypothesis, it remains disjoint from both :math:`G(t_2')` and :math:`G(t_3)`. Therefore the guards :math:`\{G(t_1), G(t_2'), G(t_3)\}` form a partition of the input space.

**Verification with input** :math:`x = (8.0, \; 2.0)`:

* :math:`G(t_1) = [8.0 \leq 5.0] = \text{false}` — disabled.
* :math:`G(t_2') = [8.0 > 5.0] \wedge [2.0 \leq 4.0] = \text{true} \wedge \text{true} = \text{true}` — **enabled**, generates a new token for :math:`B`.
* :math:`G(t_3) = [8.0 > 5.0] \wedge [2.0 > 4.0] = \text{true} \wedge \text{false} = \text{false}` — disabled.

Exactly one transition fires. :math:`\checkmark`

**Verification with input** :math:`x = (8.0, \; 6.0)`:

* :math:`G(t_1) = [8.0 \leq 5.0] = \text{false}` — disabled.
* :math:`G(t_2') = [8.0 > 5.0] \wedge [6.0 \leq 4.0] = \text{true} \wedge \text{false} = \text{false}` — disabled.
* :math:`G(t_3) = [8.0 > 5.0] \wedge [6.0 > 4.0] = \text{true} \wedge \text{true} = \text{true}` — **enabled**, generates a new token for :math:`C`.

Exactly one transition fires. :math:`\checkmark`

**Conclusion:** By the Inductive Hypothesis (**H.I.1**), we assume that for a tree with :math:`n` leaves, the guards form a valid partition. As demonstrated, every binary split partitioning a single leaf strictly preserves this property for :math:`n+1` rules. Therefore, by inductive reasoning on the number of leaves, any Decision Tree CPN module :math:`s_{DT}^{(n)}` guarantees exactly one enabled transition for every valid input :math:`x \in \mathbb{R}^d`.

.. [1] Murata, T. (1989). Petri nets: Properties, analysis and applications. *Proceedings of the IEEE*, 77(4), 541-580.