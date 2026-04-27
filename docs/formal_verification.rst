Formal Verification
===================

Formalization of the Conversion: Tree Models to Hierarchical Coloured Petri Nets (HCPN)
----------------------------------------------------------------------------------------

This section documents the mathematical foundation used by pyRuleAnalyzer to translate machine learning models (DT, RF, GBDT) into formal representations amenable to verification and structural analysis. The formalization employs **Hierarchical Coloured Petri Nets** (HCPN), which extend flat CPNs with module composition, substitution transitions, port-socket bindings, and fusion places. This enables a natural, layered representation where individual decision trees are reusable leaf modules composed into larger ensemble structures.


1. Formal Definitions
^^^^^^^^^^^^^^^^^^^^^

**Definition 1.** A CPN module is a tuple :math:`CPN_M = (P, T, A, \Sigma, V, C, G, E, I, T_{sub}, P_{port}, PT)`, where:

* :math:`P` is a finite set of places;
* :math:`T` is a finite set of transitions such that :math:`P \cap T = \emptyset`;
* :math:`A \subseteq P \times T \cup T \times P` is a set of directed arcs;
* :math:`\Sigma` is a finite non-empty set of colours;
* :math:`V` is a finite set of typed variables such that :math:`Type[v] \in \Sigma` for all variables :math:`v \in V`;
* :math:`C` is a colour function that assigns a colour set to each place;
* :math:`G: T \to EXPR_V` is a guard function that assigns a guard to each transition :math:`t` such that :math:`Type[G(t)] = Bool`. :math:`EXPR_V` stands for an expression where all variables must belong to :math:`V`;
* :math:`E: A \to EXPR_V` is an arc expression function that assigns an arc expression to each arc :math:`a` such that :math:`Type[E(a)] = C(p)_{MULTISET}`, where :math:`p` is the place connected to the arc :math:`a` and :math:`C(p)_{MULTISET}` means that an arc expression must evaluate a multiset of tokens belonging to the colour set of the connected place. Let :math:`S = \{s_1, s_2, s_3, \dots\}` be a non-empty set; a MULTISET over :math:`S` is a function :math:`m: S \to \mathbb{N}` mapping each element :math:`s \in S` into a non-negative integer :math:`m(s) \in \mathbb{N}` called the number of appearances (or coefficient) of :math:`s` in :math:`m`;
* :math:`I: P \to EXPR_{\emptyset}` is an initialization function that assigns an initialization expression to each place :math:`p` such that :math:`Type[I(p)] = C(p)_{MULTISET}`;
* :math:`T_{sub} \subseteq T` is a set of substitution transitions;
* :math:`P_{port} \subseteq P` is a set of port places; and
* :math:`PT: P_{port} \to \{IN, OUT, I/O\}` is a port type function that assigns port types to places.

**1.2 Hierarchical Coloured Petri Net**

An HCPN composes multiple CPN modules into a hierarchical structure:

.. math::

   HCPN = (S, SM, PS, FS)

where:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Element
     - Description
   * - :math:`S`
     - **Modules.** A finite set of CPN modules. Each module is a self-contained net with its own places, transitions, and port interface.
   * - :math:`SM : T_{sub} \to S`
     - **Submodule function.** Maps each substitution transition (across all modules) to the module it represents. The resulting module hierarchy must be **acyclic** (i.e. it forms a directed acyclic graph of module references).
   * - :math:`PS`
     - **Port-socket relation.** For each substitution transition :math:`t \in T_{sub}` and each port place :math:`p` of :math:`SM(t)`, :math:`PS` assigns a *socket place* :math:`p'` in the parent module such that :math:`C(p) = C(p')`. Sockets are the "attachment points" through which the parent communicates with the sub-module.
   * - :math:`FS \subseteq 2^P`
     - **Fusion sets.** Each element of :math:`FS` is a nonempty set of places (possibly from different modules) that are identified as a single logical place. All places in a fusion set share the same colour set and marking at all times. This allows multiple modules to share a common place without explicit arcs between them.


2. Model-Specific Formalizations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each model type has its own dedicated page with the complete HCPN formalization and a numerical induction example demonstrating correctness:

.. toctree::
   :maxdepth: 1

   formal_verification_dt
   formal_verification_rf
   formal_verification_gbdt


3. Rule Extraction Algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The conversion follows the flow implemented in ``rule_classifier.py``, which constructs the HCPN module hierarchy:

1. **Prior Extraction (GBDT only):** Queries scikit-learn's prior estimator to compute the initial score :math:`s_0` (log-odds for binary, centred log-priors for multiclass). This value becomes the initial token in the channel module's :math:`P_{accum}` place.
2. **Traversal:** Recursively traverses each tree structure from root to each leaf node. At every internal node, the left child receives a ``<=`` predicate and the right child receives a ``>`` predicate over the same feature and threshold. Each complete root-to-leaf path produces a rule that becomes a guarded transition within a :math:`s_{DT}` leaf module.
3. **Predicate Construction:** Concatenates the accumulated split decisions along the path into a boolean conjunction that becomes the transition guard :math:`G(t_i)`.
4. **Module Assembly:** Maps the extracted rules into the HCPN hierarchy:

   * **DT:** A single :math:`s_{DT}` module with all rules as transitions.
   * **RF:** A top module :math:`s_{RF}` with :math:`N` substitution transitions, each bound to a :math:`s_{DT}` instance. The fusion set shares :math:`P_{in}`.
   * **GBDT:** A 3-level hierarchy: :math:`s_{GBDT} \to s_{channel} \to s_{DT}`, with :math:`|C|` channel instances and :math:`K` tree instances per channel.

5. **Rule Encapsulation:** Maps each path into a :ref:`Rule<rule>` object. For DT/RF, the rule stores the class label and class distribution (sample counts at the leaf). For GBDT, the rule additionally stores the raw ``leaf_value``, the ``learning_rate``, and the ``class_group``, from which the effective contribution :math:`\eta \cdot v` is computed.

.. note::

   By converting ML models into an HCPN representation, it becomes possible to apply state-space analysis methods at each hierarchical level — detecting unreachable rules within leaf modules, verifying synchronization properties across ensemble compositions, and transforming the model's "black box" into a transparent, formally verifiable logical structure.
