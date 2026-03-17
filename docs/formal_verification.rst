Formal Verification
===================

Formalization of the Conversion: Tree Models to Hierarchical Coloured Petri Nets (HCPN)
----------------------------------------------------------------------------------------

This section documents the mathematical foundation used by pyRuleAnalyzer to translate machine learning models (DT, RF, GBDT) into formal representations amenable to verification and structural analysis. The formalization employs **Hierarchical Coloured Petri Nets** (HCPN), which extend flat CPNs with module composition, substitution transitions, port-socket bindings, and fusion places. This enables a natural, layered representation where individual decision trees are reusable leaf modules composed into larger ensemble structures.


1. Formal Definitions
^^^^^^^^^^^^^^^^^^^^^

**1.1 Coloured Petri Net**

A Coloured Petri Net (CPN) is the standard 9-tuple:

.. math::

   CPN = (\Sigma, P, T, A, N, C, G, E, I)

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Element
     - Description
   * - :math:`\Sigma`
     - **Colour sets.** The data types carried by tokens: :math:`\text{VEC} = \mathbb{R}^d` (feature vectors), :math:`\text{LABEL}` (finite set of class labels), :math:`\text{PROB} = [0,1]^{|C|}` (probability vectors, used by RF), :math:`\text{SCORE} = \mathbb{R}` (real-valued scores, used by GBDT), and :math:`\text{STAGE} = \{1, 2, \dots, K{+}1\} \subset \mathbb{N}` (stage counter, used by GBDT to enforce sequential accumulation).
   * - :math:`P`
     - **Places.** The set of places within the net.
   * - :math:`T`
     - **Transitions.** The set of transitions within the net.
   * - :math:`A`
     - **Arcs.** Directed connections between places and transitions.
   * - :math:`N`
     - **Node function.** Maps each arc to its source and destination nodes, defining the net topology.
   * - :math:`C`
     - **Colour function.** Assigns colour sets to places (e.g. :math:`C(P_{in}) = \text{VEC}`, :math:`C(P_{out}) = \text{LABEL}`).
   * - :math:`G`
     - **Guards.** Boolean expressions on transitions: :math:`G(t_i) = \bigwedge_{j \in C_i} (x_{feat} \text{ op } \theta_j)`.
   * - :math:`E`
     - **Arc expressions.** Define how tokens flow. Input arcs: :math:`E(P_{in} \to t_i) = x`. Output arcs: :math:`E(t_i \to P_{out}) = v_i`.
   * - :math:`I`
     - **Initialization.** The initial marking of the net's places.

**1.2 CPN Module**

A CPN module extends a flat CPN with an interface for hierarchical composition:

.. math::

   CPN\_module = (CPN, T_{sub}, P_{port}, PT)

where:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Element
     - Description
   * - :math:`CPN`
     - A non-hierarchical Coloured Petri Net :math:`CPN = (\Sigma, P, T, A, N, C, G, E, I)` as defined in Section 1.1.
   * - :math:`T_{sub} \subseteq T`
     - **Substitution transitions.** A subset of transitions that are placeholders for entire sub-modules. When a substitution transition is "refined", it is replaced by the internal net of the referenced sub-module.
   * - :math:`P_{port} \subseteq P`
     - **Port places.** A subset of places that serve as the module's interface with its parent. Tokens cross module boundaries exclusively through port places.
   * - :math:`PT : P_{port} \to \{In, Out, In/Out\}`
     - **Port-type function.** Assigns a direction to each port place: *In* (tokens flow into the module), *Out* (tokens flow out), or *In/Out* (bidirectional).

**1.3 Hierarchical Coloured Petri Net**

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
