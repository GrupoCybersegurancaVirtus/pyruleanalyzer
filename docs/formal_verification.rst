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

Where each element is mapped from the decision logic of the original model, allowing the data flow (coloured tokens) to represent the classification of an instance.


2. Fundamental Conversion Theorem (Decision Tree)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Definition:** A Decision Tree :math:`DT` is a set of rules :math:`R = \{r_1, r_2, \dots, r_n\}` that are exhaustive and mutually exclusive. Each rule :math:`r` is defined by a pair :math:`(C, v)`, where :math:`C` is the set of split conditions and :math:`v` is the resulting class.

**Mapping** :math:`\Psi(DT \to CPN)`:

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - CPN Element
     - Tree Model Mapping
     - Mathematical / Logical Description
   * - Colour Set (:math:`\Sigma`)
     - Feature Space :math:`\mathbb{R}^d`
     - Defines the token data type (input vector).
   * - Places (:math:`P`)
     - Process States
     - :math:`P_{in}` (Input), :math:`P_{eval}` (Processing) and :math:`P_{out}` (Classification).
   * - Transitions (:math:`T`)
     - Rule Set :math:`R`
     - Each rule :math:`r_i` becomes a transition :math:`t_i \in T`.
   * - Guards (:math:`G`)
     - Rule Conditions
     - :math:`G(t_i) = \bigwedge_{j \in C_i} (x_{feat} \text{ op } \theta_j)`.
   * - Expressions (:math:`E`)
     - Leaf Value
     - :math:`E(t_i \to P_{out}) = v_i`, where :math:`v_i` is the class label.

**Invariance Property:** Due to the mutually exclusive nature of the rules in a DT, for any initial marking :math:`M_0` containing an input token :math:`X`, exactly one transition :math:`t \in T` will be enabled, guaranteeing total determinism.


3. Variations for Ensemble Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The tool extends the base theorem to support complex models through modular compositions.

3.1 Random Forest (Parallel Composition)
"""""""""""""""""""""""""""""""""""""""""

For a forest with :math:`N` trees, the net is composed of :math:`N` sub-pages :math:`PN_{tree}`.

* **Synchronization:** A collector place :math:`P_{collect}` accumulates the output tokens from each sub-net.
* **Voting:** An aggregation transition :math:`t_{vote}` applies the mode function (majority voting):

.. math::

   v_{final} = \text{mode}(\{v_1, v_2, \dots, v_n\})

3.2 GBDT (Additive Composition)
""""""""""""""""""""""""""""""""

In Gradient Boosting, the sub-nets operate under a residual accumulation regime.

* **Token Value:** Tokens carry real-valued scores instead of discrete classes.
* **Arc Expression:** Each transition :math:`t_{ij}` contributes an increment :math:`\Delta` to the current score:

.. math::

   Score_{next} = Score_{current} + (\eta \cdot v_{ij})

Where :math:`\eta` is the model's learning rate.


4. Rule Extraction Algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The conversion follows the flow implemented in ``rule_classifier.py``:

1. **Traversal:** Traverses the tree graph down to each leaf node.
2. **Predicate:** Concatenates the path decisions into a boolean clause (Guard).
3. **Encapsulation:** Maps the path into a :ref:`Rule<rule>` object that serves as a blueprint for the CPN transition.

.. note::

   By converting ML models to this format, it becomes possible to apply state-space analysis methods to detect unreachable rules or logic conflicts, transforming the model's "black box" into a transparent logical structure.
