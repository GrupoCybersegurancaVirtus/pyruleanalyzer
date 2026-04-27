Random Forest
=============

The HCPN framework enables a natural representation of the Random Forest: each individual tree is an instance of the leaf module :math:`s_{DT}`, composed into a top-level module via substitution transitions, port-socket bindings, and fusion places.

HCPN Formalization
-------------------

For a Random Forest with :math:`N` trees, the HCPN is:

.. math::

   HCPN_{RF} = (S_{RF}, \; SM_{RF}, \; PS_{RF}, \; FS_{RF})

**Modules** :math:`S_{RF} = \{s_{RF}, \; s_{DT}\}`:

* :math:`s_{RF}` — the **top-level module** containing the voting logic.
* :math:`s_{DT}` — the **leaf module** (see :doc:`formal_verification_dt`), reused :math:`N` times.

**Top-level module** :math:`s_{RF}`:

.. math::

   s_{RF} = (CPN_{RF}, \; T_{sub} = \{t_{tree_1}, \dots, t_{tree_N}\}, \; P_{port} = \{P_{in}, P_{out}\}, \; PT)

with:

* :math:`P = \{P_{in}, P_{collect}, P_{out}\}` — input, collector, and output places.
* :math:`C(P_{in}) = \text{VEC}`, :math:`C(P_{collect}) = \text{PROB}`, :math:`C(P_{out}) = \text{LABEL}`.
* :math:`T = \{t_{tree_1}, \dots, t_{tree_N}, \; t_{vote}\}` — :math:`N` substitution transitions plus one regular aggregation transition.
* :math:`t_{vote}` — a regular transition (not in :math:`T_{sub}`) that performs soft voting.

**Submodule function** :math:`SM_{RF}`:

.. math::

   SM_{RF}(t_{tree_k}) = s_{DT} \quad \text{for } k = 1, \dots, N

Each substitution transition :math:`t_{tree_k}` is refined into the DT leaf module. Although the same module *definition* :math:`s_{DT}` is referenced :math:`N` times, each instance carries its own rule set (extracted from the :math:`k`-th tree in the forest).

**Port-socket bindings** :math:`PS_{RF}`:

For each substitution transition :math:`t_{tree_k}`:

* The *In* port :math:`P_{in}` of :math:`s_{DT}` is bound to socket :math:`P_{in}` in :math:`s_{RF}`.
* The *Out* port :math:`P_{out}` of :math:`s_{DT}` is bound to socket :math:`P_{collect}` in :math:`s_{RF}`.

Colour compatibility is ensured: the DT module's output port produces :math:`\text{PROB}` vectors (the normalised class distribution at the matched leaf), matching :math:`C(P_{collect}) = \text{PROB}`.

.. note::

   In the RF context, the DT module's output arc expression is adapted: instead of generating a new token for a discrete label, it yields the normalised class distribution :math:`\mathbf{p}_k \in [0,1]^{|C|}` at the matched leaf. The port colour is therefore :math:`\text{PROB}` rather than :math:`\text{LABEL}`.

**Fusion set** :math:`FS_{RF}`:

.. math::

   FS_{RF} = \{\{P_{in}\}\}

The input place :math:`P_{in}` is a **fusion place** shared across the top module and all :math:`N` DT sub-module instances. Every tree evaluates the same input token :math:`x` without duplicating it.

**Soft Voting (transition** :math:`t_{vote}` **):**

The aggregation transition :math:`t_{vote}` removes all :math:`N` probability tokens from :math:`P_{collect}` (enabled only when :math:`|P_{collect}| \geq N`). Its output arc expression computes:

.. math::

   \bar{\mathbf{p}} = \frac{1}{N} \sum_{k=1}^{N} \mathbf{p}_k, \qquad v_{final} = \arg\max_{c} \; \bar{p}_c

This matches the default behaviour of scikit-learn's ``RandomForestClassifier``, which uses soft voting via ``predict_proba`` averaging.

The following figure shows the parallel composition. Each sub-page :math:`Tree_k` is an independent CPN module whose output probability vector is collected in :math:`P_{collector}`. The aggregation transition :math:`T_{voter}` computes the averaged probabilities and generates a new token for the winning class in :math:`P_{final\_class}`.

.. figure:: _static/images/rf_to_cpn.png
   :align: center
   :width: 80%
   :alt: Random Forest HCPN diagram

   HCPN for Random Forest: :math:`N` substitution transitions (each refining to a DT module) feed probability vectors into a collector place, followed by a soft-voting aggregation transition.

.. note::

   The figure labels the aggregation transition as "Mode Function" for visual simplicity. In the actual implementation, soft voting (probability averaging) is used as the primary path; hard voting (mode) exists only as a fallback when class distribution data is unavailable.


Numerical Induction Example
-----------------------------

The following example demonstrates that the HCPN construction preserves correct soft-voting semantics for a forest with :math:`N` trees and remains valid when extended to :math:`N+1` trees.

**Setup:** Binary classification with classes :math:`\{0, 1\}`. Input :math:`x = (3.0, \; 7.0)`.

**Base case** :math:`(N = 2)`:

Two decision trees, each generating a new token representing a probability vector at its matched leaf:

.. list-table::
   :header-rows: 1
   :widths: 10 40 30

   * - Tree
     - Matched rule guard
     - Output :math:`\mathbf{p}_k`
   * - :math:`T_1`
     - :math:`x_1 \leq 5.0` (true)
     - :math:`(0.8, \; 0.2)`
   * - :math:`T_2`
     - :math:`x_2 > 4.0` (true)
     - :math:`(0.6, \; 0.4)`

**HCPN state trace:**

1. :math:`P_{in}` holds token :math:`x = (3.0, 7.0)`. Fusion set shares it with both DT sub-modules.
2. :math:`t_{tree_1}` fires (substitution transition refines to :math:`s_{DT}^{(1)}`): one rule matches, generates a new token for :math:`(0.8, 0.2)` in :math:`P_{collect}`.
3. :math:`t_{tree_2}` fires (substitution transition refines to :math:`s_{DT}^{(2)}`): one rule matches, generates a new token for :math:`(0.6, 0.4)` in :math:`P_{collect}`.
4. :math:`P_{collect}` now contains 2 tokens. :math:`t_{vote}` is enabled (requires :math:`N = 2` tokens):

.. math::

   \bar{\mathbf{p}} = \frac{1}{2}\bigl((0.8, 0.2) + (0.6, 0.4)\bigr) = (0.70, \; 0.30)

.. math::

   v_{final} = \arg\max(0.70, \; 0.30) = 0

Result: class :math:`0`. :math:`\checkmark`

**Inductive step** :math:`(N = 2 \to N + 1 = 3)`:

A third tree :math:`T_3` is added to the forest. The HCPN is extended by:

* Adding a substitution transition :math:`t_{tree_3}` to :math:`T_{sub}` in the top module.
* Mapping :math:`SM_{RF}(t_{tree_3}) = s_{DT}`.
* Binding its ports: :math:`P_{in} \leftrightarrow P_{in}` (via fusion), :math:`P_{out} \leftrightarrow P_{collect}`.
* Updating :math:`t_{vote}`'s enabling condition to require :math:`N+1 = 3` tokens.

.. list-table::
   :header-rows: 1
   :widths: 10 40 30

   * - Tree
     - Matched rule guard
     - Output :math:`\mathbf{p}_k`
   * - :math:`T_1`
     - :math:`x_1 \leq 5.0` (true)
     - :math:`(0.8, \; 0.2)`
   * - :math:`T_2`
     - :math:`x_2 > 4.0` (true)
     - :math:`(0.6, \; 0.4)`
   * - :math:`T_3`
     - :math:`x_1 \leq 5.0 \;\wedge\; x_2 > 6.0` (true)
     - :math:`(0.3, \; 0.7)`

**HCPN state trace:**

1. :math:`P_{in}` holds :math:`x`. Fusion shares it with all 3 sub-modules.
2. Three substitution transitions fire independently. :math:`P_{collect}` receives 3 tokens.
3. :math:`t_{vote}` is enabled (requires :math:`N+1 = 3` tokens):

.. math::

   \bar{\mathbf{p}} = \frac{1}{3}\bigl((0.8, 0.2) + (0.6, 0.4) + (0.3, 0.7)\bigr) = \left(\frac{1.7}{3}, \; \frac{1.3}{3}\right) \approx (0.567, \; 0.433)

.. math::

   v_{final} = \arg\max(0.567, \; 0.433) = 0

Result: class :math:`0`. :math:`\checkmark`

**Why the invariant is preserved:** Adding a tree :math:`T_{N+1}` to a valid :math:`N`-tree HCPN requires only:

1. One new substitution transition :math:`t_{tree_{N+1}}` with :math:`SM(t_{tree_{N+1}}) = s_{DT}`.
2. Port-socket bindings identical to those of all existing trees.
3. The fusion set :math:`FS` is unchanged — :math:`P_{in}` is already shared.
4. The voting transition :math:`t_{vote}` updates its arc weight from :math:`N` to :math:`N+1`.

The DT leaf module guarantees exactly one rule fires per tree (by the DT invariant). Therefore :math:`P_{collect}` always receives exactly :math:`N+1` tokens, :math:`t_{vote}` fires exactly once, and the averaging formula generalises directly:

.. math::

   {\bar{\mathbf{p}}}^{(N+1)} = \frac{1}{N+1}\left(\sum_{k=1}^{N} \mathbf{p}_k + \mathbf{p}_{N+1}\right) = \frac{N}{N+1}\,{\bar{\mathbf{p}}}^{(N)} + \frac{1}{N+1}\,\mathbf{p}_{N+1}

This is a convex combination of the previous average and the new tree's output, which is always well-defined and produces a valid probability vector.

**Conclusion:** By induction on :math:`N`, the RF HCPN construction correctly computes soft-voting for any number of trees. Each additional tree adds one substitution transition and one probability token, without affecting the existing modules or their determinism guarantees.
