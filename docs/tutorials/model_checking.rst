Model Checking and Its Validation
=================================

pyRuleAnalyzer converts a trained tree model into a **Hierarchical Coloured Petri Net** (HCPN) in the native `CPN Tools <https://cpntools.org>`_ ``.cpn`` format. This page covers what comes after: verifying the net, automatically and from Python, how that verification is grounded in the model-checking literature, and how its answers are validated against CPN Tools itself.

The formal mapping behind the nets (Theorem 1 for Decision Trees, Theorem 2 for GBDT, the soft-voting construction for Random Forests) is documented in :doc:`../formal_verification`.


.. _model-checking#why:

Why verify at all?
------------------

Rule refinement removes redundant rules, merges siblings and promotes survivors. When it goes wrong, **accuracy does not move**: overlapping rules usually still agree on the predicted class. What breaks is the *structure* — a rule set with overlaps is a rule list, resolved by "first match wins", and a Petri net is concurrent and has no such priority order. The net becomes nondeterministic while the accuracy report stays clean. Verification is what notices.


.. _model-checking#method:

Method
------

Four techniques are used. They establish different things, so the tool keeps them apart and reports them separately.

.. list-table::
   :header-rows: 1
   :widths: 18 42 40

   * - Technique
     - What it is
     - What it establishes
   * - CTL model checking
     - Explicit-state model checking (Clarke, Grumberg, Kroening, Peled & Veith, *Model Checking*, 2nd ed., MIT Press, 2018; Baier & Katoen, *Principles of Model Checking*, MIT Press, 2008) on the occurrence graph of the CPN, read as a Kripke structure.
     - Temporal properties of every execution of the net, for the input fixed in the initial marking.
   * - SCC analysis
     - Decomposition of the occurrence graph into strongly connected components; a home marking exists exactly when there is one terminal component (Jensen & Kristensen, *Coloured Petri Nets*, Springer, 2009).
     - Home marking (not expressible as a single CTL formula).
   * - Structural guard analysis
     - The leaf guards are conjunctions of axis-aligned bounds, so each leaf covers a box; two leaves overlap exactly when their boxes intersect.
     - Leaf disjointness for **every** input, which no single state space can show.
   * - Conformance testing
     - The class the net computes, compared with the classifier's, on a test set.
     - Functional equivalence on the tested inputs. Not a temporal property.

The occurrence graph
^^^^^^^^^^^^^^^^^^^^

:mod:`pyruleanalyzer.cpn_statespace` builds the occurrence graph under the CPN firing rule: a marking gives every place a multiset of token **values**, a binding element is enabled when its input arcs evaluate to tokens present in the marking and its guard holds, and an arc of the graph is an occurrence of a binding element. There is no abstraction — two markings that put different values in a place are different nodes — so nothing has to be argued about an abstraction being exact.

The generator supports exactly the CPN ML fragment the exporter emits (records, products and lists of reals, arithmetic, the declared soft-voting functions, multisets written with ``++``) and raises ``UnsupportedNet`` on anything else instead of approximating it.

Dead markings
^^^^^^^^^^^^^

Textbook CTL assumes a total transition relation, and the generated nets are meant to stop. The checker gives every dead marking a self-loop (Baier & Katoen), so a maximal path that ends before :math:`\varphi` is a counterexample to :math:`AF\,\varphi`.

**CPN Tools' ASK-CTL does something else.** Its ``FORALL_UNTIL`` — and therefore ``EV`` — is evaluated as "the right operand holds, or the left one holds and *every successor* satisfies the formula"; at a dead marking there are no successors, so it holds vacuously. In other words ``EV(φ)`` means :math:`AF(\varphi \lor dead)`: a net that deadlocks before predicting satisfies ``EV(PRED)``. This is read directly from ``cpnsim/statespacefiles/ASKCTL/ASKCTL.sml`` in the CPN Tools installation, and confirmed by running it (see below). The two readings differ only for ``AF``-type properties, and only when a dead marking without :math:`\varphi` is reachable. The ASK-CTL scripts the tool generates encode the textbook :math:`AF\,\varphi` as

.. code-block:: sml

   fun AFmax phi = AND(EV(phi), NOT(EXIST_UNTIL(NOT(phi), AND(DEAD, NOT(phi)))));

so CPN Tools answers the same question as the checker. The plain ``EV`` reading is still computed and compared, under the keys ``A3_ev``, ``B2_ev``, ``C3a_ev`` and ``D2a_ev``.


.. _model-checking#properties:

The properties
--------------

.. list-table::
   :header-rows: 1
   :widths: 7 26 40 12 15

   * - ID
     - Property
     - Formula
     - Method
     - Applies to
   * - A1
     - Termination
     - :math:`AF\ dead`
     - CTL
     - all
   * - A2
     - No spurious deadlock
     - :math:`\neg EF(dead \land \neg pred)`
     - CTL
     - all
   * - A3
     - Inevitable decision
     - :math:`AF\ pred`
     - CTL
     - all
   * - A4
     - Prediction recoverability
     - :math:`AG\ EF\ pred`
     - CTL
     - all
   * - A5
     - Unique output
     - :math:`AG\ |Prediction| \le 1`
     - CTL
     - all
   * - A6
     - Valid label
     - :math:`AG\ Prediction \subseteq L`
     - CTL
     - all
   * - A7
     - Safeness
     - :math:`AG\ \forall p: |p| \le 1`
     - CTL
     - all
   * - A8
     - Home marking
     - one terminal SCC
     - SCC
     - all
   * - B1
     - Leaf determinism
     - :math:`\forall T: AG\ |EN(leaves_T)| \le 1`
     - CTL
     - all
   * - B2
     - Inevitable leaf selection
     - :math:`\forall T: AF\ EN(leaves_T)`
     - CTL
     - all
   * - B3
     - Single tree output
     - :math:`\forall T: AG\ |out_T| \le 1`
     - CTL
     - all
   * - B4
     - Guard disjointness
     - :math:`\forall T, i \ne j: box_i \cap box_j = \emptyset`
     - structural
     - all
   * - C1
     - Stage precedence
     - :math:`\forall k, m: \neg E[\neg acc_{k,m-1}\ U\ acc_{k,m}]`
     - CTL
     - GBDT
   * - C2
     - No premature score
     - :math:`\forall k: \neg E[\neg acc_{k,M}\ U\ score_k]`
     - CTL
     - GBDT
   * - C3a
     - Inevitable decision firing
     - :math:`AF\ EN(decide)`
     - CTL
     - GBDT
   * - C3b
     - Decision determinism
     - :math:`AG\ |EN(decide)| \le 1`
     - CTL
     - GBDT
   * - D1
     - Complete vote
     - :math:`AG(EN(vote) \rightarrow votes\_all)`
     - CTL
     - RF
   * - D2a
     - Inevitable aggregation
     - :math:`AF\ EN(vote)`
     - CTL
     - RF
   * - D2b
     - Single aggregation
     - :math:`AG\ |Prediction| \le 1`
     - CTL
     - RF
   * - PC
     - Prediction consistency
     - :math:`net(x) = model(x)`
     - testing
     - all

:math:`EN(G)` holds in a marking with an outgoing arc whose transition is in :math:`G`; :math:`|EN(G)|` counts those arcs. :math:`T` ranges over tree subpages, :math:`k` over GBDT class channels, :math:`m` over boosting stages. The quantifiers over :math:`T`, :math:`k` and :math:`m` are conjunctions of one formula per tree, channel or stage — in particular C1 and C2 are checked channel by channel.

A few readings that matter when the table is quoted:

* **A3 is inevitability, not reachability.** Reachability would be :math:`EF\ pred`.
* **A4 is not a home-marking property.** :math:`AG\ EF\ pred` holds if every marking can reach *some* marking with a prediction, possibly different ones. The home marking is A8, computed from the SCC graph, like CPN Tools' ``ListHomeMarkings``.
* **B1 is per input**; B4 is for all inputs. A refinement can create an overlap that the tested input never reaches: B1 then holds and B4 fails.
* **A6 needs a declared label domain.** Reading :math:`L` from the net's own outputs would make the property vacuous. For GBDT and Random Forest the domain follows from the net's structure (Decide transitions, probability-vector width); for a Decision Tree it must be passed as ``class_labels``, and A6 is skipped otherwise.
* **D1 refers to the trees' outputs**, not to Vote's input arcs; otherwise a Vote that stopped waiting for a tree would satisfy it.
* **The temporal properties say the net behaves like a classifier, not that it is the right one.** A net predicting a wrong but valid class passes every CTL property; only PC notices.


.. _model-checking#usage:

Checking a generated net
------------------------

.. code-block:: python

   from pyruleanalyzer import check_cpn

   result = check_cpn("files/gbdt_final.cpn", samples=X_test.iloc[:3],
                      classifier=analyzer.classifier, test_samples=X_test,
                      class_labels=[0, 1, 2])
   print(result.report())
   assert result.passed

``samples`` are model checked, one occurrence graph each; ``test_samples`` are only used for conformance testing, which follows one maximal occurrence sequence per input and is therefore cheap enough for a whole test set.

:class:`~pyruleanalyzer.model_checker.ModelCheckResult` exposes ``passed``, ``failures``, ``skipped``, ``properties``, ``stats``, ``oracle`` (the values CPN Tools is expected to print), ``report()``, ``to_dict()``, ``latex_rows()`` and ``latex_table()``.

Counterexamples are reported as the shortest occurrence sequence reaching the violation, e.g. ``Distribute -> Init -> Feed_1 -> ...``.

State-space explosion
^^^^^^^^^^^^^^^^^^^^^

A :math:`K`-class GBDT with :math:`M` stages per class has :math:`(3M+3)^K + 2` reachable markings and :math:`K(3M+2)(3M+3)^{K-1} + 2` arcs, because the channels interleave freely; the test suite checks both formulas against generated nets. A three-class model with a hundred stages is out of reach for any explicit-state tool, CPN Tools included. When the graph exceeds ``max_nodes`` (200,000 by default) the state-space properties get **no verdict** — the structural analysis and the conformance test still run. An earlier version fell back to a compositional mode; it was removed because its preservation argument had not been proven, and an unproven reduction is not something to report results from.


.. _model-checking#validation:

Validating the checker against CPN Tools
----------------------------------------

Two claims have to hold before the checker's answers can be reported instead of CPN Tools' own.

**Agreement.** :mod:`pyruleanalyzer.cpntools_oracle` runs the CPN Tools 4.0.1 simulator headlessly: the net is compiled by CPN Tools' own Standard ML simulator (driven through Access/CPN, the library CPN IDE uses), the state-space tool is entered exactly as the GUI's *Enter State Space* does it (``switch1.sml`` … ``switch8.sml`` from ``cpnsim/statespacefiles``), ``CalculateOccGraph`` and ``CalculateSccGraph`` build the graphs, and ASK-CTL evaluates the queries generated by :meth:`CPNModelChecker.ml_program`. Every statistic (nodes, arcs, SCC nodes, SCC arcs, dead markings, home markings) and every CTL verdict is compared.

**Sensitivity.** :mod:`pyruleanalyzer.cpn_mutants` derives from each correct net one mutant per property — a small edit of guards, inscriptions or arcs that makes the property false — and the same mutants are given to both engines. Agreement on correct nets alone would also be reached by two checkers that always answer "true".

.. code-block:: bash

   python examples/cpn_tools_crosscheck.py        # writes files/crosscheck/report.md

The requirements are CPN Tools, CPN IDE (for its bundled Access/CPN libraries) and a 32-bit Java 8 runtime with ``jjs``; ``CPNToolsOracle().missing()`` lists anything absent. The test suite runs the agreement tests when ``PYRA_CPNTOOLS=1`` is set.

To run the same check by hand in the CPN Tools GUI, write the script for a net and evaluate it after *Enter State Space*:

.. code-block:: python

   CPNModelChecker("files/gbdt_final.cpn").export_askctl("files/gbdt_final.sml")

.. code-block:: sml

   use "C:/.../files/gbdt_final.sml";


.. _model-checking#refinement:

Verifying a refinement
----------------------

.. code-block:: python

   results = analyzer.model_check(which="both", samples=X_test.iloc[:3],
                                  test_samples=X_test)
   results["final"].passed

The comparison lists every property that held on the initial net and fails on the refined one as a regression.


.. _model-checking#pipeline:

The verified pipeline
---------------------

:func:`~pyruleanalyzer.verified_pipeline.verified_pipeline` chains the whole flow and makes the verification a gate:

.. code-block:: text

   1. train a scikit-learn model, or take one that is already trained
   2. extract the rules and export the initial HCPN
   3. verify the initial model                           (baseline)
   4. refine the rules
   5. verify the refined model                           (acceptance test)
   6. export: Python, binary, C header, Arduino sketch
   7. deploy to the edge device                          (optional)

.. code-block:: python

   from pyruleanalyzer import verified_pipeline

   result = verified_pipeline(
       train_csv="data/train.csv", test_csv="data/test.csv",
       target_feature="Target",
       model_type="Gradient Boosting Decision Trees",
       params={"n_estimators": 8, "max_depth": 3, "random_state": 42},
       remove_below_n_classifications=1,
       verify_samples=3,
       export_formats=("python", "binary", "c", "arduino"),
   )

With ``fail_on_violation=True`` (the default) step 5 raises ``VerificationError`` instead of exporting a model whose net violates a property; the failing result is attached to the exception. An already-trained estimator skips step 1 (``sklearn_model=clf``). Giving the board's FQBN and port also compiles and uploads the sketch with ``arduino-cli``; board sizing is covered in :doc:`modeling_for_arduino`.


.. _model-checking#cli:

Running the examples
--------------------

.. code-block:: bash

   python examples/verified_pipeline_example.py
   python examples/verified_pipeline_example.py --classes 3 --askctl
   python examples/cpn_tools_crosscheck.py
