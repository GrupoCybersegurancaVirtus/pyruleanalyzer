API Documentation
=================

The pyRuleAnalyzer module exports five main classes:

* :ref:`Rule<rule>` -- Represents a single decision path extracted from a tree.
* :ref:`RuleClassifier<rule_classifier>` -- Core rule-based classifier with prediction, export, and analysis capabilities.
* :ref:`DTAnalyzer<dt_analyzer>` -- Analysis wrapper specialized for Decision Tree models.
* :ref:`RFAnalyzer<rf_analyzer>` -- Analysis wrapper specialized for Random Forest models.
* :ref:`GBDTAnalyzer<gbdt_analyzer>` -- Analysis wrapper specialized for Gradient Boosting Decision Trees models.

Formal verification of the generated Petri net models:

* :ref:`Model Checking<model_checker>` -- Reachability graph, CTL properties and ASK-CTL generation.
* :ref:`Verified Pipeline<verified_pipeline>` -- Train, verify, refine, verify again, export and deploy.

.. toctree::

   rule
   rule_classifier
   dt_analyzer
   rf_analyzer
   gbdt_analyzer
   model_checker
   verified_pipeline
