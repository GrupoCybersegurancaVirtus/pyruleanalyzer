.. _model_checker:

Model Checking
==============

Verification of the HCPN models written by the exporter, and its validation
against CPN Tools. See :doc:`../tutorials/model_checking` for the method.

Checker
-------

.. autoclass:: pyruleanalyzer.CPNModelChecker
   :members: check, state_space, ml_program, export_askctl, as_sample

.. autoclass:: pyruleanalyzer.ModelCheckResult
   :members:

.. autofunction:: pyruleanalyzer.check_cpn

.. autofunction:: pyruleanalyzer.compare_results

.. autoclass:: pyruleanalyzer.model_checker.Property
   :members:

.. autodata:: pyruleanalyzer.PROPERTY_CATALOG
   :no-value:

Occurrence graph
----------------

.. automodule:: pyruleanalyzer.cpn_statespace
   :members: ColouredNet, StateSpace, build_state_space, UnsupportedNet

Validation
----------

.. automodule:: pyruleanalyzer.cpntools_oracle
   :members: CPNToolsOracle, compare_with_oracle, OracleUnavailable

.. automodule:: pyruleanalyzer.cpn_mutants
   :members: make_mutants, CPNDocument, Mutant
