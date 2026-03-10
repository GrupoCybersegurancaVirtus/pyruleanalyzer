Overview
============================
**pyRuleAnalyzer** is a Python-based tool designed to support rule extraction, analysis, and simplification from tree-based artificial intelligence models. It provides a comprehensive pipeline to generate interpretable models, remove redundancies, and evaluate model accuracy and interpretability.

It currently supports three scikit-learn algorithms:

* **Decision Tree** (``DecisionTreeClassifier``)
* **Random Forest** (``RandomForestClassifier``)
* **Gradient Boosting Decision Trees** (``GradientBoostingClassifier``)

Key functionalities include:

* Extracting decision rules from tree-based models.
* Identifying and removing redundant and duplicate rules (intra-tree and inter-tree).
* High-performance vectorized batch prediction via compiled numpy arrays (with optional C acceleration).
* Evaluating initial and simplified models through classification accuracy, confusion matrices, and interpretability metrics.
* Computing the Structural Complexity Score (SCS) and interpretability metrics to assess model complexity.
* Exporting standalone classifiers to Python (``.py``), binary (``.bin``), and C header (``.h``) formats.
* Saving, loading, and applying rule-based classifiers on new datasets.
* Interactive terminal-based rule editing for manual refinement with domain knowledge.

.. toctree::
   :maxdepth: 1
   :hidden:

   Overview <self>
   installation
   tutorials/index
   api/index
   formal_verification
