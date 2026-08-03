import os
import struct
import pickle
import sys
import time
import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict, Union, Tuple, Optional, Any
import pandas as pd

class RuleExporterMixin:
    """Mixin class for exporting RuleClassifier to different formats."""


    # Method to to python.
    def to_python(self, feature_names=None, filename="files/fast_classifier.py"):
        """
        Method to to python.
        
        Args:
            feature_names: Argument feature_names.
            filename: Argument filename.
            
        Returns:
            Any: Result of the operation.
        """
        return self.export_to_native_python(feature_names=feature_names, filename=filename)

    # Method to to c header.
    def to_c_header(self, filepath='model.h', guard_name='PYRULEANALYZER_MODEL_H'):
        """
        Method to to c header.
        
        Args:
            filepath: Argument filepath.
            guard_name: Argument guard_name.
            
        Returns:
            Any: Result of the operation.
        """
        return self.export_to_c_header(filepath=filepath, guard_name=guard_name)

    # Method to to binary.
    def to_binary(self, filepath='model.bin'):
        """
        Method to to binary.

        Args:
            filepath: Argument filepath.

        Returns:
            Any: Result of the operation.
        """
        return self.export_to_binary(filepath=filepath)

    # Method to to cpn tools.
    def to_cpn_tools(self, filepath='files/model.cpn', rules=None, feature_names=None,
                     sample=None, use_final=None, model_name='pyRuleAnalyzer'):
        """Export the classifier as a CPN Tools ``.cpn`` HCPN model.

        Produces a Hierarchical Coloured Petri Net in the native CPN Tools 4.x
        XML format that can be opened, visualised and simulated directly in
        `CPN Tools <https://cpntools.org>`_. Supports Decision Tree (Theorem 1),
        Gradient Boosting Decision Trees (Theorem 2, binary and multiclass) and
        Random Forest (majority voting) models.

        Args:
            filepath (str): Destination ``.cpn`` path.
            rules (list, optional): Explicit list of Rule objects to export.
                Defaults to the rule set selected by ``use_final``.
            feature_names (list, optional): Ordered feature names. Defaults to
                classifier metadata, then inference from the rules.
            sample (dict|list, optional): Example input sample used as the
                initial marking of the top-level input place so the net can be
                simulated immediately. Defaults to all zeros.
            use_final (bool, optional): If True export ``final_rules``; if False
                export ``initial_rules``; if None use final rules when present.
            model_name (str): Display name for the generated model / top page.

        Returns:
            str: The path of the written ``.cpn`` file.
        """
        from .cpn_tools_exporter import export_cpn_tools
        return export_cpn_tools(
            self, filepath, rules=rules, feature_names=feature_names,
            sample=sample, use_final=use_final, model_name=model_name,
        )
