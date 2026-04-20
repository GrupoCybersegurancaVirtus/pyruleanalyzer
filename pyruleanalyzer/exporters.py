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
