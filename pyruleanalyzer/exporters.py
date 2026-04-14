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


    def to_python(self, feature_names=None, filename="files/fast_classifier.py"):
        return self.export_to_native_python(feature_names=feature_names, filename=filename)

    def to_c_header(self, filepath='model.h', guard_name='PYRULEANALYZER_MODEL_H'):
        return self.export_to_c_header(filepath=filepath, guard_name=guard_name)

    def to_binary(self, filepath='model.bin'):
        return self.export_to_binary(filepath=filepath)
