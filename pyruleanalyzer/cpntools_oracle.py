"""The CPN Tools oracle, through CPNCheck.

Running the CPN Tools 4.0.1 simulator headlessly through Access/CPN, entering
the state-space tool and evaluating ASK-CTL is generic and lives in
:mod:`cpncheck.oracle`. This module keeps pyruleanalyzer's ``class_labels``
argument and makes sure the tree profile is registered before a net is read.
"""

from typing import Dict, Optional

from cpncheck.oracle import CPNToolsOracle, OracleUnavailable  # noqa: F401
from cpncheck.oracle import compare_with_oracle as _compare

from . import cpn_profile  # noqa: F401  (registers the tree profile)

__all__ = ["CPNToolsOracle", "OracleUnavailable", "compare_with_oracle"]


# Function to compare the model checker with CPN Tools on one generated net.
def compare_with_oracle(cpn_path: str, oracle: Optional[CPNToolsOracle] = None,
                        class_labels=None, timeout: int = 1800,
                        properties=None, options: Optional[Dict] = None):
    """CPNCheck's comparison, with the label domain as ``class_labels``.

    Args:
        cpn_path (str): The ``.cpn`` file.
        oracle (CPNToolsOracle, optional): The oracle.
        class_labels (list, optional): Label domain for A6.
        timeout (int): Seconds allowed for CPN Tools.
        properties (optional): User properties, compared too.
        options (dict, optional): Further profile settings.

    Returns:
        dict: The comparison (see :func:`cpncheck.oracle.compare_with_oracle`).
    """
    opts = dict(options or {})
    if class_labels is not None:
        opts["class_labels"] = list(class_labels)
    return _compare(cpn_path, oracle=oracle, timeout=timeout,
                    properties=properties, options=opts)
