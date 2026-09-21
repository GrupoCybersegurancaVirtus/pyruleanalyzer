"""Backwards-compatible re-export: the CPN Tools oracle moved to CPNCheck.

Running the CPN Tools 4.0.1 simulator headlessly through Access/CPN, entering
the state-space tool and evaluating ASK-CTL is not specific to the models this
package generates, so it now lives in :mod:`cpncheck.oracle`.
"""

from cpncheck.oracle import (  # noqa: F401
    CPNToolsOracle,
    OracleUnavailable,
    compare_with_oracle,
)

__all__ = ["CPNToolsOracle", "OracleUnavailable", "compare_with_oracle"]
