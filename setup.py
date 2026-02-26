import os
import sys

from setuptools import setup, Extension

# ---------------------------------------------------------------------------
# C extension for tree traversal acceleration.
#
# The extension is optional -- if compilation fails (e.g. no C compiler on
# the target machine) the package still installs and falls back to a pure
# numpy implementation at runtime.
# ---------------------------------------------------------------------------

def get_numpy_include():
    """Get numpy include directory, required for building the C extension."""
    try:
        import numpy as np
        return np.get_include()
    except ImportError:
        return ''


def build_extensions():
    """Return list of Extension objects, or empty list if numpy is missing."""
    np_inc = get_numpy_include()
    if not np_inc:
        return []

    ext = Extension(
        'pyruleanalyzer._tree_traversal',
        sources=[os.path.join('pyruleanalyzer', '_tree_traversal.c')],
        include_dirs=[np_inc],
        language='c',
    )
    return [ext]


# Allow graceful fallback: if compilation fails, install without the extension
try:
    from setuptools.command.build_ext import build_ext as _build_ext

    class BuildExtOptional(_build_ext):
        """Build C extensions but don't fail if they can't be compiled."""

        def run(self):
            try:
                super().run()
            except Exception:
                print(
                    '\n*** WARNING: C extension could not be built. '
                    'pyruleanalyzer will use the pure-numpy fallback for '
                    'tree traversal (slower but functional). ***\n',
                    file=sys.stderr,
                )

        def build_extension(self, ext):
            try:
                super().build_extension(ext)
            except Exception:
                print(
                    f'\n*** WARNING: Failed to build extension {ext.name}. '
                    f'Falling back to pure-numpy implementation. ***\n',
                    file=sys.stderr,
                )

    cmdclass = {'build_ext': BuildExtOptional}
except ImportError:
    cmdclass = {}

setup(
    ext_modules=build_extensions(),
    cmdclass=cmdclass,
)
