import os
import sys

from setuptools import setup, Extension

# ---------------------------------------------------------------------------
# C extension for tree traversal acceleration.
#
# The extension is optional -- if compilation fails (e.g. no C compiler on
# the target machine) the package still installs and falls back to a pure
# numpy implementation at runtime.
#
# The release wheels must not fall back silently: a wheel is tagged for its
# platform whether or not the extension was compiled into it. The release
# workflow sets PYRULEANALYZER_REQUIRE_EXTENSION=1, which turns a failed
# compilation into a failed build.
# ---------------------------------------------------------------------------

REQUIRE_EXTENSION = os.environ.get('PYRULEANALYZER_REQUIRE_EXTENSION') == '1'


# Function to locate numpy's C headers.
def get_numpy_include():
    """Get numpy include directory, required for building the C extension.

    Returns:
        str: The directory, or an empty string when numpy is missing.
    """
    try:
        import numpy as np
        return np.get_include()
    except ImportError:
        return ''


# Function to declare the C extension.
def build_extensions():
    """Return list of Extension objects, or empty list if numpy is missing.

    Returns:
        list: The extensions to build.

    Raises:
        RuntimeError: When the extension is required and numpy is missing.
    """
    np_inc = get_numpy_include()
    if not np_inc:
        if REQUIRE_EXTENSION:
            raise RuntimeError('numpy is needed to build the C extension')
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

        # Method to build every extension, tolerating failure unless required.
        def run(self):
            """Build the extensions.

            Raises:
                Exception: A build failure, when the extension is required.
            """
            try:
                super().run()
            except Exception:
                if REQUIRE_EXTENSION:
                    raise
                print(
                    '\n*** WARNING: C extension could not be built. '
                    'pyruleanalyzer will use the pure-numpy fallback for '
                    'tree traversal (slower but functional). ***\n',
                    file=sys.stderr,
                )

        # Method to build one extension, tolerating failure unless required.
        def build_extension(self, ext):
            """Build one extension.

            Args:
                ext (Extension): The extension.

            Raises:
                Exception: A build failure, when the extension is required.
            """
            try:
                super().build_extension(ext)
            except Exception:
                if REQUIRE_EXTENSION:
                    raise
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
