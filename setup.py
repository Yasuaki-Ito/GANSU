"""
Minimal setup.py to override the bdist_wheel platform tag.

The Python side of GANSU is pure ctypes wrapping libgansu.so, so the wheel
should be tagged `py3-none-<platform>` (Python-version-agnostic, but
platform-specific because of the bundled .so / basis data).

setuptools defaults to `py3-none-any` for pure-Python projects, which would
let pip install the wheel on macOS/Windows where the bundled Linux .so cannot
load. We force the platform tag here so PyPI selects this wheel only on
matching Linux x86_64 systems.

All other metadata lives in pyproject.toml.
"""

from setuptools import setup
from setuptools.dist import Distribution

try:
    from setuptools.command.bdist_wheel import bdist_wheel
except ImportError:
    from wheel.bdist_wheel import bdist_wheel


class BinaryDistribution(Distribution):
    """Force platlib placement (Root-Is-Purelib: false in WHEEL metadata).

    Required so that auditwheel will accept the wheel — it refuses to repair
    a "purelib" wheel that contains an ELF .so file. Setting
    has_ext_modules() = True is the canonical way to flip the flag, even
    though our extension is loaded via ctypes rather than CPython's import
    machinery.
    """
    def has_ext_modules(self):
        return True


class PlatformBdistWheel(bdist_wheel):
    """Tag the wheel as `py3-none-<plat>` instead of `cp3xx-cp3xx-<plat>`.

    The package is pure ctypes around libgansu.so, so it's ABI-agnostic and
    one wheel covers every Python 3.x. Without this override the
    has_ext_modules=True flag above would force a Python-version-specific
    tag, requiring N wheels for N Python versions.
    """
    def get_tag(self):
        _, _, plat = super().get_tag()
        return ("py3", "none", plat)


setup(
    distclass=BinaryDistribution,
    cmdclass={"bdist_wheel": PlatformBdistWheel},
)
