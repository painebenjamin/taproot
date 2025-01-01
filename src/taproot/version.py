from .util.package_util import get_installed_package_version
try:
    version = get_installed_package_version("taproot")
except:
    from packaging.version import Version
    version = Version('0.0.0')
