import pathlib
from os import environ as os_env

LIB_DIR = pathlib.Path(__file__).resolve().parent

if "SHEETSAGE_CACHE_DIR" in os_env:
    CACHE_DIR = pathlib.Path(os_env["SHEETSAGE_CACHE_DIR"])
else:
    CACHE_DIR = pathlib.Path(pathlib.Path.home(), ".sheetsage")
CACHE_DIR = CACHE_DIR.resolve()

# NOTE: This changes the test discovery pattern from "test*.py" (default) to "*test.py".
def load_tests(loader, standard_tests, pattern):
    package_tests = loader.discover(start_dir=LIB_DIR, pattern="*test.py")
    standard_tests.addTests(package_tests)
    return standard_tests
