"""Sanity check - was the template config file copied and set up already?"""

try:
    from cover_recognition.paths import (SPORT_COV_LARGE, SPORT_COV_SMALL,
                                         EEVA_COV_LARGE, FTEST_DIR)
except (ImportError, ModuleNotFoundError) as import_error:
    raise RuntimeError(
        "Copy `cover_recognition/paths.template.py` into `cover_recognition"
        "/paths.py` and set up the paths within before running this"
        " project!") from import_error
