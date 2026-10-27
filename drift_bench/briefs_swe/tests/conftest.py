"""Keep pytest from collecting the brief test files in place.

The ``<brief_id>_test.py`` files import ``solution``, which only exists in
the temp directory that ``swe_verify.run_tests`` builds. A bare ``pytest``
from the repository root would otherwise collect them and fail. The
harness copies a single test file into the temp directory, without this
conftest, so collection works there.
"""

collect_ignore_glob = ["*_test.py"]
