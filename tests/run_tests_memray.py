import sys
import pytest

sys.exit(pytest.main([
    "tests/",
    "--cov=src/ampiimts",
    "--cov-report=term-missing",
    "--cov-branch",
    "--memray"
]))
