import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

@pytest.fixture
def data_dir():
    return DATA_DIR

@pytest.fixture
def output_dir(tmp_path):
    """Use temp dir for test outputs to avoid polluting real output/."""
    return tmp_path
