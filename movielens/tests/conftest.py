import shutil
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT / "code"))
sys.path.insert(0, str(HERE))

from fixtures import make_movielens, make_tiny_model  # noqa: E402


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    return make_tiny_model(tmp_path_factory.mktemp("tiny_model"))


@pytest.fixture(scope="session")
def tiny_scorer(tiny_model_dir):
    from run import load_config
    from model import FrozenScorer

    cfg = load_config("movielens_1m", "smoke", [f"model.name={tiny_model_dir}", "model.device=cpu",
                                                "model.max_prompt_tokens=4096"], PROJECT)
    return FrozenScorer.load(cfg)


def make_root(tmp: Path) -> Path:
    """A self-contained project root: configs copied, synthetic raw data placed."""
    root = tmp / "proj"
    shutil.copytree(PROJECT / "configs", root / "configs")
    make_movielens(root / "data" / "raw" / "movielens_1m")
    return root


@pytest.fixture
def proj_root(tmp_path):
    return make_root(tmp_path)


def overrides(model_dir) -> list[str]:
    return [f"model.name={model_dir}", "model.device=cpu", "model.max_prompt_tokens=4096", "model.microbatch=4"]
