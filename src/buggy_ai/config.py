from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


ConfigDict = Dict[str, Any]


def load_config(path: str | Path) -> ConfigDict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
