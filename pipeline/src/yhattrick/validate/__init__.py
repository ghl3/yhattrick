"""Final validation stage: confirm the inference stage's invariants hold across the exported
artifacts. Reads what the pipeline produced (intermediate parquet, model tables, site JSON) and
checks the additive-theory relationships — without refitting anything. See `docs/modeling.md`.

Run: `uv run python -m yhattrick.validate` (or `make validate`)."""
from .core import Check, report, seasons_present
from .checks import run_all

__all__ = ["Check", "report", "run_all", "seasons_present"]
