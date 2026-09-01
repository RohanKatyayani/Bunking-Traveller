import sys
from pathlib import Path

# app/ isn't a package (no __init__.py) and modules there import each other
# with plain `from rag_pipeline import ...`, so make it importable from tests too.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
