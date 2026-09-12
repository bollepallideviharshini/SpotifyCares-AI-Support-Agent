"""
conftest.py — pytest configuration: adds src/ to sys.path so tests can
import modules directly without installing the package.
"""
import sys
from pathlib import Path

# Make src/ importable in all tests
sys.path.insert(0, str(Path(__file__).parent / "src"))
