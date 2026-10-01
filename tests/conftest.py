"""
pytest configuration for the Laguerre project test suite.

Adds the project root to sys.path so research-code imports (LMesh, Lconjugacy) work
without PYTHONPATH. hanan and kayviz are pip-installed (editable), and each
has its own test suite under hanan/tests and kayviz/tests.
"""

import sys
import os

_root    = os.path.dirname(os.path.dirname(__file__))   # project root
for _p in (_root,):
    if _p not in sys.path:
        sys.path.insert(0, _p)
