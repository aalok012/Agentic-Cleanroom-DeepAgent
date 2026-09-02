"""Exploratory pipeline arms.

Nothing in here is part of the clean-room pipeline. Modules under this package deliberately
relax guarantees the main pipeline enforces, so they live outside ``agents/deep/`` where the
AST guards in ``tests/test_isolation.py`` protect the real drivers.
"""
