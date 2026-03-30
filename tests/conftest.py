"""Shared fixtures for body measurement tests.

Usage:
    pytest tests/ -v              # no renders
    pytest tests/ -v --view       # save 4-view PNGs to tests/results/
"""

import os

import pytest

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def pytest_addoption(parser):
    parser.addoption(
        "--view", action="store_true", default=False,
        help="Render 4-view PNGs to body/tests/results/",
    )


@pytest.fixture
def view(request):
    return request.config.getoption("--view")
