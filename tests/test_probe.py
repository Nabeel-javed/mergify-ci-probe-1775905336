import os
import random


def test_stable_passes():
    assert 2 + 2 == 4


def test_flaky_signal():
    # Produce a mix of pass/fail outcomes across repeated runs to seed CI Insights.
    seed = f"{os.getenv('GITHUB_RUN_ID', 'local')}:{os.getenv('GITHUB_RUN_ATTEMPT', '0')}:{random.random()}"
    value = sum(ord(c) for c in seed) % 5
    assert value != 0
