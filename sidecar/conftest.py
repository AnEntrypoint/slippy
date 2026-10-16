import random

import pytest

from test_pipeline_cpu import StubBackbone, fake_seqs


@pytest.fixture(scope="module")
def backbone():
    return StubBackbone()


@pytest.fixture(scope="module")
def seqs():
    return fake_seqs(12, random.Random(5))
