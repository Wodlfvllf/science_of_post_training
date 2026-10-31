"""Keep tiny random CPU model tests fast without changing inference code's policy."""

import pytest


@pytest.fixture(scope="session", autouse=True)
def small_cpu_threads():
    import torch

    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(old)