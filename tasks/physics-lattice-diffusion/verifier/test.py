import pytest
import numpy as np

def test_lattice_diffusion_basic():
    lattice_size = 16
    grid = np.zeros((lattice_size, lattice_size))
    assert grid.shape == (16, 16)
    assert np.all(grid == 0)
