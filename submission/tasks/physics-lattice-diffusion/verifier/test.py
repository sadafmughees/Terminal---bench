import pytest
import numpy as np

def test_lattice_diffusion_basic():
    # Sanity check for lattice diffusion task
    lattice_size = 16
    grid = np.zeros((lattice_size, lattice_size))
    assert grid.shape == (16, 16)
    assert np.all(grid == 0)
