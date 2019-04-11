

import math
import numpy as np
import h5py
import xarray as xr
from mpi4py import MPI
from data_conversion import hdf5_to_xarray
import dedalus.public as de

import logging
logger = logging.getLogger(__name__)


comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size


# Data parameters
first_set = 1
last_set = 11
dir = 'rbc'
output_filename = 'Nu.dat'


def load_flux_xarray(hdf5_filename):
    """
    Load HDF5. Compute flux. Integrate over phi. Convert to xarray.
    """
    # Load data
    with h5py.File(hdf5_filename, 'r') as file:
        T = hdf5_to_xarray(file['tasks']['T'])
        ur = hdf5_to_xarray(file['tasks']['u0'])
        utheta = hdf5_to_xarray(file['tasks']['u1'])
    # Compute flux
    theta = T.coords['theta']
    uz = ur * np.cos(theta) - utheta * np.sin(theta)
    F = T * uz
    return F_int


## Integrate flux over phi
# Distribute sets in blocks
n_sets = last_set - first_set + 1
F_ints = []
for i in block_distributed_range(n_sets, comm)
    i += first_set
    # Load flux
    logger.info(f'Loading flux {i}')
    hdf5_filename = '%s/%s_s%i.h5' %(dir, dir, i)
    F = load_flux_xarray(hdf5_filename)
    # Integrate over phi
    F_int['g'] = F.mean('theta').data * 2 * np.pi
    # Transform to coeff space
    F['g']
    F_ints.append(F_int)
## Interpolate to r and p
# Gather and concatenate all sets over time
F_int = xr.concat(F_ints, 't')
F_ints = comm.allgather(F_int)
F_int = xr.concat(F_ints, 't')
# Distribute interpolation
zp_pairs = list(itertools.product(z_grid, p_grid))
zp_interp = []
for i in block_distributed_range(zp_pairs.size, comm):
    z, p = zp_pairs[i]
    interp = None
    zp_interp.append(interp)
# Gather interpolation
zp_interp = comm.gather(zp_interp, root=0)
if comm.rank == 0:
    zp_interp = np.array(zp_interp).reshape((z_grid.size, p_grid.size))


def block_distributed_range(size, comm):
    """Block distribute an integer range."""
    # Basic block distribution
    block = math.ceil(size / comm.size)
    start = comm.rank * block
    end = start + block
    # Avoid running over the end
    start = min(start, size)
    end = min(end, size)
    # Return local iterator
    return range(start, end)

