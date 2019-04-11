

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


def F_int_phi(hdf5_filename):
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
    # Integrate over phi
    F_int = F.sum('theta') * 2 * np.pi / theta.size
    return F_int


# Distribute sets in blocks
blocks = int(np.ceil((1 + last_set - first_set) / size))
start_set = first_set + rank * blocks
end_set = min(first_set + (rank + 1) * blocks, last_set + 1)
# Concatenate assigned set data
F_ints = []
for i in range(start_set, end_set):
    logger.info(f'Analyzing output {i}')
    hdf5_filename = '%s/%s_s%i.h5' %(dir, dir, i)
    F_ints.append(F_int_phi(hdf5_filename))
F_int = xr.concat(F_ints, 't')
# Gather and concatenate all sets
F_ints = comm.allgather(F_int)
F_int = xr.concat(F_ints, 't')
# Distribute interpolation points

print(F_int)
