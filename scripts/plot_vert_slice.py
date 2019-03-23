"""
Plot planes from joint analysis files.

Usage:
    plot_vert_slices.py <files>... [--noextrap] [--output=<dir>]

Options:
    --noextrap      Don't extrapolate to coordinate endpoints
    --output=<dir>  Output directory [default: ./frames]

"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ioff()
import h5py
import xarray as xr

import logging
logger = logging.getLogger(__name__)


def hdf5_to_xarray(h5_data):
    """Convert hdf5 dataset to xarray."""
    data = h5_data[:]
    coords = [(dim.label, dim[0][:]) for dim in h5_data.dims]
    xr_data = xr.DataArray(data, coords=[coords[0], coords[3], coords[2], coords[1]])
    return xr_data

def load_task_xarray(filename, task, skip=1):
    """Load saved task as xarray."""
    # Load file
    file = h5py.File(filename, 'r')
    # Get data
    xr = hdf5_to_xarray(file['tasks'][task])
    return xr

def extrap_full_sphere(X):
    """Extrapolate spherical grid data to singularities."""
    # Extrapolate in theta
    X_0 = X.interp(theta=0, kwargs={'fill_value': 'extrapolate'})
    X_pi = X.interp(theta=np.pi, kwargs={'fill_value': 'extrapolate'})
    X = xr.concat([X_pi, X, X_0], dim='theta')
    # Extrapolate in r
    X_0 = X.interp(r=0, kwargs={'fill_value': 'extrapolate'})
    X = xr.concat([X_0, X], dim='r')
    return X


def main(filename, output_path, extrapolate=True):

    logger.info('Plotting from file: %s' %filename)

    # Plot settings
    dpi = 200
    cmap = 'RdBu'
    plt.figure(figsize=(6,6))

    # Load temperature perturbation
    T = load_task_xarray(filename, 'T')
    if extrapolate:
        T = extrap_full_sphere(T)

    # Add cartesian coords
    T.coords['x'] = T.coords['r'] * np.sin(T.coords['theta']) * np.cos(T.coords['phi'])
    T.coords['y'] = T.coords['r'] * np.sin(T.coords['theta']) * np.sin(T.coords['phi'])
    T.coords['z'] = T.coords['r'] * np.cos(T.coords['theta'])

    # Background temperature
    T0 = -T.coords['z']
    # Box half-side length
    s = 1 / np.sqrt(3)
    # Azimuthal selections
    phi0 = 0
    phi1 = int(T.coords['phi'].size / 2)

    for i in range(T.shape[0]):
        logger.info('  Plotting from write: %i' %i)
        plt.clf()
        # Plot total temperature
        Ti = T0 + T.isel(t=i)
        vmax = np.max(np.abs(Ti))
        Ti.isel(phi=phi0).plot(x='x', y='z', add_colorbar=False, vmin=-vmax, vmax=vmax, cmap=cmap)
        Ti.isel(phi=phi1).plot(x='x', y='z', add_colorbar=False, vmin=-vmax, vmax=vmax, cmap=cmap)
        plt.title('')
        plt.axis('equal')
        plt.axis('off')
        plt.savefig(str(output_path.joinpath('sphere_%03i.png' %i)), dpi=dpi)
        # Save with box
        line, = plt.plot([s,s,-s,-s,s], [-s,s,s,-s,-s], '--k', lw=1)
        plt.savefig(str(output_path.joinpath('spherebox_%03i.png' %i)), dpi=dpi)
        line.set_visible(False)
        # Save zoomed to box
        plt.xlim(-s, s)
        plt.ylim(-s, s)
        plt.savefig(str(output_path.joinpath('box_%03i.png' %i)), dpi=dpi)


if __name__ == "__main__":

    import pathlib
    from mpi4py import MPI
    from docopt import docopt
    from dedalus.tools.parallel import Sync
    from dedalus.tools import logging

    # Processes arguments
    args = docopt(__doc__)
    files = args['<files>']
    output_path = pathlib.Path(args['--output']).absolute()
    extrapolate = not args['--noextrap']

    # Create output directory if needed
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()

    # Distribute and loop over files
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
    local_files = files[rank::size]
    for file in local_files:
        main(file, output_path, extrapolate)

