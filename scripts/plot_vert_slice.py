"""
Plot planes from joint analysis files.

Usage:
    plot_2d_series.py <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./frames]

"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
from dedalus.extras import plot_tools
import xarray as xr
import h5py
from mpi4py import MPI
rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.size

def hdf5_to_xarray(h5_data):
    """Convert hdf5 dataset to xarray dataset."""
    data = h5_data[:]
    coords = [(dim.label, dim[0][:]) for dim in h5_data.dims]
    xr_data = xr.DataArray(data, coords=[coords[0], coords[3], coords[2], coords[1]])
    return xr_data

def get_data(filename, task, skip=1):
    # Load file
    file = h5py.File(filename, 'r')
    # Get data
    #return file['tasks'][task]
    xr = hdf5_to_xarray(file['tasks'][task])
    return xr

def extrap_full_sphere(X):
    # Extrapolate in theta
    X_0 = X.interp(theta=0, kwargs={'fill_value': 'extrapolate'})
    X_pi = X.interp(theta=np.pi, kwargs={'fill_value': 'extrapolate'})
    X = xr.concat([X_pi, X, X_0], dim='theta')
    # Extrapolate in r
    X_0 = X.interp(r=0, kwargs={'fill_value': 'extrapolate'})
    X = xr.concat([X_0, X], dim='r')
    return X

def main(filename):
    """Save plot of specified tasks for given range of analysis writes."""

    # Plot settings
    savename_func = lambda write: 'write_{:06}.png'.format(write)
    dpi = 200
    s = 1 / np.sqrt(3)
    plt.figure(figsize=(6,6))

    T = extrap_full_sphere(get_data(filename, 'T'))
    T.coords['z'] = T.coords['r'] * np.cos(T.coords['theta'])
    T.coords['x'] = T.coords['r'] * np.sin(T.coords['theta']) * np.cos(T.coords['phi'])
    background = T.coords['z']

    for i in range(T.shape[0]):
        dTi = T.isel(t=i) - background
        vmax = np.max(np.abs(dTi))
        dTi.isel(phi=0).plot(x='x', y='z', add_colorbar=False, vmin=-vmax, vmax=vmax, cmap='RdBu')
        dTi.isel(phi=192).plot(x='x', y='z', add_colorbar=False, vmin=-vmax, vmax=vmax, cmap='RdBu')
        plt.clf()
        plt.title('')
        plt.axis('equal')
        plt.axis('off')
        plt.savefig('frames/sphere_%03i.png' %i, dpi=dpi)
        line, = plt.plot([s,s,-s,-s,s], [-s,s,s,-s,-s], '--k', lw=1)
        plt.savefig('frames/spherebox_%03i.png' %i, dpi=dpi)
        line.set_visible(False)
        plt.xlim(-s, s)
        plt.ylim(-s, s)
        plt.savefig('frames/box_%03i.png' %i, dpi=dpi)

if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync

    args = docopt(__doc__)

    output_path = pathlib.Path(args['--output']).absolute()
    # Create output directory if needed
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()
    for file in args['<files>'][rank::size]:
        main(file)

