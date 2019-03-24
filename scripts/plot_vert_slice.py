"""
Plot planes from joint analysis files.

Usage:
    plot_vert_slices.py <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./frames]

"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import h5py
import xarray as xr

import logging
logger = logging.getLogger(__name__)


def hdf5_to_xarray(h5_data, slices=None):
    """Convert hdf5 dataset to xarray."""
    if slices is None:
        slices = (slice(None),) * len(h5_data.shape)
    data = h5_data[slices]
    coords = [(dim.label, dim[0][:]) for dim, s in zip(h5_data.dims, slices)]
    coords = [coords[0], coords[3], coords[2], coords[1]]
    coords = [(c[0], c[1][s]) for c,s in zip(coords, slices)]
    xr_data = xr.DataArray(data, coords=coords)
    return xr_data


def extrap_theta(X):
    X_0 = X.interp(theta=0, kwargs={'fill_value': 'extrapolate'})
    X_pi = X.interp(theta=np.pi, kwargs={'fill_value': 'extrapolate'})
    return xr.concat([X_0, X, X_pi], dim='theta')


def extrap_r(X):
    # Extrapolate in r
    X_0 = X.interp(r=0, kwargs={'fill_value': 'extrapolate'})
    return xr.concat([X_0, X], dim='r')


def extrap_phi(X):
    X_2pi = X.interp(phi=2*np.pi, kwargs={'fill_value': 'extrapolate'})
    return xr.concat([X, X_2pi], dim='phi')


def hdf5_vert_slice_to_xarray(X_hdf5, phi):
    # Get phi indeces
    N_phi = X_hdf5.shape[1]
    phi0 = int(np.round(phi / 2 / np.pi * N_phi))
    phi1 = int(np.round((phi + np.pi) / 2 / np.pi * N_phi))
    # Get phi slices
    slices = (slice(None), slice(phi0, phi0+1), slice(None), slice(None))
    X0 = hdf5_to_xarray(X_hdf5, slices)
    slices = (slice(None), slice(phi1, phi1+1), slice(None), slice(None))
    X1 = hdf5_to_xarray(X_hdf5, slices)
    # Concatenate and reorder
    X = xr.concat((X0, X1), dim='phi')
    X = X.sortby('phi')
    return X


def main(filename, output_path):

    logger.info('Plotting from file: %s' %filename)

    # Plot settings
    phi = 0
    dpi = 200
    cmap = 'RdBu_r'
    plt.figure(figsize=(6,6))

    # Load temperature perturbation from hdf5
    file = h5py.File(filename, 'r')
    T_hdf5 = file['tasks']['T']
    writes = file['scales']['write_number']

    # Load xarray slice
    T = hdf5_vert_slice_to_xarray(T_hdf5, phi)
    T = T.reindex(theta=T.theta[::-1])
    T = extrap_r(T)
    T = extrap_theta(T)

    # Add cartesian coords
    T.coords['x'] = T.coords['r'] * np.sin(T.coords['theta']) * np.cos(T.coords['phi'])
    T.coords['y'] = T.coords['r'] * np.sin(T.coords['theta']) * np.sin(T.coords['phi'])
    T.coords['z'] = T.coords['r'] * np.cos(T.coords['theta'])

    # Background temperature
    T0 = -T.coords['z']
    # Box half-side length
    s = 1 / np.sqrt(3)

    # Plot writes
    for i in range(T.shape[0]):
        logger.info('  Plotting from write: %i' %writes[i])
        plt.clf()
        # Plot total temperature
        Ti = T0 + T.isel(t=i)
        #vmax = np.max(np.abs(Ti))
        vmax = 1
        Ti.isel(phi=0).plot(x='x', y='z', add_colorbar=False, vmin=-vmax, vmax=vmax, cmap=cmap)
        Ti.isel(phi=1).plot(x='x', y='z', add_colorbar=False, vmin=-vmax, vmax=vmax, cmap=cmap)
        plt.title('')
        plt.axis('equal')
        plt.axis('off')
        plt.savefig(str(output_path.joinpath('sphere_%03i.png' %writes[i])), dpi=dpi)
        # Save with box
        ## WRONG FOR PHI != 0
        line, = plt.plot([s,s,-s,-s,s], [-s,s,s,-s,-s], '--k', lw=1)
        plt.savefig(str(output_path.joinpath('spherebox_%03i.png' %writes[i])), dpi=dpi)
        line.set_visible(False)
        # Save zoomed to box
        plt.xlim(-s, s)
        plt.ylim(-s, s)
        plt.savefig(str(output_path.joinpath('box_%03i.png' %writes[i])), dpi=dpi)

    file.close()


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
        main(file, output_path)

