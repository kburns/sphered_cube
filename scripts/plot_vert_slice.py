"""
Plot planes from joint analysis files.

Usage:
    plot_vert_slices.py <files>... [--output=<dir>] [--level=<lev>]

Options:
    --output=<dir>  Output directory [default: ./frames]
    --level=<lev>  Output directory [default: -1]

"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import h5py
import xarray as xr
try:
    import dask.array as da
except ImportError:
    print('Dask unavailable')
from dedalus.tools.array import axslice

import logging
logger = logging.getLogger(__name__)


def hdf5_to_xarray(h5_data, slices=None):
    """Convert hdf5 dataset to xarray."""
    if slices is None:
        slices = (slice(None),) * len(h5_data.shape)
    data = h5_data[slices]
    coords = [(dim.label, dim[0][:]) for dim, s in zip(h5_data.dims, slices)]
    coords = [coords[0], coords[3], coords[2], coords[1]]  # Swap r and phi
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


def joint_vert_slice_to_xarray(X_hdf5, phi):
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


def distributed_vert_slice_to_xarray(set_path, mesh, level, task, phi):
    view = DistributedSetViewer(set_path, mesh, level)
    X = view.load_task_xarray(task)
    print(X.shape)
    # Get phi indeces
    N_phi = X.coords['phi'].size
    phi0 = int(np.round(phi / 2 / np.pi * N_phi))
    phi1 = int(np.round((phi + np.pi) / 2 / np.pi * N_phi))
    # Get phi slices
    X0 = X.isel(phi=phi0)
    X1 = X.isel(phi=phi1)
    # Concatenate and reorder
    X = xr.concat((X0, X1), dim='phi')
    X = X.sortby('phi')
    return X


class DistributedSetViewer:

    def __init__(self, set_path, mesh, level=0, blocksize=2):
        self.set_path = set_path = pathlib.Path(set_path)
        set_stem = set_path.stem
        # Get process mesh
        procs = np.arange(np.prod(mesh)).reshape(mesh)
        D = len(mesh) - 1
        for i in range(level):
            if procs.shape[D] == 1:
                D = D - 1
            procs = procs[axslice(D, None, None, blocksize)]
        self.procs = procs
        # Load proc files
        self.proc_files = []
        for proc in self.procs.ravel():
            if level == 0:
                proc_path = set_path.joinpath(f"{set_stem}_p{proc}.h5")
            else:
                proc_path = set_path.joinpath(f"{set_stem}_p{proc}_l{level}.h5")
            proc_file = h5py.File(str(proc_path), 'r')
            self.proc_files.append(proc_file)

    def load_task_daskarray(self, task, index=None, chunks=None):
        # Load proc datasets
        proc_dsets = np.empty(len(self.proc_files), dtype=object)
        for i, proc_file in enumerate(self.proc_files):
            # Load dataset
            dset = proc_file['tasks'][task]
            # Cast to dask array
            if chunks is None:
                chunks = dset.chunks
            dset = da.from_array(dset, chunks=chunks)
            if index is not None:
                dset = dset[index]
            proc_dsets[i] = dset
        # Shape into nested list
        proc_dsets = proc_dsets.reshape(self.procs.shape)
        proc_dsets = proc_dsets.tolist()
        # Build using dask blocking
        dset = da.block(proc_dsets)
        return dset

    def load_task_xarray(self, task, **kw):
        # Get coords
        scales = self.proc_files[0]['scales']
        coords = {'t': scales['sim_time'],
                  'sim_time': ('t', scales['sim_time']),
                  'timestep': ('t', scales['timestep']),
                  'wall_time': ('t', scales['wall_time']),
                  'write_number': ('t', scales['write_number']),
                  'iteration': ('t', scales['iteration']),
                  'r': scales['r/1.5'],
                  'theta': scales['theta/1.5'],
                  'phi': scales['phi/1.5']}
        # Cask from dask to xarray
        dset = self.load_task_daskarray(task, **kw)
        return xr.DataArray(dset, dims=['t', 'phi', 'theta', 'r'], coords=coords)


def main(filename, output_path, level=-1):

    logger.info('Plotting from file: %s' %filename)

    # Plot settings
    mesh = [2, 2]
    phi = 0
    dpi = 200
    cmap = 'RdBu_r'
    plt.figure(figsize=(6,6))
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)

    # Load xarray slice
    if level >= 0:
        T = distributed_vert_slice_to_xarray(filename, mesh, level, 'T', phi)
        writes = T.coords['write_number'].data
    else:
        # Load temperature perturbation from hdf5
        with h5py.File(filename, 'r') as file:
            T = joint_vert_slice_to_xarray(file['tasks']['T'], phi)
            writes = file['scales']['write_number'][:]

    # Reindex and extrapolate
    T = T.reindex(theta=T.theta[::-1])
    T = extrap_r(T)
    T = extrap_theta(T)

    # Add cartesian coords
    T.coords['x'] = T.coords['r'] * np.sin(T.coords['theta']) * np.cos(T.coords['phi'] - phi)
    T.coords['y'] = T.coords['r'] * np.sin(T.coords['theta']) * np.sin(T.coords['phi'] - phi)
    T.coords['z'] = T.coords['r'] * np.cos(T.coords['theta'])

    # Background temperature
    T0 = -T.coords['z']
    # Box half-side length
    sz = 1 / np.sqrt(3)
    # Box width at phi
    phi_mod = ((np.pi/4 + phi) % (np.pi/2)) - np.pi/4
    sx = sz / np.cos(phi_mod)
    # Plot padding
    δ = 0.01

    # Plot writes
    for i in range(T.coords['t'].size):
        logger.info('  Plotting from write: %i' %writes[i])
        plt.clf()
        # Plot total temperature
        Ti = T0 + T.isel(t=i)
        vmax = 1
        Ti.isel(phi=0).plot(x='x', y='z', add_colorbar=False, vmin=-vmax, vmax=vmax, cmap=cmap)
        Ti.isel(phi=1).plot(x='x', y='z', add_colorbar=False, vmin=-vmax, vmax=vmax, cmap=cmap)
        plt.title('')
        plt.xlim(-1-δ, 1+δ)
        plt.ylim(-1-δ, 1+δ)
        plt.axis('off')
        plt.savefig(str(output_path.joinpath('sphere_%06i.png' %writes[i])), dpi=dpi)
        # Save with box
        line, = plt.plot([sx,sx,-sx,-sx,sx], [-sz,sz,sz,-sz,-sz], '--k', lw=1, alpha=0.5)
        plt.savefig(str(output_path.joinpath('spherebox_%06i.png' %writes[i])), dpi=dpi)
        line.set_visible(False)
        # Save zoomed to box
        plt.fill_between([-1.1, 1.1], [sz, sz], [1.1, 1.1], color='white')
        plt.fill_between([-1.1, 1.1], [-sz, -sz], [-1.1, -1.1], color='white')
        plt.fill_between([-1.1, -sx], [-1.1, -1.1], [1.1, 1.1], color='white')
        plt.fill_between([sx, 1.1], [-1.1, -1.1], [1.1, 1.1], color='white')
        plt.xlim(-1-δ, 1+δ)
        plt.ylim(-1-δ, 1+δ)
        plt.savefig(str(output_path.joinpath('box_%06i.png' %writes[i])), dpi=dpi)


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
    level = int(args['--level'])

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
        main(file, output_path, level)

