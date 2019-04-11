


import h5py
import xarray as xr


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



