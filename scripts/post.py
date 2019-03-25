"""
Post-processing helpers.

"""

import pathlib
import h5py
import numpy as np
from mpi4py import MPI
import itertools
import shutil
import time
from concurrent.futures import wait

from mpi4py.futures import MPICommExecutor
from dedalus.tools.general import natural_sort
from dedalus.tools.array import axslice

import logging
logger = logging.getLogger(__name__.split('.')[-1])

MPI_RANK = MPI.COMM_WORLD.rank
MPI_SIZE = MPI.COMM_WORLD.size


def visit_writes(set_paths, function, **kw):
    """
    Apply function to writes from a list of analysis sets.

    Parameters
    ----------
    set_paths : list of str or pathlib.Path
        List of set paths
    function : function(set_path, start, count, **kw)
        A function on an HDF5 file, start index, and count.

    Other keyword arguments are passed on to `function`

    Notes
    -----
    This function is parallelized over writes, and so can be effectively
    parallelized up to the number of writes from all specified sets.

    """
    set_paths = natural_sort(str(sp) for sp in set_paths)
    arg_list = zip(set_paths, *get_assigned_writes(set_paths))
    for set_path, start, count in arg_list:
        if count:
            logger.info("Visiting set {} (start: {}, end: {})".format(set_path, start, start+count))
            function(set_path, start, count, **kw)


def get_assigned_writes(set_paths):
    """
    Divide writes from a list of analysis sets between MPI processes.

    Parameters
    ----------
    set_paths : list of str or pathlib.Path
        List of set paths

    """
    set_paths = natural_sort(str(sp) for sp in set_paths)
    # Distribute all writes in blocks
    writes = get_all_writes(set_paths)
    block = int(np.ceil(sum(writes) / MPI_SIZE))
    proc_start = MPI_RANK * block
    # Find set start/end indices
    writes = np.array(writes)
    set_ends = np.cumsum(writes)
    set_starts = set_ends - writes
    # Find proc start indices and counts for each set
    starts = np.clip(proc_start, a_min=set_starts, a_max=set_ends)
    counts = np.clip(proc_start+block, a_min=set_starts, a_max=set_ends) - starts
    return starts-set_starts, counts


def get_all_writes(set_paths):
    """
    Get write numbers from a list of analysis sets.

    Parameters
    ----------
    set_paths : list of str or pathlib.Path
        List of set paths

    """
    set_paths = natural_sort(str(sp) for sp in set_paths)
    writes = []
    for set_path in set_paths:
        with h5py.File(str(set_path), mode='r') as file:
            writes.append(file.attrs['writes'])
    return writes


def get_all_sets(base_path, distributed=False, wrap=False):
    """
    Divide analysis sets from a FileHandler between MPI processes.

    Parameters
    ----------
    base_path : str or pathlib.Path
        Base path of FileHandler output
    distributed : bool, optional
        Divide distributed sets instead of merged sets (default: False)
    wrap : bool, optional


    """
    base_path = pathlib.Path(base_path)
    base_stem = base_path.stem

    if distributed:
        set_paths = base_path.glob("{}_*".format(base_stem))
        set_paths = filter(lambda path: path.is_dir(), set_paths)
    else:
        set_paths = base_path.glob("{}_*.h5".format(base_stem))
    set_paths = natural_sort(set_paths)
    return set_paths


def get_assigned_sets(base_path, distributed=False, wrap=False):
    """
    Divide analysis sets from a FileHandler between MPI processes.

    Parameters
    ----------
    base_path : str or pathlib.Path
        Base path of FileHandler output
    distributed : bool, optional
        Divide distributed sets instead of merged sets (default: False)
    wrap : bool, optional


    """
    set_paths = get_all_sets(base_path, distributed=distributed, wrap=wrap)
    return set_paths[MPI_RANK::MPI_SIZE]


def merge_analysis(base_path, cleanup=False):
    """
    Merge distributed analysis sets from a FileHandler.

    Parameters
    ----------
    base_path : str or pathlib.Path
        Base path of FileHandler output
    cleanup : bool, optional
        Delete distributed files after merging (default: False)

    Notes
    -----
    This function is parallelized over sets, and so can be effectively
    parallelized up to the number of distributed sets.

    """
    set_path = pathlib.Path(base_path)
    logger.info("Merging files from {}".format(base_path))

    set_paths = get_assigned_sets(base_path, distributed=True)
    for set_path in set_paths:
        merge_distributed_set(set_path, cleanup=cleanup)


def tree_merge_analysis(base_path, cleanup=False):
    """
    Merge distributed analysis sets from a FileHandler.

    Parameters
    ----------
    base_path : str or pathlib.Path
        Base path of FileHandler output
    cleanup : bool, optional
        Delete distributed files after merging (default: False)

    Notes
    -----
    This function is parallelized over sets, and so can be effectively
    parallelized up to the number of distributed sets.

    """
    set_path = pathlib.Path(base_path)
    logger.info("Merging files from {}".format(base_path))

    with MPICommExecutor() as executor:
        if executor is not None:
            set_paths = get_all_sets(base_path, distributed=True)
            for set_path in set_paths:
                tree_merge_distributed_set(set_path, executor, cleanup=cleanup)


def merge_distributed_set(set_path, cleanup=False):
    """
    Merge a distributed analysis set from a FileHandler.

    Parameters
    ----------
    set_path : str of pathlib.Path
        Path to distributed analysis set folder
    cleanup : bool, optional
        Delete distributed files after merging (default: False)

    """
    set_path = pathlib.Path(set_path)
    logger.info("Merging set {}".format(set_path))

    set_stem = set_path.stem
    proc_paths = set_path.glob("{}_p*.h5".format(set_stem))
    proc_paths = natural_sort(proc_paths)
    joint_path = set_path.parent.joinpath("{}.h5".format(set_stem))

    # Create joint file, overwriting if it already exists
    with h5py.File(str(joint_path), mode='w') as joint_file:
        # Setup joint file based on first process file (arbitrary)
        merge_setup(joint_file, proc_paths[0])
        # Merge data from all process files
        for proc_path in proc_paths:
            merge_data(joint_file, proc_path)
    # Cleanup after completed merge, if directed
    if cleanup:
        for proc_path in proc_paths:
            proc_path.unlink()
        set_path.rmdir()


def tree_merge_distributed_set(set_path, executor, blocksize=2, cleanup=False):
    set_path = pathlib.Path(set_path)
    logger.info("Merging set {}".format(set_path))
    set_stem = set_path.stem
    # Get process mesh
    proc_path = set_path.joinpath(f"{set_stem}_p0.h5")
    with h5py.File(str(proc_path), mode='r') as proc_file:
        proc_dset = proc_file['tasks']['T']
        global_shape = np.array(proc_dset.attrs['global_shape'])
        local_shape = np.array(proc_dset.attrs['count'])
    mesh = np.ceil(global_shape/local_shape).astype(int)
    # Loop backwards over process mesh
    procs = np.arange(np.prod(mesh)).reshape(mesh)
    level = 0
    for D in reversed(range(len(mesh))):
        M = procs.shape[D]
        # Recursively merge blocks
        while M > 1:
            futures = []
            chunks = int(np.ceil(M / blocksize))
            proc_blocks = np.array_split(procs, chunks, axis=D)
            proc_blocks = [procs.reshape((-1, procs.shape[D])) for procs in proc_blocks]
            proc_blocks = [procs.tolist() for procs in proc_blocks]
            proc_blocks = list(itertools.chain(*zip(*proc_blocks)))
            for proc_block in proc_blocks:
                f = executor.submit(merge_level_procs, set_path, level, proc_block, D)
                futures.append(f)
            procs = procs[axslice(D, None, None, blocksize)]
            M = procs.shape[D]
            level += 1
            wait(futures)
    # Copy final output
    proc_path = set_path.joinpath(f"{set_stem}_p0_l{level}.h5")
    joint_path = set_path.parent.joinpath(f"{set_stem}.h5")
    shutil.copy(str(proc_path), str(joint_path))


def merge_level_procs(set_path, level, procs, axis, cleanup=False):
    logger.info(f"Merging level procs: {set_path} level {level} procs {procs} axis {axis}")
    set_path = pathlib.Path(set_path)
    set_stem = set_path.stem
    if level == 0:
        proc_paths = [set_path.joinpath(f"{set_stem}_p{proc}.h5") for proc in procs]
    else:
        proc_paths = [set_path.joinpath(f"{set_stem}_p{proc}_l{level}.h5") for proc in procs]
    joint_path = set_path.joinpath(f"{set_stem}_p{procs[0]}_l{level+1}.h5")
    # Create joint file, overwriting if it already exists
    with h5py.File(str(joint_path), mode='w') as joint_file:
        # Setup joint file based on first process file (arbitrary)
        merge_setup(joint_file, proc_paths[0], axis=axis, N=len(procs))
        # Merge data from all process files
        for proc_path in proc_paths:
            merge_data(joint_file, proc_path)
    # Cleanup after completed merge, if directed
    if cleanup:
        for proc_path in proc_paths:
            proc_path.unlink()


def merge_setup(joint_file, proc_path, axis=None, N=None):
    """
    Merge HDF5 setup from part of a distributed analysis set into a joint file.

    Parameters
    ----------
    joint_file : HDF5 file
        Joint file
    proc_path : str or pathlib.Path
        Path to part of a distributed analysis set

    """
    proc_path = pathlib.Path(proc_path)
    logger.info("Merging setup from {}".format(proc_path))

    with h5py.File(str(proc_path), mode='r') as proc_file:
        # File metadata
        try:
            joint_file.attrs['set_number'] = proc_file.attrs['set_number']
        except KeyError:
            joint_file.attrs['set_number'] = proc_file.attrs['file_number']
        joint_file.attrs['handler_name'] = proc_file.attrs['handler_name']
        try:
            joint_file.attrs['writes'] = writes = proc_file.attrs['writes']
        except KeyError:
            joint_file.attrs['writes'] = writes = len(proc_file['scales']['write_number'])
        # Copy scales (distributed files all have global scales)
        proc_file.copy('scales', joint_file)
        # Figure out joint shape
        # Tasks
        joint_tasks = joint_file.create_group('tasks')
        proc_tasks = proc_file['tasks']
        for taskname in proc_tasks:
            # Setup dataset with automatic chunking
            proc_dset = proc_tasks[taskname]
            # Spatial shape
            if axis is None:
                spatial_shape = proc_dset.attrs['global_shape']
            else:
                local_shape = np.array(proc_dset.attrs['count'])
                local_shape[axis] *= N
                global_shape = np.array(proc_dset.attrs['global_shape'])
                spatial_shape = np.minimum(local_shape, global_shape)
            joint_shape = (writes,) + tuple(spatial_shape)
            joint_dset = joint_tasks.create_dataset(name=proc_dset.name,
                                                    shape=joint_shape,
                                                    dtype=proc_dset.dtype,
                                                    chunks=True)
            # Dataset metadata
            joint_dset.attrs['global_shape'] = proc_dset.attrs['global_shape']
            joint_dset.attrs['start'] = proc_dset.attrs['start']
            joint_dset.attrs['count'] = spatial_shape
            joint_dset.attrs['task_number'] = proc_dset.attrs['task_number']
            #joint_dset.attrs['constant'] = proc_dset.attrs['constant']
            joint_dset.attrs['grid_space'] = proc_dset.attrs['grid_space']
            #joint_dset.attrs['scales'] = proc_dset.attrs['scales']
            # Dimension scales
            for i, proc_dim in enumerate(proc_dset.dims):
                joint_dset.dims[i].label = proc_dim.label
                for scalename in proc_dim:
                    scale = joint_file['scales'][scalename]
                    joint_dset.dims.create_scale(scale, scalename)
                    joint_dset.dims[i].attach_scale(scale)


def merge_data(joint_file, proc_path):
    """
    Merge data from part of a distributed analysis set into a joint file.

    Parameters
    ----------
    joint_file : HDF5 file
        Joint file
    proc_path : str or pathlib.Path
        Path to part of a distributed analysis set

    """
    proc_path = pathlib.Path(proc_path)
    logger.info("Merging data from {}".format(proc_path))

    with h5py.File(str(proc_path), mode='r') as proc_file:
        for taskname in proc_file['tasks']:
            joint_dset = joint_file['tasks'][taskname]
            proc_dset = proc_file['tasks'][taskname]
            # Merge across spatial distribution
            pstart = proc_dset.attrs['start']
            jstart = joint_dset.attrs['start']
            pcount = proc_dset.attrs['count']
            spatial_slices = tuple(slice(ps-js, ps-js+c) for (ps,js,c) in zip(pstart, jstart, pcount))
            # Merge maintains same set of writes
            slices = (slice(None),) + spatial_slices
            joint_dset[slices] = proc_dset[:]


def delete_analysis(base_path):
    """
    Delete distributed analysis sets from a FileHandler.

    Parameters
    ----------
    base_path : str or pathlib.Path
        Base path of FileHandler output

    Notes
    -----
    This function is parallelized over sets, and so can be effectively
    parallelized up to the number of distributed sets.

    """
    set_path = pathlib.Path(base_path)
    logger.info("Deleting files from {}".format(base_path))

    set_paths = get_assigned_sets(base_path, distributed=True)
    for set_path in set_paths:
        set_path = pathlib.Path(set_path)
        logger.info("Deleting set {}".format(set_path))

        set_stem = set_path.stem
        proc_paths = set_path.glob("{}_p*.h5".format(set_stem))
        proc_paths = natural_sort(proc_paths)

        for proc_path in proc_paths:
            proc_path.unlink()
        set_path.rmdir()
