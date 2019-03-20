
import h5py
import numpy as np
import pathlib
from mpi4py import MPI

import logging
logger = logging.getLogger(__name__.split('.')[-1])

class Sync:
    """
    Context manager for synchronizing MPI processes.

    Parameters
    ----------
    enter : boolean, optional
        Apply MPI barrier on entering context. Default: True
    exit : boolean, optional
        Apply MPI barrier on exiting context. Default: True

    """

    def __init__(self, comm=MPI.COMM_WORLD, enter=True, exit=True):
        self.comm = comm
        self.enter = enter
        self.exit = exit

    def __enter__(self):
        if self.enter:
            self.comm.Barrier()
        return self

    def __exit__(self, type, value, traceback):
        if self.exit:
            self.comm.Barrier()


class FileHandler:
    """
    Handler that writes tasks to an HDF5 file.

    Parameters
    ----------
    filename : str
        Base of filename, without an extension
    max_writes : int, optional
        Maximum number of writes per set (default: infinite)
    max_size : int, optional
        Maximum file size to write to, in bytes (default: 2**30 = 1 GB).
        (Note: files may be larger after final write.)

    """

    def __init__(self, base_path, domain, B, max_writes=np.inf, max_size=2**30, parallel=False, write_num=1, set_num=1, **kw):

        self.domain = domain
        self.B = B
        self.tasks = []

        # Check base_path
        base_path = pathlib.Path(base_path).absolute()
        if any(base_path.suffixes):
            raise ValueError("base_path should indicate a folder for storing HDF5 files.")
        with Sync(self.domain.distributor.comm_cart):
            if self.domain.distributor.rank == 0:
                if not base_path.exists():
                    base_path.mkdir()

        # Attributes
        self.base_path = base_path
        self.max_writes = max_writes
        self.max_size = max_size
        self.parallel = parallel
        self._sl_array = np.zeros(1, dtype=int)

        self.set_num = set_num - 1
        self.current_path = None
        self.total_write_num = write_num - 1
        self.file_write_num = 0

        if parallel:
            # Set HDF5 property list for collective writing
            self._property_list = h5py.h5p.create(h5py.h5p.DATASET_XFER)
            self._property_list.set_dxpl_mpio(h5py.h5fd.MPIO_COLLECTIVE)

    def add_task(self, field, layout='g', name='om', index=0, proj=None):
        """Add task to handler."""

        # Build task dictionary
        task = dict()
        task['field'] = field
        task['layout_name'] = layout
        task['layout'] = self.domain.distributor.get_layout_object(layout)
        task['name'] = name
        task['index'] = index
        task['proj'] = proj

        self.tasks.append(task)

    def check_file_limits(self):
        """Check if write or size limits have been reached."""

        write_limit = (self.file_write_num == self.max_writes)
        size_limit = (self.current_path.stat().st_size >= self.max_size)
        if not self.parallel:
            # reduce(size_limit, or) across processes
            comm = self.domain.distributor.comm_cart
            self._sl_array[0] = size_limit
            comm.Allreduce(MPI.IN_PLACE, self._sl_array, op=MPI.LOR)
            size_limit = self._sl_array[0]

        return (write_limit or size_limit)

    def get_file(self):
        """Return current HDF5 file, creating if necessary."""
        # Create file on first call
        if not self.current_path:
            return self.new_file()
        # Create file at file limits
        if self.check_file_limits():
            return self.new_file()
        # Otherwise open current file
        if self.parallel:
            comm = self.domain.distributor.comm_cart
            return h5py.File(str(self.current_path), 'a', driver='mpio', comm=comm)
        else:
            return h5py.File(str(self.current_path), 'a')

    def new_file(self):
        """Generate new HDF5 file."""

        domain = self.domain

        # Create next file
        self.set_num += 1
        self.file_write_num = 0
        comm = domain.distributor.comm_cart
        if self.parallel:
            # Save in base directory
            file_name = '%s_s%i.hdf5' %(self.base_path.stem, self.set_num)
            self.current_path = self.base_path.joinpath(file_name)
            file = h5py.File(str(self.current_path), 'w-', driver='mpio', comm=comm)
        else:
            # Save in folders for each filenum in base directory
            folder_name = '%s_s%i' %(self.base_path.stem, self.set_num)
            folder_path = self.base_path.joinpath(folder_name)
            with Sync(domain.distributor.comm_cart):
                if domain.distributor.rank == 0:
                    if not folder_path.exists():
                        folder_path.mkdir()
            file_name = '%s_s%i_p%i.h5' %(self.base_path.stem, self.set_num, comm.rank)
            self.current_path = folder_path.joinpath(file_name)
            file = h5py.File(str(self.current_path), 'w')

        self.setup_file(file)

        return file

    def setup_file(self, file):

        domain = self.domain

        # Metadeta
        file.attrs['set_number'] = self.set_num
        file.attrs['handler_name'] = self.base_path.stem
        file.attrs['writes'] = self.file_write_num
        if not self.parallel:
            file.attrs['mpi_rank'] = domain.distributor.comm_cart.rank
            file.attrs['mpi_size'] = domain.distributor.comm_cart.size

        # Scales
        scale_group = file.create_group('scales')
        # Start time scales with shape=(0,) to chunk across writes
        scale_group.create_dataset(name='sim_time', shape=(0,), maxshape=(None,), dtype=np.float64)
        scale_group.create_dataset(name='timestep', shape=(0,), maxshape=(None,), dtype=np.float64)
        scale_group.create_dataset(name='wall_time', shape=(0,), maxshape=(None,), dtype=np.float64)
        scale_group.create_dataset(name='iteration', shape=(0,), maxshape=(None,), dtype=np.int)
        scale_group.create_dataset(name='write_number', shape=(0,), maxshape=(None,), dtype=np.int)
        for axis, basis in enumerate(domain.bases):
            coeff_name = basis.element_label + basis.name
            scale_group.create_dataset(name=coeff_name, data=basis.elements)
            scale_group.create_group(basis.name)

        # Tasks
        task_group =  file.create_group('tasks')
        for task_num, task in enumerate(self.tasks):
            layout = task['layout']
            proj = task['proj']
            if proj != None:
                axis = proj[0]
                if axis == 'r': axis=2
                if axis == 'theta': axis=1
                if axis == 'phi': axis=0
                task['proj'] = (axis,proj[1])
            proj = task['proj']
            gnc_shape, gnc_start, write_shape, write_start, write_count = self.get_write_stats(layout, proj, index=0)
            if np.prod(write_shape) <= 1:
                # Start with shape[0] = 0 to chunk across writes for scalars
                file_shape = (0,) + tuple(write_shape)
            else:
                # Start with shape[0] = 1 to chunk within writes
                file_shape = (1,) + tuple(write_shape)
            file_max = (None,) + tuple(write_shape)
            dset = task_group.create_dataset(name=task['name'], shape=file_shape, maxshape=file_max, dtype=layout.dtype)
            if not self.parallel:
                dset.attrs['global_shape'] = gnc_shape
                dset.attrs['start'] = gnc_start
                dset.attrs['count'] = write_count

            # Metadata and scales
            dset.attrs['task_number'] = task_num
            dset.attrs['grid_space'] = layout.grid_space

            # Time scales
            dset.dims[0].label = 't'
            for sn in ['sim_time', 'wall_time', 'timestep', 'iteration', 'write_number']:
                scale = scale_group[sn]
                dset.dims.create_scale(scale, sn)
                dset.dims[0].attach_scale(scale)

            # Spatial scales
            if task_num == 0:
                scale_group['r'].create_dataset(name=str(self.domain.dealias[2]), data=self.B.radius)
            lookup = 'r/%s' %str(self.domain.dealias[2])
            scale = scale_group[lookup]
            dset.dims.create_scale(scale,lookup)
            dset.dims[1].label = 'r'
            dset.dims[1].attach_scale(scale)
            if task_num == 0:
                scale_group['theta'].create_dataset(name=str(self.domain.dealias[1]), data=self.B.S.grid)
            lookup = 'theta/%s' %str(self.domain.dealias[1])
            scale = scale_group[lookup]
            dset.dims.create_scale(scale,lookup)
            dset.dims[2].label = 'theta'
            dset.dims[2].attach_scale(scale)
            if task_num == 0:
                scale_group['phi'].create_dataset(name=str(self.domain.dealias[0]), data=self.domain.bases[0].grid(self.domain.dealias[0]))
            lookup = 'phi/%s' %str(self.domain.dealias[0])
            scale = scale_group[lookup]
            dset.dims.create_scale(scale,lookup)
            dset.dims[3].label = 'phi'
            dset.dims[3].attach_scale(scale)

    def process(self, wall_time, sim_time, timestep, iteration):
        """Save task outputs to HDF5 file."""

        file = self.get_file()
        self.total_write_num += 1
        self.file_write_num += 1
        file.attrs['writes'] = self.file_write_num
        index = self.file_write_num - 1

        # Update time scales
        sim_time_dset = file['scales/sim_time']
        wall_time_dset = file['scales/wall_time']
        timestep_dset = file['scales/timestep']
        iteration_dset = file['scales/iteration']
        write_num_dset = file['scales/write_number']

        sim_time_dset.resize(index+1, axis=0)
        sim_time_dset[index] = sim_time
        wall_time_dset.resize(index+1, axis=0)
        wall_time_dset[index] = wall_time
        timestep_dset.resize(index+1, axis=0)
        timestep_dset[index] = timestep
        iteration_dset.resize(index+1, axis=0)
        iteration_dset[index] = iteration
        write_num_dset.resize(index+1, axis=0)
        write_num_dset[index] = self.total_write_num

        # Create task datasets
        for task_num, task in enumerate(self.tasks):
            out = task['field']
            out.require_layout(task['layout_name'])
            dset = file['tasks'][task['name']]
            dset.resize(index+1, axis=0)

            proj = task['proj']

            if proj != None:
                layout = task['layout']

                lshape = layout.local_shape(self.domain.dealias)
                start = layout.start(self.domain.dealias)
                axis = proj[0]
                proj_index = proj[1]
                if (start[axis] <= proj_index) and (start[axis] + lshape[axis] > proj_index):
                    includes_data = True
                else:
                    includes_data = False

                if includes_data:
                    output = np.copy(out.data[task['index']])
                    shape = output.shape
                    if axis==0: output = np.copy(output[proj_index-start[axis]].reshape(1,shape[1],shape[2]))
                    if axis==1: output = np.copy(output[:,proj_index-start[axis]].reshape(shape[0],1,shape[2]))
                    if axis==2: output = np.copy(output[:,:,proj_index-start[axis]].reshape(shape[0],shape[1],1))
                else:
                    lshape_copy = np.copy(lshape)
                    lshape_copy[axis] = 0
                    output = np.zeros(lshape_copy)
            else:
                output = np.copy(out.data[task['index']])

            memory_space, file_space = self.get_hdf5_spaces(task['layout'], proj, index)
            if self.parallel:
                dset.id.write(memory_space, file_space, output, dxpl=self._property_list)
            else:
                dset.id.write(memory_space, file_space, output)
#                dset.id.write(memory_space, file_space, out.data[task['index']])

        file.close()

    def get_write_stats(self, layout, proj, index):
        """Determine write parameters for nonconstant subspace of a field."""

        # References
        gshape = layout.global_shape(self.domain.dealias)
        lshape = layout.local_shape(self.domain.dealias)
        start = layout.start(self.domain.dealias)
        if proj != None:
            axis = proj[0]
            proj_index = proj[1]
            if (start[axis] <= proj_index) and (start[axis] + lshape[axis] > proj_index):
                includes_data = True
            else:
                includes_data = False

        # Build counts, taking just the first entry along constant axes
        write_count = lshape.copy()
        if proj != None:
            if includes_data: write_count[axis] = 1
            else: write_count[axis] = 0

        # Collectively writing global data
        global_nc_shape = gshape.copy()
        if proj != None:
            global_nc_shape[axis] = 1
        global_nc_start = start.copy()
        if proj != None:
            if includes_data:
                global_nc_start[axis] = 0
            else:
                global_nc_start[axis] = 1

        if self.parallel:
            # Collectively writing global data
            write_shape = global_nc_shape
            write_start = global_nc_start
        else:
            # Independently writing local data
            write_shape = write_count
            write_start = 0 * start

        return global_nc_shape, global_nc_start, write_shape, write_start, write_count

    def get_hdf5_spaces(self, layout, proj, index):
        """Create HDF5 space objects for writing nonconstant subspace of a field."""

        # References
        lshape = layout.local_shape(self.domain.dealias)
        start = layout.start(self.domain.dealias)
        gnc_shape, gnc_start, write_shape, write_start, write_count = self.get_write_stats(layout, proj, index)

        # Build HDF5 spaces
        memory_shape = tuple(write_shape)
        memory_start = tuple(0 * start)
        memory_count = tuple(write_count)
        memory_space = h5py.h5s.create_simple(memory_shape)
        memory_space.select_hyperslab(memory_start, memory_count)

        file_shape = (index+1,) + tuple(write_shape)
        file_start = (index,) + tuple(write_start)
        file_count = (1,) + tuple(write_count)
        file_space = h5py.h5s.create_simple(file_shape)
        file_space.select_hyperslab(file_start, file_count)

        return memory_space, file_space
