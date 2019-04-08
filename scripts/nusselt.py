
import numpy as np
import h5py
from simpleball import SimpleBall
from mpi4py import MPI
import parameters as params

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

start = 1
end = 1000
dir = params.snapshots_dir

SB = SimpleBall(params.R, params.L_max, params.N_max, params.R_max, params.L_dealias, params.N_dealias, mesh=params.mesh)
weight_r = SB.weight_r / SB.r

N_theta = params.L_dealias * (params.L_max + 1)
N_phi = 2 * N_theta

midp = int(N_theta / 2)
midm = midp - 1

t_list = []
Nu_list = []

for i in range(start+rank, end, size):

    filename = '%s/%s_s%i.h5' %(dir, dir, i)
    f = h5py.File(filename)

    for j in range(len(f['scales/sim_time'])):

        print(i,j)

        T = 0.5*(np.array(f['tasks/T'][j, :, midm, :]) + np.array(f['tasks/T'][j, :, midp, :]))
        w =-0.5*(np.array(f['tasks/u1'][j, :, midm, :]) + np.array(f['tasks/u1'][j, :, midp, :]))

        Nu = (params.R / params.L)**2 * np.sum(weight_r * w * T) * 2 * np.pi / N_phi
        Nu_list.append(Nu)
        t_list.append(f['scales/sim_time'][j])

    f.close()

print(Nu_list)
print(t_list)

t_list_list = comm.gather(t_list,root=0)
Nu_list_list = comm.gather(Nu_list,root=0)
if rank == 0:
    t_all = []
    Nu_all = []
    for (t_list,Nu_list) in zip(t_list_list,Nu_list_list):
        for (t, Nu) in zip(t_list,Nu_list):
            t_all.append(t)
            Nu_all.append(Nu)
    t = np.array(t_all)
    Nu = np.array(Nu_all)
    torder = np.argsort(t)
    t = t[torder]
    Nu = Nu[torder]

    data = np.array((t,Nu))
    np.savetxt('Nu.dat',data)

    print(t)
    print(Nu)

