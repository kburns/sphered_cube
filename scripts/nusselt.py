
import numpy as np
import h5py
from dedalus_sphere import ball128 as ball
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

start = 1
end = 55
dir = 'rbc_hres'
L_max = 255
L_dealias=3/2

z, weight_r = ball.quadrature(int((L_max+1)*L_dealias-1),a=0.0)
weight_r = weight_r[None,:]

midp = int((L_max+1)*L_dealias/2)
midm = midp-1

t_list = []
Nu_list = []

for i in range(start+rank,end,size):

  filename = '%s/%s_s%i.h5' %(dir,dir,i)
  f = h5py.File(filename)

  for j in range(len(f['scales/sim_time'])):

    print(i,j)

    T = 0.5*(np.array(f['tasks/T'][j,:,midm,:]) + np.array(f['tasks/T'][j,:,midp,:]))
    w =-0.5*(np.array(f['tasks/uth'][j,:,midm,:]) + np.array(f['tasks/uth'][j,:,midp,:]))

    Nu = np.sum(w*T*weight_r*(np.pi)/((L_max+1)*L_dealias))
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

