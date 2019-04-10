
import numpy as np
import h5py
from simpleball import SimpleBall
from mpi4py import MPI
import parameters as params
from dedalus_sphere import ball128, sphere128, disk128
from dedalus_sphere import ball_wrapper
import dedalus.public as de

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

# Output parameters
first = 1
last = 1000
dir = params.snapshots_dir
output_filename = 'Nu.dat'

# Interpolation parameters
Nz = params.N + 1
Np = params.N + 1

def interp_zp(fc_m0, simpleball, z, p):
    """
    Interpolate ell-n data for m=s=0 to a point in (z, p).
    """
    B = simpleball.B
    # Get normalized spherical coordinates
    r_norm = np.sqrt(z**2 + p**2) / simpleball.radius
    z_proj = 2 * r_norm**2 - 1
    theta = np.arctan2(p, z)
    # Interpolate to r_norm
    fc_m0_r = []
    for ell in range(B.ell_min, B.ell_max+1):
        ell_local = ell - B.ell_min
        fc_m0_ell = fc_m0[ell_local]
        W = ball128.polynomial(B.N_max+B.R_max-B.N_min(ell), 0, ell, np.array([z_proj]), a=B.a)
        pullW = (W.T).astype(np.float64)
        N = B.N_max - B.N_min(ell-B.R_max) + 1
        interp = pullW[0,:N].dot(fc_m0_ell)
        fc_m0_r.append(interp)
    fc_m0_r = np.array(fc_m0_r)
    # Interpolate to theta
    Y = sphere128.Y(B.L_max, 0, 0, np.array([np.cos(theta)]))
    pullY = (Y.T).astype(np.float64)
    return pullY[0].dot(fc_m0_r)


def integrate_disks(field, simpleball, z_grid, p_grid, p_weights):
    """
    Integrate spherical scalar over disks.

    Parameters
    ----------
    field : scalar field
    simpleball : SimpleBall instance
    Z : array of z points for interpolation
    P : array of polar radius points for interpolation, scaled for unit disk
    WP : weights for polar radius quadrature, scaled for unit disk
    """
    R = simpleball.radius
    Nz = z_grid.size
    Np = p_grid.size
    output = np.zeros(Nz, dtype=np.float64)
    # Take m=0 component
    fc = field['c']
    fc_m0 = [fcl[:,0].real for fcl in fc]
    # Integrate level by level
    for i in range(Nz):
        zi = z_grid[i]
        # Rescale radial weights using disk radius
        Rz = np.sqrt(R**2 - zi**2)
        pz_weights = p_weights * Rz**2
        # Interpolate and integrate
        sum = 0
        for j in range(Np):
            fc_m0_zp = interp_zp(fc_m0, simpleball, zi, Rz*p_grid[j])
            sum = sum + pz_weights[j] * fc_m0_zp
        output[i] = sum
    return output

# Setup
SB = SimpleBall(params.R, params.L_max, params.N_max, params.R_max, params.L_dealias, params.N_dealias, comm=MPI.COMM_SELF)
weight_r = SB.weight_r

N_theta = params.L_dealias * (params.L_max + 1)
N_phi = 2 * N_theta

midp = int(N_theta / 2)
midm = midp - 1

# Preallocate flux field
F = ball_wrapper.TensorField_3D(0, SB.B, SB.domain)

# Vertical grid
z_basis = de.Chebyshev('z', Nz, interval=[-SB.radius, SB.radius])
z_grid = z_basis.grid(1)

# Normalized polar grid
pz_grid, p_weights = disk128.quadrature(Np-1, a=0, niter=3, report_error=False)
pz_grid = pz_grid.astype(np.float64)
p_grid = np.sqrt((1 + pz_grid) / 2)
p_weights = p_weights.astype(np.float64)
p_weights *= np.pi / np.sum(p_weights)

# Loop over outputs
t_list = []
Nu_list = []
Nu_profs = []

for i in range(first+rank, last+1, size):
    print(f'Analyzing output: {i}')

    filename = '%s/%s_s%i.h5' %(dir, dir, i)
    with h5py.File(filename, 'r') as f:
        for j in range(len(f['scales/sim_time'])):

            ## Nusselt midplane
            # Interpolate to midplane
            T_mid = 0.5 * (f['tasks/T'][j, :, midm, :] + f['tasks/T'][j, :, midp, :])
            w_mid = -0.5 * (f['tasks/u1'][j, :, midm, :] + f['tasks/u1'][j, :, midp, :])
            # Integrate over phi
            Nu = np.sum(w_mid * T_mid, axis=0) * 2 * np.pi / N_phi
            # Integrate over r
            Nu = np.sum(weight_r * Nu)
            # Normalize by box area
            Nu = Nu / params.L**3
            Nu_list.append(Nu)
            t_list.append(f['scales/sim_time'][j])

            ## Nusselt profiles
            T = f['tasks']['T'][j]
            ur = f['tasks']['u0'][j]
            utheta = f['tasks']['u1'][j]
            uz = ur * np.cos(SB.theta) - utheta * np.sin(SB.theta)
            # Integrate flux field
            F['g'] = T * uz
            F_int = integrate_disks(F, SB, z_grid, p_grid, p_weights)
            # Normalize by box area
            Nu_prof = F_int / params.L**2
            Nu_profs.append(Nu_prof)

# Collect from all processes
t = comm.gather(t_list, root=0)
Nu_mid = comm.gather(Nu_list, root=0)
Nu_prof = comm.gather(Nu_profs, root=0)

if rank == 0:
    # Combine outputs
    flatten = lambda list_of_lists: [item for list in list_of_lists for item in list]
    t = np.array(flatten(t))
    Nu_mid = np.array(flatten(Nu_mid))
    Nu_prof = np.array(flatten(Nu_prof))
    # Sort by time
    torder = np.argsort(t)
    t = t[torder]
    Nu_mid = Nu_mid[torder]
    Nu_prof = Nu_prof[torder]
    # Write to disk
    np.savez(output_filename, t=t, Nu_mid=Nu_mid, Nu_prof=Nu_prof, z_grid=z_grid)

