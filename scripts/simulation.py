
import os
import time
import numpy as np
from mpi4py import MPI
import dedalus.public as de
from dedalus.extras.flow_tools import GlobalArrayReducer

from dedalus_sphere import ball_wrapper as ball
from dedalus_sphere import timesteppers
from simpleball import SimpleBall, StateVector
from evaluator import FileHandler
import equations
import parameters as params

import logging
logger = logging.getLogger(__name__)


# Find MPI rank
comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

# Domain and fields
SB = SimpleBall(params.R, params.L_max, params.N_max, params.R_max, params.L_dealias, params.N_dealias, mesh=params.mesh)
B = SB.B
domain = SB.domain

u  = ball.TensorField_3D(1, B, domain)
p  = ball.TensorField_3D(0, B, domain)
T  = ball.TensorField_3D(0, B, domain)

u_rhs = ball.TensorField_3D(1, B, domain)
p_rhs = ball.TensorField_3D(0, B, domain)
T_rhs = ball.TensorField_3D(0, B, domain)
psi = ball.TensorField_3D(0, B, domain)

# Initial condition
r = SB.r
phi = SB.phi
theta = SB.theta

gshape = domain.dist.grid_layout.global_shape(scales=domain.dealias)
slices = domain.dist.grid_layout.slices(scales=domain.dealias)
rand = np.random.RandomState(seed=42)
T['g'] = params.amp * rand.standard_normal(gshape)[slices]

# Volume penalization mask
x = r * np.cos(phi) * np.sin(theta)
y = r * np.sin(phi) * np.sin(theta)
z = r * np.cos(theta)
Xinf = np.maximum(np.maximum(np.abs(x), np.abs(y)), np.abs(z))
psi['g'] = 0.5 + 0.5*np.tanh((Xinf - params.L/2) / params.delta)

# State vector
def SVWrap(*args):
    return StateVector(SB, args)

state_vector = StateVector(SB, [u, p, T])
NL = StateVector(SB, [u_rhs, p_rhs, T_rhs])
timestepper = timesteppers.SBDF2(SVWrap, u, p, T)

# Matrices
M, L, P, LU = [], [], [], []
for ell in range(SB.ell_start, SB.ell_end+1):
    logger.info('Building pencil ell = %i' %ell)
    N_ell = B.N_max - B.N_min(ell - B.R_max)
    M_ell, L_ell = equations.matrices(B, N_ell, ell, params.alpha_BC, params.R, params.Prandtl)
    M.append(M_ell.astype(np.complex128))
    L.append(L_ell.astype(np.complex128))
    P.append(M_ell.astype(np.complex128))
    LU.append([None])

# Output
if rank == 0:
    if not os.path.exists('{:s}/'.format(params.snapshots_dir)):
        os.mkdir('{:s}/'.format(params.snapshots_dir))

snapshots = FileHandler(params.snapshots_dir, domain, B, max_writes=1)
snapshots.add_task(T, name='T', index=0)
snapshots.add_task(u, name='u0', index=0)
snapshots.add_task(u, name='u1', index=1)
snapshots.add_task(u, name='u2', index=2)

# CFL
reducer = GlobalArrayReducer(domain.dist.comm_cart)
dr = np.gradient(r[0,0])
dangle = 1 / (params.L_max + 1)

def calculate_dt(dt_old):
    local_freq = u['g'][0]/dr + u['g'][1]/dangle + u['g'][2]/dangle
    global_freq = reducer.global_max(local_freq)
    if global_freq == 0.:
        dt = np.inf
    else:
        dt = 1 / global_freq
        dt *= params.safety
    if dt > params.dt_max:
        dt = params.dt_max
    if dt < dt_old*(1+params.threshold) and dt > dt_old*(1-params.threshold):
        dt = dt_old
    return dt

# Timestepping loop
start_time = time.time()
t = 0.
iter = 0
dt = params.dt_max
snapshots_time = -1e-20

while t < params.t_end:
    equations.nonlinear(state_vector, NL, t, M, params.R, params.Prandtl, params.Rayleigh, params.epsilon, psi)
    if iter % 10 == 0:
        Tmax = np.max(T['g'])
        Tmax = reducer.reduce_scalar(Tmax, MPI.MAX)
        logger.info("iter: {:d}, dt={:e}, t={:e}, T_max={:e}".format(iter, dt, t, Tmax))
    if t > snapshots_time:
        snapshots_time += params.snapshots_cadence
        snapshots.process(time.time(), t, dt, iter)
    if iter % 10 == 0:
        dt = calculate_dt(dt)
    timestepper.step(dt, state_vector, B, L, M, P, NL, LU)
    t += dt
    iter += 1

end_time = time.time()
logger.info('Simulation runtime: %f' %(end_time-start_time))

