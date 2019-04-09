
import numpy as np

# Parallelization
mesh = [12,32]

# Spatial discretization
N = 127
L_max = N
N_max = N
R_max = 3
alpha_BC = 0
L_dealias = 3/2
N_dealias = 3/2

# Physical parameters
R = np.sqrt(3) / 2
L = 1
dx = R / N
Prandtl = 1
Ra_danger = 20
Rayleigh = Ra_danger * Prandtl * (L / dx)**(8/3)
t_ff = (Rayleigh * Prandtl)**(-1/2)

# Volume penalization
ep_danger = 100
epsilon = ep_danger * (L / dx)**2
delta = dx / ep_danger**0.5

# Initial conditions
amp = 1e-3

# Temporal discretization
t_end = 100 * t_ff
dt_max = 1 / epsilon / 2
safety = 0.4
threshold = 0.1

# Outputs
snapshots_dir = 'rbc'
snapshots_cadence = t_ff/10


