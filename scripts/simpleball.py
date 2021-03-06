
import numpy as np
import dedalus.public as de
from dedalus_sphere import ball_wrapper


class SimpleBall:

    def __init__(self, radius, L_max, N_max, R_max, L_dealias, N_dealias, m_max=None, mesh=None, comm=None):
        if m_max is None:
            m_max  = L_max
        self.radius = radius
        self.L_max = L_max
        self.N_max = N_max
        self.R_max = R_max
        self.L_dealias = L_dealias
        self.N_dealias = N_dealias
        # Domain
        phi_basis = de.Fourier('phi', 2*(L_max+1), interval=(0, 2*np.pi), dealias=L_dealias)
        theta_basis = de.Chebyshev('theta', L_max+1, interval=(0, np.pi), dealias=L_dealias)
        r_basis = de.Chebyshev('r', N_max+1, interval=(0, 1), dealias=N_dealias)
        self.domain = domain = de.Domain([phi_basis, theta_basis, r_basis], grid_dtype=np.float64, mesh=mesh, comm=comm)
        # Local m
        mesh = self.domain.distributor.mesh
        if len(mesh) == 0:
            phi_layout = domain.distributor.layouts[3]
            th_m_layout = domain.distributor.layouts[2]
            ell_r_layout = domain.distributor.layouts[1]
            r_ell_layout = domain.distributor.layouts[1]
        elif len(mesh) == 1:
            phi_layout = domain.distributor.layouts[4]
            th_m_layout = domain.distributor.layouts[2]
            ell_r_layout = domain.distributor.layouts[1]
            r_ell_layout = domain.distributor.layouts[1]
        elif len(mesh) == 2:
            phi_layout = domain.distributor.layouts[5]
            th_m_layout = domain.distributor.layouts[3]
            ell_r_layout = domain.distributor.layouts[2]
            r_ell_layout = domain.distributor.layouts[1]
        self.m_start = th_m_layout.slices(scales=1)[0].start
        self.m_end = th_m_layout.slices(scales=1)[0].stop-1
        self.m_size = self.m_end - self.m_start + 1
        self.ell_start = r_ell_layout.slices(scales=1)[1].start
        self.ell_end = r_ell_layout.slices(scales=1)[1].stop-1
        # Ball wrapper
        N_theta = int((L_max+1)*L_dealias)
        N_r = int((N_max+1)*N_dealias)
        self.B = B = ball_wrapper.Ball(N_max, L_max, N_theta=N_theta, N_r=N_r, R_max=R_max,
            ell_min=self.ell_start, ell_max=self.ell_end, m_min=self.m_start, m_max=self.m_end, a=0.)
        # Grids
        grid_slices = phi_layout.slices(domain.dealias)
        self.phi = domain.grid(0, scales=domain.dealias)[grid_slices[0], :, :]
        self.theta = B.grid(1, dimensions=3)[:, grid_slices[1], :] # local
        self.r = radius * B.grid(2, dimensions=3)[:, :, grid_slices[2]] # local
        self.weight_theta = B.weight(1, dimensions=3)[:, grid_slices[1], :]
        self.weight_r = radius**3 * B.weight(2, dimensions=3)[:, :, grid_slices[2]]


class StateVector:

    def __init__(self, simpleball, tensors):
        self.simpleball = sb = simpleball
        self.tensors = tensors
        self.data = []
        for ell in range(sb.ell_start, sb.ell_end+1):
            if ell == 0:
                taus = np.zeros(1)
            else:
                taus = np.zeros(4)
            ell_local = ell - sb.ell_start
            for m in range(sb.m_start, sb.m_end+1):
                m_local = m - sb.m_start
                comps = [t['c'][ell_local][:, m_local] for t in tensors]
                comps.append(taus)
                self.data.append(np.concatenate(comps))

    def pack(self, tensors):
        sb = self.simpleball
        for ell in range(sb.ell_start, sb.ell_end+1):
            if ell == 0:
                taus = np.zeros(1)
            else:
                taus = np.zeros(4)
            ell_local = ell - sb.ell_start
            for m in range(sb.m_start, sb.m_end+1):
                m_local = m - sb.m_start
                comps = [t['c'][ell_local][:, m_local] for t in tensors]
                comps.append(taus)
                self.data[ell_local*sb.m_size+m_local] = np.concatenate(comps)

    def unpack(self, tensors):
        sb = self.simpleball
        for t in tensors:
            t.layout = 'c'
        for ell in range(sb.ell_start, sb.ell_end+1):
            ell_local = ell - sb.ell_start
            for m in range(sb.m_start, sb.m_end+1):
                m_local = m - sb.m_start
                ell_data = self.data[ell_local*sb.m_size + m_local]
                i0 = 0
                for t in tensors:
                    t_data = t['c'][ell_local][:, m_local]
                    i1 = i0 + t_data.size
                    t_data[:] = ell_data[i0:i1]
                    i0 = i1
