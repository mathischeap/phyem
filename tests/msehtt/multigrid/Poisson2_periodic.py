# -*- coding: utf-8 -*-
"""
mpiexec -n 4 python tests/msehtt/multigrid/Poisson2_periodic.py
"""
import numpy as np
import phyem as ph

N = 2
K = 4
c = 0

ls = ph.samples.wf_div_grad(n=2, degree=N, orientation='outer', periodic=True)[0]
# ls.pr()

msehtt, obj = ph.fem.apply('msehtt-smg', locals())
tgm = msehtt.tgm()
msehtt.config(tgm)(
    'crazy', element_layout=K, c=c, bounds=([0.25, 1.25], [0.25, 1.25]), periodic=True,
    mgc=3,
)

msehtt_mesh = msehtt.base['meshes'][r'\mathfrak{M}']
msehtt.config(msehtt_mesh)(tgm, including='all')
# tgm.visualize()

phi = msehtt.base['forms'][r'potential']
u = msehtt.base['forms'][r'velocity']
f = msehtt.base['forms'][r'source']


def phi_func(t, x, y):
    """"""
    return - np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y) + t * 0


phi.cf = ph.vc.scalar(phi_func)
phi[0].reduce()
# phi[0].visualize.quick()

u.cf = - phi.cf.codifferential()
f.cf = - u.cf.exterior_derivative()
f[0].reduce()
# f[0].visualize.quick()

msehtt_ls = obj['ls'].apply()

# msehtt_ls.pr()

linear_system = msehtt_ls(0)
linear_system.customize.set_dof(-1, phi[0].cochain.of_dof(-1))

Axb = linear_system.assemble()
# Axb.spy()
# print(Axb.condition_number, Axb.rank, Axb.num_singularities, Axb.shape)

x, message, info = Axb.solve('spsolve')
# print(message)

assert u[0].error() < 0.03
assert phi[0].error() < 0.005
