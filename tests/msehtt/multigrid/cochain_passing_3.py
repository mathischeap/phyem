r"""
mpiexec -n 4 python tests/msehtt/multigrid/cochain_passing_3.py
"""
import numpy as np
import phyem as ph

N = 2
K = 5
levels = 2

ph.config.set_embedding_space_dim(3)
ph.config.set_high_accuracy(True)
ph.config.set_pr_cache(False)

manifold = ph.manifold(3, periodic=False)
mesh = ph.mesh(manifold)

S0 = ph.space.new('Lambda', 0, orientation='outer')
S1 = ph.space.new('Lambda', 1, orientation='outer')
S2 = ph.space.new('Lambda', 2, orientation='outer')
S3 = ph.space.new('Lambda', 3, orientation='outer')

f0 = S0.make_form(r'\tilde{\omega}^0', 'outer-form-0')
f1 = S1.make_form(r'\tilde{\omega}^1', 'outer-form-1')
f2 = S2.make_form(r'\tilde{\omega}^2', 'outer-form-2')
f3 = S3.make_form(r'\tilde{\omega}^3', 'outer-form-3')

ph.space.finite(N)

# ------------- implementation ---------------------------------------------------
msehtt, obj = ph.fem.apply('msehtt-smg', locals())

tgm = msehtt.tgm()
msehtt.config(tgm)(
    'crazy',
    element_layout=K,
    c=0,
    bounds=([0, 1], [0, 1], [0, 1]),
    periodic=True,
    mgc=levels,
)
# tgm.get_level().visualize()

_mesh = obj['mesh']
msehtt.config(_mesh)(tgm, including='all')
# _mesh.visualize.matplot()

F0 = obj['f0']
F1 = obj['f1']
F2 = obj['f2']
F3 = obj['f3']


def fx(t, x, y, z):
    return np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y) * np.cos(2 * np.pi * z) * np.exp(t)


def fy(t, x, y, z):
    return np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y) * np.sin(2 * np.pi * z) * np.exp(t)


def fz(t, x, y, z):
    return np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y) * np.sin(2 * np.pi * z) * np.exp(t)


def fw(t, x, y, z):
    return np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y) * np.cos(2 * np.pi * z) * np.exp(t)


vector = ph.vc.vector(fx, fy, fz)
scalar = ph.vc.scalar(fw)

F0.cf = scalar
F1.cf = vector
F2.cf = vector
F3.cf = scalar

benchmark_errors = [0.08, 0.04, 0.06, 0.04]
for i, F in enumerate([F0, F1, F2, F3]):
    F.get_level(1)[0].reduce()
    tgm.pass_cochain(F.get_level(1), 0, F.get_level(0), 0, complete_only=True)
    error0 = F.get_level(0)[0].error()
    assert error0 < benchmark_errors[i]
