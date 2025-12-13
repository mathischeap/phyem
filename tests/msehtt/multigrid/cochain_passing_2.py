r"""
mpiexec -n 4 python tests/msehtt/multigrid/cochain_passing_2.py
"""
import numpy as np
import phyem as ph

N = 2

ph.config.set_embedding_space_dim(2)
ph.config.set_high_accuracy(True)
ph.config.set_pr_cache(False)

manifold = ph.manifold(2, periodic=True)
mesh = ph.mesh(manifold)

Out0 = ph.space.new('Lambda', 0, orientation='outer')
Out1 = ph.space.new('Lambda', 1, orientation='outer')
Out2 = ph.space.new('Lambda', 2, orientation='outer')

Inn0 = ph.space.new('Lambda', 0, orientation='inner')
Inn1 = ph.space.new('Lambda', 1, orientation='inner')
Inn2 = ph.space.new('Lambda', 2, orientation='inner')

o0 = Out0.make_form(r'\tilde{\omega}^0', 'outer-form-0')
o1 = Out1.make_form(r'\tilde{\omega}^1', 'outer-form-1')
o2 = Out2.make_form(r'\tilde{\omega}^2', 'outer-form-2')

i0 = Inn0.make_form(r'{\omega}^0', 'inner-form-0')
i1 = Inn1.make_form(r'{\omega}^1', 'inner-form-1')
i2 = Inn2.make_form(r'{\omega}^2', 'inner-form-2')

ph.space.finite(N)

# ------------- implementation ---------------------------------------------------
msehtt, obj = ph.fem.apply('msehtt-smg', locals())

tgm = msehtt.tgm()

msehtt.config(tgm)('crazy', element_layout=6, c=0, mgc=2, periodic=True)
# tgm.visualize()

_mesh = obj['mesh']
msehtt.config(_mesh)(tgm, including='all')

# _mesh.visualize()

fo0 = obj['o0']
fo1 = obj['o1']
fo2 = obj['o2']

fi0 = obj['i0']
fi1 = obj['i1']
fi2 = obj['i2']


def fx(t, x, y):
    return np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y) * np.exp(t)


def fy(t, x, y):
    return np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y) * np.exp(t)


def fw(t, x, y):
    return np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y) * np.exp(t)


def f_norm(x, y):
    return fx(0, x, y)**2 + fy(0, x, y)**2


def fw0_square(x, y):
    return fw(0, x, y)**2


vector = ph.vc.vector(fx, fy)
scalar = ph.vc.scalar(fw)

fo0.cf = scalar
fi0.cf = scalar
fo1.cf = vector
fi1.cf = vector
fo2.cf = scalar
fi2.cf = scalar

fo0.get_level(1)[0].reduce()
tgm.pass_cochain(fo0.get_level(1), 0, fo0.get_level(0), 0, complete_only=True)
error0 = fo0.get_level(0)[0].error()

fi1.get_level(1)[0].reduce()
tgm.pass_cochain(fi1.get_level(1), 0, fi1.get_level(0), 0, complete_only=True)
error1o = fi1.get_level(0)[0].error()

fo1.get_level(1)[0].reduce()
tgm.pass_cochain(fo1.get_level(1), 0, fo1.get_level(0), 0, complete_only=True)
error1i = fo1.get_level(0)[0].error()

fi2.get_level(1)[0].reduce()
tgm.pass_cochain(fi2.get_level(1), 0, fi2.get_level(0), 0, complete_only=True)
error2 = fi2.get_level(0)[0].error()

error_array = np.array([error0, error1o, error1i, error2])
np.testing.assert_array_almost_equal(error_array, np.array([0.00467257, 0.02882566, 0.02882566, 0.02845538]))
