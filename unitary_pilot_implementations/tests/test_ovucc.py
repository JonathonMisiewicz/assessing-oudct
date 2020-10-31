import numpy as np
from unitary_pilot_implementations import scripts

np.set_printoptions(precision=13, linewidth=200, suppress=True)

molecule = {
    "charge": +1,
    "num_unpaired": +1,
    "geom": [
         ('O', (0.000000000000, 0.000000000000, -0.143225816552)),
         ('H', (0.000000000000, 1.638036840407, 1.136548822547)),
         ('H', (0.000000000000, -1.638036840407, 1.136548822547))
         ],
    "basis": "sto-3g"
}

def test_2():
    vals = scripts.runner.subspace(molecule, solver=scripts.D2.simultaneous, test=True, comp_grad=False)[0]
    np.testing.assert_approx_equal(vals["energy"], -74.71451994537853, significant=10)
    np.testing.assert_allclose(vals["mu"], vals["deriv"], atol=1.0e-10)

def test_3():
    vals = scripts.runner.subspace(molecule, solver=scripts.D3.simultaneous, test=True, comp_grad=False)[0]
    np.testing.assert_approx_equal(vals["energy"], -74.7132519570999, significant=10)
    np.testing.assert_allclose(vals["mu"], vals["deriv"], atol=1.0e-10)

def test_4():
    vals = scripts.runner.subspace(molecule, solver=scripts.D4.simultaneous, test=True, comp_grad=False)[0]
    np.testing.assert_approx_equal(vals["energy"], -74.71356449866714, significant=10)
    np.testing.assert_allclose(vals["mu"], vals["deriv"], atol=1.0e-10)

def test_5():
    vals = scripts.runner.subspace(molecule, solver=scripts.D5.simultaneous, test=True, comp_grad=False)[0]
    np.testing.assert_approx_equal(vals["energy"], -74.71356911296462, significant=10)
    np.testing.assert_allclose(vals["mu"], vals["deriv"], atol=1.0e-10)

