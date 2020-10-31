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
    vals = scripts.runner.subspace(molecule, solver=scripts.odc12.simultaneous, test=True, comp_grad=False)[0]
    np.testing.assert_approx_equal(vals["energy"], -74.71370634648927, significant=10)
    np.testing.assert_allclose(vals["mu"], vals["deriv"], atol=1.0e-10)

def test_3():
    vals = scripts.runner.subspace(molecule, solver=scripts.odcD3.simultaneous, test=True, comp_grad=False)[0]
    np.testing.assert_approx_equal(vals["energy"], -74.71254983207328, significant=10)
    np.testing.assert_allclose(vals["mu"], vals["deriv"], atol=1.0e-10)

def test_4():
    vals = scripts.runner.subspace(molecule, solver=scripts.odcD4.simultaneous, test=True, comp_grad=False)[0]
    np.testing.assert_approx_equal(vals["energy"], -74.71357368209092, significant=10)
    np.testing.assert_allclose(vals["mu"], vals["deriv"], atol=1.0e-10)

def test_5():
    vals = scripts.runner.subspace(molecule, solver=scripts.odcD5.simultaneous, test=True, comp_grad=False)[0]
    np.testing.assert_approx_equal(vals["energy"], -74.71357831044938, significant=10)
    np.testing.assert_allclose(vals["mu"], vals["deriv"], atol=1.0e-10)

