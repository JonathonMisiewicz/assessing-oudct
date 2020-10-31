import numpy as np
from unitary_pilot_implementations import scripts

# That the auto-generated projective code, which uses most of the same auto-generated machinery as the variational code
# matches 10.1063/1.3598471, Table VI is an important accuracy check.
# This code WILL not run with out-of-the-box Psi4. You need to add the basis set from Section II of the Supporting Material
# to Psi4's basis set library. I've included it in this directory.

molecule = {
    "charge": 0,
    "num_unpaired": 0,
    "geom": [
        ('Be', (0, 0, 0)),
        ('H', (2, +(2.54-0.46*2), 0)),
        ('H', (2, -(2.54-0.46*2), 0))
        ],
    "basis": "evangelista_custom"
}

def test_1():
    vals = scripts.runner.subspace(molecule, solver=scripts.pUCCSD1.conventional, test=False, comp_grad=False)[0]
    np.testing.assert_approx_equal(vals["energy"], -15.763684331, significant=8)

def test_2():
    vals = scripts.runner.subspace(molecule, solver=scripts.pUCCSD2.conventional, test=False, comp_grad=False)[0]
    np.testing.assert_approx_equal(vals["energy"], -15.695942575, significant=8)

def test_3():
    vals = scripts.runner.subspace(molecule, solver=scripts.pUCCSD3.conventional, test=False, comp_grad=False)[0]
    np.testing.assert_approx_equal(vals["energy"], -15.694224787, significant=8)

def test_4():
    vals = scripts.runner.subspace(molecule, solver=scripts.pUCCSD4.conventional, test=False, comp_grad=False)[0]
    np.testing.assert_approx_equal(vals["energy"], -15.694617317, significant=8)
