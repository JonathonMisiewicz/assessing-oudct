import psi4
from psi4.driver import driver_findif
from unitary_pilot_implementations import scripts
import numpy as np

"""
Computes a gradient analytically and by finite difference, to confirm the accuracy of the gradients.
To test a different method, just change the 'solver' argument on the scripts.runner.subspace call.
"""

np.set_printoptions(precision=10)

h2o = psi4.geometry("""
            1 2
        units bohr
         O 0.000000000000 0.000000000000 -0.143225816552
         H 0.000000000000 1.638036840407 1.136548822547
         H 0.000000000000 -1.638036840407 1.136548822547
        """)

psi4.set_options({
    "reference": "uhf",
    "basis": "sto-3g",
    "points": 5
})

h2o.update_geometry()
h2o.reinterpret_coordentry(False)
h2o.fix_orientation(True)

findif_meta_dict = driver_findif.gradient_from_energies_geometries(h2o) 
ndisp = len(findif_meta_dict["displacements"]) - 1 

def fc_wrapper(molecule=None, **kwargs):
    schema = molecule.to_schema(1)
    # Convert the MolSSI schema to something I can work with.
    geom_arr = []
    for i, symbol in enumerate(schema["molecule"]["symbols"]):
        geom_arr.append((symbol, schema["molecule"]["geometry"][i*3:(i+1)*3]))
    mol_dict = { 
        "charge": int(schema["molecule"]["molecular_charge"]),
        "num_unpaired": schema["molecule"]["molecular_multiplicity"] - 1,
        "geom": geom_arr,
        "basis": psi4.core.get_global_option("basis")
    }
    vals = scripts.runner.subspace(molecule=mol_dict, solver=scripts.odc12.simultaneous, comp_grad=True, e_thresh=1e-9, r_thresh=1e-9, **kwargs)[0]
    psi4.core.set_variable('CURRENT ENERGY', vals["energy"])
    basis = psi4.core.BasisSet.build(molecule, 'BASIS', mol_dict["basis"], quiet=True)
    wfn = psi4.core.Wavefunction(molecule, basis)
    vals["wfn"] = wfn 
    return vals

def process_displacement(func, molecule, displacement):
    geom_array = np.reshape(displacement["geometry"], (-1, 3)) 
    molecule.set_geometry(psi4.core.Matrix.from_array(geom_array))
    vals = func(molecule, return_wfn=True)
    displacement["energy"] = vals["energy"]
    displacement["gradient"] = vals["gradient"]
    psi4.core.clean()
    return vals["wfn"]

process_displacement(fc_wrapper, h2o, findif_meta_dict["reference"])


for n, displacement in enumerate(findif_meta_dict["displacements"].values()):
    process_displacement(fc_wrapper, h2o, displacement)

G = driver_findif.assemble_gradient_from_energies(findif_meta_dict) 
print("GRADIENTS")
print(G)
print(findif_meta_dict["reference"]["gradient"])
