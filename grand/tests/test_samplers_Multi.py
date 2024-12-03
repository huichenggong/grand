from pathlib import Path
import gzip

import pytest
from mpi4py import MPI

import numpy as np
from openmm import app, unit, openmm
from openmmtools.integrators import BAOABIntegrator

from grand import samplers


def check_water_paramters(gcncmc_mover, topology, g_list):
    """
    Make sure :
        ghost water has charge=0, lambda=0
        normal water has charge=-0.834, lambda=1
    """
    res_list = [res for res in topology.residues()]
    charge_list =  [-0.834, 0.417, 0.417]
    sigma_list =   [3.15075e-01, 0.08908987, 0.08908987] # nm
    epsilon_list = [6.35968e-01, 0.0       , 0.0] # kJ/mol
    for res in res_list[1:]: # res 0 is methane
        if res.index in g_list:
            # this is a ghost water,
            # self.nonbonded_force has chg=0, and
            # self.custom_nb_force has lambda=0
            for at, sig_ans, eps_ans in zip(res.atoms(), sigma_list, epsilon_list):
                [charge, sigma, epsilon] = gcncmc_mover.nonbonded_force.getParticleParameters(at.index)
                assert charge.value_in_unit(unit.elementary_charge) == 0
                [sigma, epsilon, lam ] = gcncmc_mover.custom_nb_force.getParticleParameters(at.index)
                assert lam == 0.0
                assert sigma   == pytest.approx(sig_ans)
                assert epsilon == pytest.approx(eps_ans)

        else:
            # This is a normal water
            for at, chg_answer, sig_ans, eps_ans in zip(res.atoms(), charge_list, sigma_list, epsilon_list):
                [charge, sigma, epsilon] = gcncmc_mover.nonbonded_force.getParticleParameters(at.index)
                assert charge.value_in_unit(unit.elementary_charge) == chg_answer
                [sigma, epsilon, lam ] = gcncmc_mover.custom_nb_force.getParticleParameters(at.index)
                assert lam == 1.0
                assert sigma == pytest.approx(sig_ans)
                assert epsilon == pytest.approx(eps_ans)
                # assert False

@pytest.mark.mpi(minsize=2)
def test_exchange_identical_U():

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(size, rank)
    multi_dir = [Path(f"../data/tests/methane/{i}") for i in range(4)]
    run_dir = multi_dir[rank]

    # load system
    with gzip.open(multi_dir[0]/"system.xml.gz", 'rt') as f: # use identical system for all replicas
        system = openmm.XmlSerializer.deserialize(f.read())

    # load topology
    prmtop = app.AmberPrmtopFile("../data/tests/methane/06_solv.prmtop")
    topology = prmtop.topology

    with open(run_dir/"ghost_list0.dat") as f:
        line = f.readlines()[0]
        ghost_list = [int(i) for i in line.split(",")]
    md_parm = {"dt"    : 2 * unit.femtosecond,
               "ref_t" : 298 * unit.kelvin,
               "tau_t" : 1 * unit.picosecond,
               }

    integrator = BAOABIntegrator(md_parm["ref_t"], 1/unit.picosecond, md_parm["dt"])
    atoms_region = {"ref_atoms":[{"chain":0, "name": "C", "resname": "MOL", "resid": "1"}],
                    "radius":1.0 * unit.nanometer}
    output_name_dict = {"gcmc_log": str(run_dir / "md.log"),
                        "dcd": str(run_dir / "md.dcd"),
                        "rst": str(run_dir / "md.rst7"),
                        "ghosts": str(run_dir / "md.dat"),
                        "energy": str(run_dir / "md.csv"),
                        }
    gcncmc_mover = samplers.NonequilibriumGCMCSphereSamplerMultiState(
        system=system,
        topology=topology,
        temperature=md_parm["ref_t"],
        timeStep=md_parm["dt"],
        integrator=integrator,
        nPertSteps=399,  # number of perturbation steps (Hamiltonian switching)
        nPropStepsPerPert=10,  # number of propagation steps per perturbation step (constant Hamiltonian, relaxation)
        referenceAtoms=atoms_region["ref_atoms"],
        sphereRadius=atoms_region["radius"],
        log=output_name_dict["gcmc_log"],
        dcd=output_name_dict["dcd"],
        rst=output_name_dict["rst"],
        ghostFile=output_name_dict["ghosts"],
        overwrite=True
    )
    with open(run_dir / "state.xml", 'r') as f:
        s_restart = openmm.XmlSerializer.deserialize(f.read())
    topology.setPeriodicBoxVectors(s_restart.getPeriodicBoxVectors())
    with open(run_dir / "ghost_list0.dat") as f:
        line = f.readlines()[0]
        ghost_list = [int(i) for i in line.split(",")]

    sim = app.Simulation(topology, system, gcncmc_mover.compound_integrator)
    sim.context.setPeriodicBoxVectors(*s_restart.getPeriodicBoxVectors())
    sim.context.setPositions(s_restart.getPositions())
    gcncmc_mover.initialise(sim.context, ghost_list)
    ghost_list = gcncmc_mover.getWaterStatusResids(0)
    g_list_answer = {
        0: [671, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699],
        1: [41, 567, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699],
        2: [71, 125, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699],
        3: [680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699],
        }
    assert ghost_list == g_list_answer[rank]

    state = gcncmc_mover.context.getState(getPositions=True)
    pos_local = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)


    pos_answer = {0:np.array([1.5446407794952393, 1.6207016706466675, 1.3611570596694946]),
                  1:np.array([1.2057862281799316, 1.4349833726882935, 1.9373836517333984]),
                  2:np.array([1.1866472959518433, 1.291158676147461 , 1.6850924491882324]),
                  3:np.array([1.6214585304260254, 1.549951434135437 , 1.5036505460739136]),}
    vel_answer = {0:np.array([.7428833246231079 , -.43068626523017883, -.17129312455654144 ]),
                  1:np.array([.2567459046840668 , .3278700113296509  , -.09530912339687347 ]),
                  2:np.array([-.4813627302646637, -.393468976020813  , -.024286923930048943]),
                  3:np.array([.7628964185714722 , .14759217202663422 , .35749149322509766  ]),}
    # check the coordinate of the first atom
    assert np.allclose(pos_local[0], pos_answer[rank])

    gcncmc_mover.allgather_pos(pos_local, ghost_list)
    # check the coordinate of the first atom in other ranks (self.all_positions)
    for i, pos in enumerate( gcncmc_mover.all_positions ):
        assert np.allclose(pos[0], pos_answer[i])
    # check the ghost_list in other ranks (self.ghost_list_all)
    for i, g_list in enumerate( gcncmc_mover.ghost_list_all ):
        assert g_list == g_list_answer[i]

    # ghost water has chg=0 and lambda=0, normal water has a proper charge and lambda=1
    check_water_paramters(gcncmc_mover, topology, ghost_list)

    # Let's try an exchange here, the acceptance ratio should be 1
    gcncmc_mover.exchange_neighbor_swap()
    gcncmc_mover.report(sim)
    g_list = gcncmc_mover.getWaterStatusResids(0)
    # swap 0-1, 2-3 in answers
    g_list_answer[0], g_list_answer[1] = g_list_answer[1], g_list_answer[0]
    g_list_answer[2], g_list_answer[3] = g_list_answer[3], g_list_answer[2]

    # check the ghost_list is correct
    assert g_list == g_list_answer[rank]

    # get the state and check if the ghost_water has no force
    state = sim.context.getState(getForces=True)
    forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
    for res in topology.residues():
        if res.index in g_list:
            at_list = [at for at in res.atoms()]
            assert len(at_list) == 3
            for at in at_list:
                assert np.allclose(forces[at.index], np.zeros(3))

    # check if all the water has the correct lambda in customNonbondedForce
    check_water_paramters(gcncmc_mover, topology, g_list)

    # check if the position is swapped
    pos_answer[0], pos_answer[1] = pos_answer[1], pos_answer[0]
    pos_answer[2], pos_answer[3] = pos_answer[3], pos_answer[2]
    state = gcncmc_mover.context.getState(getPositions=True)
    pos_local = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    assert np.allclose(pos_local[0], pos_answer[rank])
