import argparse
import time
import sys
from pathlib import Path
import json

import numpy as np
from mpi4py import MPI

from openmm import openmm, unit, app
from openmmtools.integrators import BAOABIntegrator, NonequilibriumLangevinIntegrator

import grand


def read_moves(filename):
    n_moves    = 0
    n_accepted = 0
    with open(filename, encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines[-1::-1]:
        if " move(s) completed (" in line:
            n_moves    = int(line.split()[4])
            n_accepted = int(line.split()[7].strip('('))
            break
    return n_moves, n_accepted

def read_cycle(filename):
    cycle = 0
    with open(filename, encoding="utf-8") as f:
        lines = f.readlines()
        for l in lines[-1:0:-1]:
            if "Cycle" in l:
                word = [w for w in l.split(",") if "Cycle" in w][0]
                cycle = int(word.split()[-1])
                break
    return cycle

def count_time(time_start):
    current_time = time.time()
    elapsed_time = current_time - time_start
    n_hours, remainder   = divmod(elapsed_time, 3600)
    n_minutes, n_seconds = divmod(remainder, 60)
    n_hours, n_minutes = int(n_hours), int(n_minutes)
    return n_hours, n_minutes, n_seconds, elapsed_time

def stop_simulation_bcast(rank, elapsed_time, maxh):
    if rank == 0:
        if elapsed_time >= maxh * 3600:
            stop_simulation = True
        else:
            stop_simulation = False
    else:
        stop_simulation = False
    stop_simulation = MPI.COMM_WORLD.bcast(stop_simulation, root=0)
    return stop_simulation

def log_simulation_info(args, mmdp_inputs, gcncmc_mover):
    """
    Log all the information about the simulation
    """
    # log all command line arguments
    gcncmc_mover.logger.info(f"Command line: {' '.join(sys.argv)}")
    for arg in vars(args):
        gcncmc_mover.logger.info(f"    {arg:9s}: {getattr(args, arg)}")

    # log MD parameters
    gcncmc_mover.logger.info("MD parameters:")
    for key, value in vars(mmdp_inputs).items():
        gcncmc_mover.logger.info(f"    {key:12s}: {value}")

    # log gcncmc_mover parameters
    gcncmc_mover.logger.info("GCNCMC parameters:")
    gcncmc_mover.logger.info(f"    mu = {gcncmc_mover.excessChemicalPotential.value_in_unit(unit.kilojoule_per_mole)} kJ/mol, "
                             f"{gcncmc_mover.excessChemicalPotential.value_in_unit(unit.kilocalorie_per_mole)} kcal/mol, "
                             f"{gcncmc_mover.excessChemicalPotential/gcncmc_mover.kT} kT")
    gcncmc_mover.logger.info(f"    Center atom(s)     : {gcncmc_mover.ref_atoms}")
    gcncmc_mover.logger.info(f"    Sphere radius      : {gcncmc_mover.sphere_radius.value_in_unit(unit.nanometer)} nm")
    protocol_time_ps = ((gcncmc_mover.n_pert_steps + 1) * gcncmc_mover.n_prop_steps_per_pert * gcncmc_mover.time_step).value_in_unit(unit.picosecond)
    gcncmc_mover.logger.info(f"    Non-eq GC protocol : ({gcncmc_mover.n_pert_steps} + 1) * {gcncmc_mover.n_prop_steps_per_pert} "
                             f"= {(gcncmc_mover.n_pert_steps + 1) * gcncmc_mover.n_prop_steps_per_pert} steps "
                             f"= {protocol_time_ps} ps")

def main():
    time_start = time.time()
    parser = argparse.ArgumentParser(
        usage="mpirun -np <X> %(prog)s -multidir <dir> -sys sys.xml.gz -p top.psf -deffnmi md0 -deffnmo md1 -mmdp md.mmdp...",
        description=f"grand {grand.__version__}. This is the script for running GCNCMC with Replica Exchange.")

    parser.add_argument("-sys", metavar="     sys.xml.gz   ", default="sys.xml.gz",
                        help="Serialized system file. It will be loaded as a `openmm.System` object. This system should "
                             "include all bonded/non-bonded, constraints, but not pressure coupling. It can be xml or xml.gz.")
    parser.add_argument("-mmdp", metavar="    md.mmdp      ", default="md.mmdp", required=True,
                        help="Input file with MD parameters. Only limited gmx mdp keywords are supported.")
    parser.add_argument("-multidir", metavar="multi_dir    ", required=True, nargs='+',
                        type=Path,
                        help="The directory for the multi-simulation. If not provided, simulation cannot be run.")
    parser.add_argument("-p", metavar="       top.psf/top.parm7", default="top.psf", required=True,
                        help="Input topology file")
    parser.add_argument("-irst", metavar="    start.rst7   ",
                        help="Initial coordinates/velocities/box file")
    parser.add_argument("-orst", metavar="    md_rst.rst7  ",
                        help="Checkpoint file for output")
    parser.add_argument("-ighosts", metavar=" ghosts0.dat  ",
                        help="Input ghost water file for restarting")
    parser.add_argument("-oghosts", metavar=" ghosts1.dat  ",
                        help="Output ghost water file,")
    parser.add_argument("-ilog", metavar="    md_gcmc0.log ",
                        help="gcmc log file from the previous run. The number of completed/accepted GC moves, and RE cycle will be read from this file.")
    parser.add_argument("-olog", metavar="    md_gcmc1.log ",
                        help="Output log file")
    parser.add_argument("-odcd", metavar="    md_gcmc1.dcd ",
                        help="Output dcd file")
    parser.add_argument("-opdb", metavar="    md.pdb       ", default="Final.pdb",
                        help="Output pdb file. if the simulation finishes, the last frame will be written to this file.")
    parser.add_argument("-atom", metavar="    atom.json    ", required=True,
                        help="Input atom json file for atom selection")
    parser.add_argument("-maxh", metavar="    23.8 hour    ", default=23.8, type=float,
                        help="Maximal number of hours to run. Time will only be checked after each cycle.")
    parser.add_argument("-deffnmi", metavar=" md_in        ",
                        help="The default filename for all input. -irst, -ighosts, -ilog, will be ignored.")
    parser.add_argument("-deffnmo", metavar=" md_out       ",
                        help="The default filename for all output. -orst, -oghosts, -olog will be ignored.")
    parser.add_argument("-re_cycle", metavar="100          ", default=100, type=int,
                        help="Number of replica exchange cycles to run.")

    args = parser.parse_args()

    multi_dir = args.multidir
    rank = MPI.COMM_WORLD.Get_rank()
    run_dir = multi_dir[rank]

    # deffnmo and deffnmi cannot be the same
    if args.deffnmi and args.deffnmo and args.deffnmi == args.deffnmo:
        raise ValueError("deffnmi and deffnmo cannot be the same.")
    # if the default filenames are provided, use deffnm
    if args.deffnmi:
        args.irst = f"{args.deffnmi}.rst7"
        args.ighosts = f"{args.deffnmi}.dat"
        args.ilog = f"{args.deffnmi}.log"
    if args.deffnmo:
        args.orst = f"{args.deffnmo}.rst7"
        args.oghosts = f"{args.deffnmo}.dat"
        args.olog = f"{args.deffnmo}.log"
        args.odcd = f"{args.deffnmo}.dcd"
        args.opdb = f"{args.deffnmo}.pdb"

    # load system, topology, and restart files
    system = grand.utils.load_sys(str(run_dir/args.sys))
    topology = grand.utils.load_top(args.p)
    s_rst = app.AmberInpcrdFile(run_dir/args.irst)
    topology.setPeriodicBoxVectors(s_rst.getBoxVectors())

    # load ghost_list
    with open(run_dir/args.ighosts, 'r') as f:
        line = f.readlines()[-1]
        ghost_list = [int(i) for i in line.split(",")]

    # load simulation parameters
    mmdp_inputs = grand.utils.mmdp_parser().read( args.mmdp )

    # load atom selection
    with open(args.atom, 'r') as f:
        atoms_region = json.load(f)

    # set up integrator
    integrator = BAOABIntegrator(mmdp_inputs.ref_t, 1 / mmdp_inputs.tau_t, mmdp_inputs.dt)

    # set up sampler
    gcncmc_mover = grand.samplers.NonequilibriumGCMCSphereSamplerMultiState(
        system=system,
        topology=topology,
        temperature=mmdp_inputs.ref_t,
        timeStep=mmdp_inputs.dt,
        integrator=integrator,
        nPertSteps=mmdp_inputs.n_pert_steps,  # number of perturbation steps (Hamiltonian switching)
        nPropStepsPerPert=mmdp_inputs.n_prop_steps_per_pert,  # number of propagation steps per perturbation step (constant Hamiltonian, relaxation)
        excessChemicalPotential=mmdp_inputs.ex_potential,
        standardVolume=mmdp_inputs.standard_volume,
        referenceAtoms=atoms_region["ref_atoms"],
        sphereRadius=atoms_region["radius"] * unit.nanometer,
        log=run_dir/args.olog,
        dcd=str(run_dir/args.odcd),
        rst=str(run_dir/args.orst),
        ghostFile=run_dir/args.oghosts,
        overwrite=True
    )

    sim = app.Simulation(topology, system, gcncmc_mover.compound_integrator)
    sim.context.setPeriodicBoxVectors(*s_rst.getBoxVectors())
    sim.context.setPositions(s_rst.getPositions())
    sim.context.setVelocities(s_rst.getVelocities())
    gcncmc_mover.logger.info(f"Initial positions and velocities are loaded from {args.irst}")

    gcncmc_mover.initialise(sim.context, ghost_list)
    gcncmc_mover.logger.info(f"Initial ghost_list: {ghost_list}")

    if (args.ilog is not None) and (run_dir/args.ilog).exists():
        gcncmc_mover.re_cycle = read_cycle(run_dir/args.ilog)
        gcncmc_mover.n_moves, gcncmc_mover.n_accepted = read_moves(run_dir/args.ilog)

    # make sure all the gcncmc_mover.re_cycle are the same
    cycle_list = MPI.COMM_WORLD.allgather(gcncmc_mover.re_cycle)
    if len(set(cycle_list)) != 1:
        raise ValueError(f"Replica Exchange Cycle number is not the same in all replicas: {cycle_list}")

    log_simulation_info(args, mmdp_inputs, gcncmc_mover)

    # run the simulation
    gcncmc_mover.logger.info("Simulation starts")
    n_hours, n_minutes, n_seconds, elapsed_time = count_time(time_start)
    time_up_flag = False
    while gcncmc_mover.re_cycle < args.re_cycle:
        time_up_flag = stop_simulation_bcast(rank, elapsed_time, args.maxh)
        if time_up_flag:
            gcncmc_mover.logger.info(f"Time has exceeded maxh({args.maxh}). Stop here.")
            break

        # Actual simulation, execute md_gc_re_protocol
        for step_name, step_n in mmdp_inputs.md_gc_re_protocol:
            gcncmc_mover.logger.info(f"{step_name} : {step_n}")
            if step_name == "MD":
                sim.step(step_n)
            elif step_name == "GC":
                gcncmc_mover.move(sim.context, step_n)
            elif step_name == "RE":
                gcncmc_mover.exchange_neighbor_swap()
            else:
                raise ValueError(f"Unknown step name: {step_name}")
        gcncmc_mover.report(sim)
        n_hours, n_minutes, n_seconds, elapsed_time = count_time(time_start)
        gcncmc_mover.logger.info(
            f"Cycle {gcncmc_mover.re_cycle}, rank: {rank}, dir: {run_dir}, {n_hours} h {n_minutes} m {n_seconds:.2f} s")
    if not time_up_flag:
        state = sim.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=True)
        with open(run_dir / args.opdb, "w") as f:
            app.PDBFile.writeFile(topology, state.getPositions(), f, keepIds=True)
    gcncmc_mover.logger.info(f"grand_RE_MPI finished in {n_hours} h {n_minutes} m {n_seconds:.2f}")
