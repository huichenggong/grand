import argparse
import time
import sys
from pathlib import Path
import json

import numpy as np

from openmm import openmm, unit, app
from openmmtools.integrators import BAOABIntegrator, NonequilibriumLangevinIntegrator

import grand

class mdp_parser:
    def __init__(self):
        self.integrator = openmm.LangevinIntegrator
        self.dt = 2/1000 * unit.picosecond
        self.nstmaxh = 1000                # time up check interval
        self.nsteps = 5000                 # number of step
        self.ncycle = 0                    # number of cycle
        self.nstxout_compressed = 5000     # save xtc trajectory every X step, 0 means no saving
        self.nstdcd = 0                    # save xtc trajectory every X step, 0 means no saving
        self.nstenergy = 1000              # save csv file every X step
        self.tau_t = 2.0 * unit.picosecond # 1/gama, inverse friction constant
        self.ref_t = 298 * unit.kelvin     # reference temperature
        self.gen_vel = False               #
        self.gen_temp = 298 * unit.kelvin  #
        self.restraint = False             #
        self.res_fc = 1000                 # restraint force constant, in kJ/mol/nm^2
        self.pcoupltype = None             # can be "None", "isotropic", "semiisotropic/membrane", "anisotropic"
        self.ref_p = 1.0 * unit.bar        #
        self.nstpcouple = 25               # in steps
        self.surface_tension = 0.0         # in kJ/mol/nm^2
        # GCMC
        self.ex_potential    = -26.5254     * unit.kilojoule_per_mole
        self.standard_volume = 29.814952e-3 * unit.nanometer**3
        self.n_pert_steps = 399            # number of perturbation steps (Hamiltonian switching)
        self.n_prop_steps_per_pert = 50    # number of propagation steps per perturbation step (constant Hamiltonian, relaxation)

    def read(self, input_mdp):
        with open(input_mdp) as f:
            lines = f.readlines()
        for line in lines:
            if line.find(';') >= 0: line = line.split(';')[0]
            line = line.strip()
            if "=" in line:
                segments = line.split('=')
                input_param = segments[0].lower().strip().replace("-","_")
                inp_val = segments[1].strip()
                if input_param == "integrator":
                    if   inp_val == "LangevinIntegrator":       self.integrator = openmm.LangevinIntegrator
                    elif inp_val == "LangevinMiddleIntegrator": self.integrator = openmm.LangevinMiddleIntegrator
                    else: raise ValueError(f"{inp_val} is not support in mdp_parser")
                if input_param == "dt":                 self.dt = float(inp_val) * unit.picosecond
                if input_param == "nstmaxh":            self.nstmaxh = int(inp_val)
                if input_param == "nsteps":             self.nsteps = int(inp_val)
                if input_param == "ncycle":             self.ncycle = int(inp_val)
                if input_param == "nstxout_compressed": self.nstxout_compressed = int(inp_val)
                if input_param == "nstdcd":             self.nstdcd = int(inp_val)
                if input_param == "nstenergy":          self.nstenergy = int(inp_val)
                if input_param == "tau_t":              self.tau_t = float(inp_val) * unit.picosecond
                if input_param == "ref_t":              self.ref_t = float(inp_val) * unit.kelvin
                if input_param == "gen_vel":
                    if   inp_val.lower() in ["yes", "on" ]: self.gen_vel = True
                    elif inp_val.lower() in ["no",  "off"]:  self.gen_vel = False
                    else : raise ValueError(f"{inp_val} is not a valid input for gen_vel")
                if input_param == "gen_temp":   self.gen_temp = float(inp_val) * unit.kelvin
                if input_param == "restraint":
                    if   inp_val.lower() in ["yes", "on" ]:  self.restraint = True
                    elif inp_val.lower() in ["no",  "off"]:  self.restraint = False
                    else : raise ValueError(f"{inp_val} is not a valid input for restraint")
                if input_param == "res_fc":     self.res_fc = float(inp_val)
                if input_param == "pcoupltype":
                    if inp_val.lower() in ["isotropic", "membrane", "anisotropic"]:
                        self.pcoupltype = inp_val.lower()
                    elif inp_val.lower()== "semiisotropic":
                        self.pcoupltype = "membrane"
                    elif inp_val.lower() == "none":
                        self.pcoupltype = None
                    else:
                        raise ValueError(f"{inp_val} is not a valid input for pcoupltype")
                if input_param == "ref_p":      self.ref_p = [float(i) for i in inp_val.split()] * unit.bar
                if input_param == "nstpcouple": self.nstpcouple = int(inp_val)
                if input_param == "surface_tension": self.surface_tension = float(inp_val)
                if input_param == "ex_potential"          : self.ex_potential    = float(inp_val) * unit.kilojoule_per_mole
                if input_param == "standard_volume"       : self.standard_volume = float(inp_val) * unit.nanometer**3
                if input_param == "n_pert_steps"          : self.n_pert_steps = int(inp_val)
                if input_param == "n_prop_steps_per_pert" : self.n_prop_steps_per_pert = int(inp_val)
        return self

def read_moves(filename):
    n_completed = 0
    n_accepted = 0
    with open(filename, 'r') as f:
        lines = f.readlines()
    for line in lines[-1::-1]:
        if " move(s) completed (" in line:
            n_completed = int(lines[-1].split()[4])
            n_accepted = int(lines[-1].split()[7].strip('('))
            break
    return n_completed, n_accepted

def check_continuation(logger, rst_file, t_file):
    """
    Check the input arguments and determine if the simulation is a continuation or a fresh start.
    if rst is provided,
        and it exists, then it is a continuation.
        and it does not exist, then it is a fresh start. state file should exist.
    if rst is not provided, then it is a fresh start. -t file should exist.
    """
    if rst_file is not None and Path(rst_file).is_file():
        logger.info(f"Load/continue simulation from {rst_file}")
        return True

    if t_file is not None and Path(t_file).is_file():
        logger.info(f"Arg -rst is not provided. Start a new simulation from {t_file}.")
        return False
    else:
        raise FileNotFoundError(f"Neither -rst nor -t can be found.")


def main():
    time_start = time.time()
    parser = argparse.ArgumentParser(
        description=f"""grand {grand.__version__}. This is the script for running GCNCMC with Replica Exchange.""")
    parser.add_argument("-sys", metavar="     sys.xml.gz  ", default="sys.xml.gz",
                        help="Serialized system file. It will be loaded as a `openmm.System` object. This system should "
                             "include all bonded/non-bonded, constraints, but not pressure coupling. It can be xml or xml.gz.")
    parser.add_argument("-mdp", metavar="     md.mdp      ", default="md.mdp",
                        help="Input file with MD parameters. Only limitied gmx mdp keywords are supported.")
    parser.add_argument("-multidir", metavar="  multi_dir ", required=True, nargs='+',
                        help="The directory for the multi-simulation. If not provided, simulation cannot be run.")
    parser.add_argument("-p", metavar="       top.psf/top.parm7", default="top.psf",
                        help="Input topology file")
    parser.add_argument("-t", metavar="       start.rst7  ",
                        help="Initial coordinates/velocities/box file")
    parser.add_argument("-rst", metavar="     md_rst.rst7 ",
                        help="Restart input file, amber rst7 format. If rst is provided, the simulation will continue from the rst file.")
    parser.add_argument("-ighosts", metavar=" ghosts-0.txt", default="ghosts-0.txt",
                        help="Input ghost water file for starting or continuing")
    parser.add_argument("-ilog", metavar="    md_gcmc.log ",
                        help="gcmc log file from the previous run. The number of completed/accepted moves will be read from this file.")
    parser.add_argument("-atom", metavar="    atom.json   ", default="atom.json",
                        help="Input atom json file for atom selection")
    parser.add_argument("-maxh", metavar="    23.8 hour   ", default=23.8, type=float,
                        help="Maximal number of hours to run. Time will only be checked after each cycle.")
    parser.add_argument("-deffnm", metavar="  md          ", default="md",
                        help="The default filename for all output.")
