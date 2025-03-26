import dimos
import torch
import numpy as np
import sys

from time import time

torch.set_default_dtype(torch.float32)
torch.set_default_device("cuda:0")

c_graph = False
warmup_iterations = 50
steps_per_warmup = 25
measure_iterations = 50
steps_per_measure = 50
system = sys.argv[1]
name = sys.argv[2]

print("Started on:", system, name)
# {"factor_ix", "dhfr", "trp-cage_sol", "ala2_tip3p"}
# {"no_constraints_mm", "constraints_mm", "no_constraints_mlmm", "constraints_mlmm", "no_constraints_ml", "constraints_ml"}

positions = dimos.read_positions(f"{system}.gro")
if "no_constraints" in name:
    constraint_option=None
    timestep = 0.5
else:
    constraint_option="h_angles"
    timestep = 2.0

base_sys = dimos.GromacsForceField(f"{system}.top", f"{system}.gro",cutoff=10, switch_distance=7.5, nonbonded_type="PME", constraint_option=constraint_option, create_graph=c_graph)

if name in ["no_constraints_mm", "constraints_mm"]:
    syst = base_sys
elif name in ["no_constraints_mlmm", "constraints_mlmm"]:
    ml_atoms = base_sys.other_molecules[0]
    del base_sys
    system_excluding_protein = dimos.GromacsForceField(f"{system}.top", f"{system}.gro", cutoff=10, switch_distance=7.5, nonbonded_type=None, constraint_option=constraint_option, excluded_bonded_atoms=ml_atoms, create_graph=c_graph)
    smaller_gromacs_ml_system = system_excluding_protein.as_mace_system("mace-mp-large", atom_range=(0,len(ml_atoms)))
    syst = dimos.mlmm.MLMMsystem(ml_atoms, smaller_gromacs_ml_system, system_excluding_protein, electrostatic_type=None, create_graph=c_graph)
    syst.constraint_handler = system_excluding_protein.constraint_handler
else:
    syst = base_sys.as_mace_system("mace-mp-large")
    syst.constraint_handler = base_sys.constraint_handler

integrator = dimos.NoseHooverDynamics(timestep, 300.0, 0.05,  syst, dtype=torch.float64)
sim = dimos.MDSimulation(syst, integrator, initial_pos=positions,temperature=300.0)

torch.compiler.reset()
torch._dynamo.reset()
sim_step = torch.compile(sim.step, mode="max-autotune-no-cudagraphs")

for warm_it in range(warmup_iterations):
    print(warm_it)
    sim_step(steps_per_warmup)

output_file = open(f"{system}_{name}_benchmark.dat","w")
measurements = np.zeros((measure_iterations,3))
for meas_it in range(measure_iterations):
    start_time = time()
    sim_step(steps_per_measure)
    end_time = time()
    time_period = end_time - start_time
    fraction_of_day = time_period/(24*60*60)
    print(name, time_period, time_period/steps_per_measure, sim.integrator.get_timestep()*steps_per_measure*1e-6/fraction_of_day, sim.measure_potential_energy())
    measurements[meas_it,0] = time_period
    measurements[meas_it,1] = time_period/steps_per_measure
    measurements[meas_it,2] = sim.integrator.get_timestep()*steps_per_measure*1e-6/fraction_of_day
    
ns_per_day = measurements[:,2].mean()
ns_per_day_std = measurements[:,2].std(ddof=1) / np.sqrt(measure_iterations)
output_file.write(f"{ns_per_day}\t{ns_per_day_std}\n")
output_file.flush()