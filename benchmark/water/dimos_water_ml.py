import dimos
import parmed
import torch
import time
import numpy as np
#import openmm.app as mmapp
#import openmm as mm
import sys
import os

from mace.calculators import mace_mp
from ase import build
from ase.md import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units

from ase.build import molecule
from ase import Atoms
from ase.io import write as asewrite

#import torch._dynamo
#torch._dynamo.config.suppress_errors = True

torch.backends.cuda.matmul.allow_tf32 = False

warmup = 5
warmup_steps_between_pause = 25
steps_between_pause = 35
measure_steps = 40
stepsize = 0.5

#tool="OpenMMReference"
#tool="OpenMMCPU"
tool=sys.argv[1]#"dimosCPUcompile"
model=sys.argv[2]
box_size=int(sys.argv[3])

ff_accuracy=int(sys.argv[4])
integration_accuracy=int(sys.argv[5])

if ff_accuracy==32:
    dtype_ff = torch.float32
else:
    dtype_ff = torch.float64
if integration_accuracy==32:
    dtype_integration = torch.float32
else:
    dtype_integration = torch.float64

num_threads = os.getenv("OMP_NUM_THREADS", default = None)
#os.environ["OPENMM_CPU_THREADS"] = str(num_threads)

#for nonbonded in ["Cutoff", "PME"]: # , "Ewald"]: "Cutoff",
output_file = open(f"{tool}model{model}_dtypeff{dtype_ff}_dtypeint{dtype_integration}_device{torch.get_default_device()}_threads{num_threads}_ml.dat", "a")
#for box_size in (20,30,40,50,60,70,80,90,100,110,120,130,140):
print(f"###### {box_size} and {model} #######")      

if "CPU" in tool:
    torch.set_default_device("cpu")
elif "CUDA" in tool:
    torch.set_default_device("cuda:0")
else:
    raise Exception("Platform not found")

if "dimos" in tool:
    rst7_file = parmed.amber.Rst7(f'amber{box_size}.inpcrd')  
    positions = torch.tensor(rst7_file.positions.value_in_unit(parmed.unit.angstrom))

    mm_system = dimos.ff.AmberForceField(f'amber{box_size}.prmtop', cutoff=5.0, switch_distance=3.5, nonbonded_type="Cutoff", periodic=True, dtype=dtype_ff)

    #system = MaceFoundationalSystem(masses, atomic_numbers, mace_name="mace-off-small", box=box, periodic=periodic)
    if "mace" in model:
        if "small" in model:
            mace_name = "mace-mp-small"
        elif "medium" in model:
            mace_name = "mace-mp-medium"
        elif "large" in model:
            mace_name = "mace-mp-large"            
        if "cuEq" in tool:
            dimos_system = mm_system.as_mace_system(mace_name=mace_name, enable_cueq=True)
        else:
            dimos_system = mm_system.as_mace_system(mace_name=mace_name, enable_cueq=False)
    elif "orb" in model:
        dimos_system = mm_system.as_orb_system()

    integrator = dimos.integrators.LangevinDynamics(stepsize, 300, 0.1, dimos_system, dtype=dtype_integration)

    simulation = dimos.simulation.MDSimulation(dimos_system, integrator, positions, temperature=300)

    if "compile" in tool:
        print("Compilation started")
        torch.compiler.reset()
        step = torch.compile(simulation.step, mode="max-autotune-no-cudagraphs")
        print("...done")                
    else:
        step = simulation.step

elif "ase" in tool:
    rst7_file = parmed.amber.Rst7(f'amber{box_size}.inpcrd')
    positions = torch.tensor(rst7_file.positions.value_in_unit(parmed.unit.angstrom))
    if "CPU" in tool:
        device_str = "cpu"
    elif "CUDA" in tool:
        device_str = "cuda:0"
    else:
        raise NotImplementedError("Platform not found")
    if ff_accuracy==32:
        accuracy_str = "float32"
    else:
        accuracy_str = "float64"

    if "mace" in model:
        if "small" in model:
            size = "small"
        elif "medium" in model:
            size = "medium"
        elif "large" in model:
            size = "large"   
        if "cuEq" in tool:
            if "CPU" in tool:
                macemp = mace_mp(enable_cueq=True, model=size, default_dtype=accuracy_str, device=device_str)
            else:
                torch.set_default_device("cpu")
                macemp = mace_mp(enable_cueq=True, model=size, default_dtype=accuracy_str, device=device_str)
                torch.set_default_device("cuda:0")
        else:
            macemp = mace_mp(enable_cueq=False, model=size, default_dtype=accuracy_str, device=device_str)
        
        water = molecule("H2O")
        water_symbols = water.get_chemical_symbols()

        symbols = []
        for _ in range(positions.size(0)//3):
            symbols.extend(water_symbols)

        atoms = Atoms(symbols=symbols, positions=positions.detach().cpu().numpy(), cell=[box_size] * 3, pbc=True)
        if "compile" in tool:
            print("Compilation started")
            torch.compiler.reset()
            macemp = torch.compile(macemp, mode="max-autotune-no-cudagraphs")
            print("...done")
        
        atoms.calc = macemp

        MaxwellBoltzmannDistribution(atoms, 300 * units.kB)
        dyn = Langevin(atoms, stepsize * units.fs, 300 * units.kB, 0.1/units.fs)

        step = dyn.run
    elif "orb" in model:
        # TODO: Add dtype handling.
        from orb_models.forcefield import atomic_system, pretrained
        from orb_models.forcefield.calculator import ORBCalculator
        orbff = pretrained.orb_v2(device=device_str)
        #orbff.node_head = None
        water = molecule("H2O")
        water_symbols = water.get_chemical_symbols()

        symbols = []
        for _ in range(positions.size(0)//3):
            symbols.extend(water_symbols)

        atoms = Atoms(symbols=symbols, positions=positions.detach().cpu().numpy(), cell=[box_size] * 3, pbc=True)
        system_config = atomic_system.SystemConfig(radius=6.0, max_num_neighbors=150)

        calc = ORBCalculator(orbff, system_config=system_config, device=device_str)
        atoms.calc = calc
        #atoms.set_calculator(calc)

        MaxwellBoltzmannDistribution(atoms, 300 * units.kB)
        dyn = Langevin(atoms, stepsize * units.fs, 300 * units.kB, 0.1/units.fs)

        step = dyn.run
elif "openMM" in tool:
    print("openMM does not support mace-mp, and otherwise I could not make it run because of missing symbols, most likely from a mismatch of Nvidia versions")
else:
    raise NotImplementedError("Platform not recognized")

for s in range(warmup):
    print(f"WarmUp step {s+1}/{warmup}")
    step(warmup_steps_between_pause)

#if "dimos" in tool:
config_file = open(f"water{box_size}_ml.pdb","w")

individual_measurements = []
start_time = time.time()
for s in range(measure_steps):
    #print(f"Measure step {s+1}/{measure_steps}: ", end="", flush=True)
    individual_start = time.time()
    step(steps_between_pause)
    runtime = (time.time()-individual_start)/steps_between_pause
    if "dimos" in tool:
        mm_system.write_pdb(simulation.pos, config_file, unwrap_bonds=False)
    elif "ase" in tool:
        asewrite(config_file, atoms)
    print(f"{runtime}")
    individual_measurements.append(runtime)
total_time = time.time()-start_time
measurements = np.array(individual_measurements)
mean_time = measurements.mean()
std_sem_time = measurements.std(ddof=1) / np.sqrt(np.size(measurements))
print(f"{box_size}\t{positions.size(0)}\t{total_time}\t{total_time/(measure_steps*steps_between_pause)}\t{total_time*measure_steps*steps_between_pause*stepsize*1e-6/(24*60*60)}\t{mean_time}\t{std_sem_time}", file=output_file, flush=True)



        
