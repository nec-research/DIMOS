import dimos
import parmed
import torch
import time
import numpy as np
import openmm.app as mmapp
import openmm as mm
import sys
import os

#import torch._dynamo
#torch._dynamo.config.suppress_errors = True

torch.backends.cuda.matmul.allow_tf32 = False
#torch.set_float32_matmul_precision('high')
#torch.cuda.memory._record_memory_history()

warmup = 5
warmup_steps_between_pause = 25
steps_between_pause = 35
measure_steps = 40
stepsize = 0.5

#tool="OpenMMReference"
#tool="OpenMMCPU"
tool=sys.argv[1]#"dimosCPUcompile"
nonbonded=sys.argv[2]
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
os.environ["OPENMM_CPU_THREADS"] = str(num_threads)

#for nonbonded in ["Cutoff", "PME"]: # , "Ewald"]: "Cutoff",
output_file = open(f"{tool}_nonbonded{nonbonded}_dtypeff{dtype_ff}_dtypeint{dtype_integration}_device{torch.get_default_device()}_threads{num_threads}.dat", "a")
#for box_size in (20,30,40,50,60,70,80,90,100,110,120,130,140):
print(f"###### {box_size} and {nonbonded} #######")      

if "dimos" in tool:
    if "CPU" in tool:
        torch.set_default_device("cpu")
    elif "CUDA" in tool:
        torch.set_default_device("cuda:0")
    else:
        raise Exception("Platform not found")
    
    rst7_file = parmed.amber.Rst7(f'amber{box_size}.inpcrd')  
    positions = torch.tensor(rst7_file.positions.value_in_unit(parmed.unit.angstrom))

    dimos_system = dimos.ff.AmberForceField(f'amber{box_size}.prmtop', cutoff=10, switch_distance=7.5, nonbonded_type=nonbonded, periodic=True, dtype=dtype_ff)

    integrator = dimos.integrators.LangevinDynamics(stepsize, 300, 1.0, dimos_system, dtype=dtype_integration)

    simulation = dimos.simulation.MDSimulation(dimos_system, integrator, positions, temperature=300)

    if "compile" in tool:
        print("Compilation started")
        torch.compiler.reset()
        step = torch.compile(simulation.step, mode="max-autotune-no-cudagraphs")
        print("...done")                
    else:
        step = simulation.step
    #output = open("compile_explainer","w")
    #print(torch._dynamo.explain(step)(1), file=output, flush=True)
elif "openMM" in tool:
    rst7_file = parmed.amber.Rst7(f'amber{box_size}.inpcrd')  
    positions = torch.tensor(rst7_file.positions.value_in_unit(parmed.unit.angstrom))
    openmm_prmtop = mmapp.AmberPrmtopFile(f'amber{box_size}.prmtop')
    match nonbonded:
        case "Cutoff":
            openmm_system = openmm_prmtop.createSystem(removeCMMotion=False,constraints=None,rigidWater=False, implicitSolvent=None, nonbondedMethod=mmapp.forcefield.CutoffPeriodic, nonbondedCutoff=10*mm.unit.angstrom, switchDistance=7.5*mm.unit.angstrom, solventDielectric=78.5)
        case "Ewald":
            openmm_system = openmm_prmtop.createSystem(removeCMMotion=False,constraints=None,rigidWater=False, implicitSolvent=None, nonbondedMethod=mmapp.forcefield.Ewald, nonbondedCutoff=10*mm.unit.angstrom, switchDistance=7.5*mm.unit.angstrom, solventDielectric=78.5)
        case "PME":
            openmm_system = openmm_prmtop.createSystem(removeCMMotion=False,constraints=None,rigidWater=False, implicitSolvent=None, nonbondedMethod=mmapp.forcefield.PME, nonbondedCutoff=10*mm.unit.angstrom, switchDistance=7.5*mm.unit.angstrom, solventDielectric=78.5)
    openmm_integrator = mm.LangevinMiddleIntegrator(300, 1.0/mm.unit.femtoseconds, stepsize*mm.unit.femtoseconds)
    properties = {}
    print("TOOL", tool)
    if "Reference" in tool:
        openmm_platform = mm.Platform.getPlatformByName('Reference')
    elif "CUDA" in tool:
        openmm_platform = mm.Platform.getPlatformByName('CUDA')
        properties["DeviceIndex"] = "0"
        if dtype_ff == torch.float64 and dtype_integration == torch.float64:
            properties["Precision"] = "double"
        elif dtype_ff == torch.float32 and dtype_integration == torch.float32:
            properties["Precision"] = "single"
        else:
            properties["Precision"] = "mixed"
    elif "CPU" in tool:
        openmm_platform = mm.Platform.getPlatformByName('CPU')
    else:
        raise Exception("Platform not found")
    openmm_simulation = mmapp.Simulation(openmm_prmtop.topology, openmm_system, openmm_integrator, openmm_platform, properties)
    openmm_simulation.context.setPositions(rst7_file.positions)
    openmm_simulation.context.setVelocitiesToTemperature(300)
    step = openmm_simulation.step
elif "torchmd" in tool:
    from torchmd.forcefields.ff_parmed import ParmedForcefield
    from torchmd.forces import Forces
    from torchmd.parameters import Parameters

    from torchmd.integrator import maxwell_boltzmann
    from torchmd.systems import System

    from torchmd.integrator import Integrator

    torch.set_default_dtype(dtype_ff)

    if "CPU" in tool:
        device_string = "cpu"
    elif "CUDA" in tool:
        device_string = "cuda:0"
    else:
        raise Exception("Platform not found")

    class mol():
        def __init__(self, numAtoms):
            self.numAtoms = numAtoms
            self.atomtype = np.array(['O1', 'H1', 'H1'] * (numAtoms//3))
            self.charge = np.array([-0.834,  0.417,  0.417] * (numAtoms//3))
            self.masses = np.array([15.99943, 1.007947, 1.007947] * (numAtoms//3))
            
            bonds = []
            for atom in range(numAtoms//3):
                local_atom = int(atom*3)
                bonds.append([local_atom,local_atom+1])
                bonds.append([local_atom,local_atom+2])
            self.bonds = np.array(bonds)

            angles = []
            for atom in range(numAtoms//3):
                local_atom = int(atom*3)
                angles.append([local_atom+1,local_atom+0,local_atom+2])
            self.angles = np.array(angles)

            self.dihedrals = []
            self.impropers = []

    rst7_file = parmed.amber.Rst7(f'amber{box_size}.inpcrd')  
    positions = torch.tensor(rst7_file.positions.value_in_unit(parmed.unit.angstrom)).requires_grad_(True).unsqueeze(-1).detach().numpy()

    mol_obj = mol(positions.shape[0])
    mol_obj.box = np.array([box_size, box_size, box_size])

    ff = ParmedForcefield(mol_obj, f'amber{box_size}.prmtop')
    param = Parameters(ff, mol_obj, device=device_string)

    force = Forces(param, 
                cutoff=10.0,
                rfa=True,
                switch_dist=7.5,
                terms=[
                    "bonds",
                    "angles",
                    "electrostatics",
                    "lj",
                ],
    )

    system = System(mol_obj.numAtoms, nreplicas=1, precision=dtype_ff, device=device_string)
    system.set_positions(positions)
    system.set_box(mol_obj.box)
    system.set_velocities(maxwell_boltzmann(param.masses, T=300, replicas=1))
    
    langevin_temperature = 300  # K
    langevin_gamma = 0.1
    timestep = 0.5  # fs

    integrator = Integrator(system, force, timestep, device_string, gamma=langevin_gamma, T=langevin_temperature)

    if "compile" in tool:
        print("Compilation started")
        torch.compiler.reset()
        step = torch.compile(integrator.step, mode="max-autotune")#, mode="max-autotune-no-cudagraphs")
        print("...done")                
    else:
        step = integrator.step
else:
    raise NotImplementedError("Platform not recognized")

for s in range(warmup):
    print(f"WarmUp step {s+1}/{warmup}")
    step(warmup_steps_between_pause)
#torch.cuda.memory._dump_snapshot("my_snapshot.pickle")
#print(torch.cuda.memory_summary())
#exit()

if "dimos" in tool:
    config_file = open(f"dimos{box_size}.pdb","w")

individual_measurements = []
start_time = time.time()
for s in range(measure_steps):
    #print(f"Measure step {s+1}/{measure_steps}: ", end="", flush=True)
    individual_start = time.time()
    step(steps_between_pause)
    runtime = (time.time()-individual_start)/steps_between_pause
    #if "dimos" in tool:
    #    simulation.sys.write_pdb(simulation.pos, config_file, unwrap_bonds=False)
    print(f"{runtime}")
    individual_measurements.append(runtime)
total_time = time.time()-start_time
measurements = np.array(individual_measurements)
mean_time = measurements.mean()
std_sem_time = measurements.std(ddof=1) / np.sqrt(np.size(measurements))
print(f"{box_size}\t{positions.shape[0]}\t{total_time}\t{total_time/(measure_steps*steps_between_pause)}\t{total_time*measure_steps*steps_between_pause*stepsize*1e-6/(24*60*60)}\t{mean_time}\t{std_sem_time}", file=output_file, flush=True)



        
