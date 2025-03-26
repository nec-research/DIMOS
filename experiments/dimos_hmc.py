import dimos
import torch
import math
import numpy as np
import sys

torch.set_default_device("cuda:0")
objectives = {"energy", "dihedral", "movement_heavy", "movement_protein", "movement_heavy_protein", "movement_heavy_light02", "movement_heavy_light03", "movement_heavy_light05", "movement_heavy_light08", "movement_heavy_light10"}

train = bool(int(sys.argv[1]))
objective = sys.argv[2]
equilibrate = 2000

assert objective in objectives, f"Objective '{objective}' not recognized"

if not train:
    input_name = sys.argv[3]


name = f"hmc_TRAIN{int(train)}_OBJ{objective}_equilibrate{equilibrate}"

batch_size = 10
base_std = 1.0

output_file = open(f"{name}_collective.dat","w")
output_files_config = [open(f"{name}_config_{i}.pdb", "w") for i in range(batch_size)]
output_files_measure = [open(f"{name}_measure_{i}.dat", "w") for i in range(batch_size)]

gromace_topology = dimos.project_path("benchmark/proteins/ala2_tip3p.top")
coordinate_file = dimos.project_path("benchmark/proteins/ala2_tip3p.gro")

phi_atoms = [4, 6, 8, 14]
psi_atoms = [6, 8, 14, 16]

def get_dihedral_angle(pos, atoms, periodic, box):
    v1 = dimos.get_distance_vectors(pos, [atoms[0], atoms[1]], periodic, box)
    v2 = dimos.get_distance_vectors(pos, [atoms[1], atoms[2]], periodic, box)
    v3 = dimos.get_distance_vectors(pos, [atoms[2], atoms[3]], periodic, box)

    crossv1 = torch.linalg.cross(v1, v2)
    crossv2 = torch.linalg.cross(v2, v3)
    crossv3 = torch.linalg.cross(v2, crossv1)

    normv1 = torch.linalg.vector_norm(crossv1)
    normv2 = torch.linalg.vector_norm(crossv2)
    normv3 = torch.linalg.vector_norm(crossv3)

    normcrossv2 = crossv2 / normv2

    cos_phi = torch.sum(crossv1 * normcrossv2) / normv1
    sin_phi = torch.sum(crossv3 * normcrossv2) / normv3
    phi = -torch.atan2(sin_phi, cos_phi)

    return phi

dimos_system = dimos.GromacsForceField(gromace_topology, coordinate_file, nonbonded_type="PME", create_graph=True)
positions = dimos.read_positions(coordinate_file)

protein_indices = dimos_system.other_molecules[0]

mask_water_hydrogen = torch.zeros_like(dimos_system.masses, dtype=torch.bool)
mask_water_oxygen = torch.zeros_like(dimos_system.masses, dtype=torch.bool)

for mol in dimos_system.length3_molecules:
    for m in mol:
        if math.isclose(dimos_system.masses[m], 1.0079, abs_tol=1e-3):
            mask_water_hydrogen[m] = True
            mass_hydrogen = torch.tensor(dimos_system.masses[m])
        else:
            mask_water_oxygen[m] = True
            mass_oxygen = torch.tensor(dimos_system.masses[m])
total_mass = dimos_system.masses.sum()

if train:
    timestep = torch.tensor([0.1], requires_grad=True)
    base_loc = torch.tensor([10.0], requires_grad=True)
    protein_masses = torch.tensor(dimos_system.masses[protein_indices]).requires_grad_()

    mask_heavy_atoms = dimos_system.masses > 1.50

    mass_hydrogen.requires_grad_()
    mass_oxygen.requires_grad_()

    folded_normal = dimos.FoldedNormal(base_loc, base_std)

    def loss_fn(hmc_move_object, step, pos, vel):
        initial_potential_energy = hmc_move_object.initial_potential_energy
        initial_total_energy = hmc_move_object.initial_total_energy
        potential_energy = hmc_move_object.integrator.e
        kinetic_energy = dimos.utils.kinetic_energy(vel, hmc_move_object.sys)
        total_energy = potential_energy + kinetic_energy
        delta_energy = total_energy - initial_total_energy
        acceptance_prob = torch.exp(-delta_energy / hmc_move_object.kBT)
        actual_acc_prob = torch.min(acceptance_prob, torch.tensor([1.0]))

        match objective:
            case "dihedral":
                phi_after = get_dihedral_angle(pos, phi_atoms, hmc_move_object.sys.periodic, hmc_move_object.sys.box)
                psi_after = get_dihedral_angle(pos, psi_atoms, hmc_move_object.sys.periodic, hmc_move_object.sys.box)

                phi_initial = get_dihedral_angle(hmc_move_object.initial_pos, phi_atoms, hmc_move_object.sys.periodic, hmc_move_object.sys.box)
                psi_initial = get_dihedral_angle(hmc_move_object.initial_pos, psi_atoms, hmc_move_object.sys.periodic, hmc_move_object.sys.box)

                phi_dist = dimos.periodic_correction(phi_after - phi_initial, 2*torch.pi)
                psi_dist = dimos.periodic_correction(psi_after - psi_initial, 2*torch.pi)

                dist = (phi_dist + psi_dist)**2
            case "energy":
                dist = (potential_energy-initial_potential_energy)**2
            case "movement_heavy":
                jump_distance = pos[mask_heavy_atoms] - hmc_move_object.initial_pos[mask_heavy_atoms]
                pos_dist = dimos.utils.periodic_correction(jump_distance, hmc_move_object.sys.box)
                dist = pos_dist**2.0
            case "movement_heavy_light":
                jump_distance_heavy = pos[mask_heavy_atoms] - hmc_move_object.initial_pos[mask_heavy_atoms]
                pos_dist_heavy = dimos.utils.periodic_correction(jump_distance_heavy, hmc_move_object.sys.box)
                jump_distance_light = pos[~mask_heavy_atoms] - hmc_move_object.initial_pos[~mask_heavy_atoms]
                pos_dist_light = dimos.utils.periodic_correction(jump_distance_light, hmc_move_object.sys.box)
                dist = torch.sum(pos_dist_heavy**2.0) + 0.1*torch.sum(pos_dist_light**2.0)
            case "movement_heavy_light02":
                jump_distance_heavy = pos[mask_heavy_atoms] - hmc_move_object.initial_pos[mask_heavy_atoms]
                pos_dist_heavy = dimos.utils.periodic_correction(jump_distance_heavy, hmc_move_object.sys.box)
                jump_distance_light = pos[~mask_heavy_atoms] - hmc_move_object.initial_pos[~mask_heavy_atoms]
                pos_dist_light = dimos.utils.periodic_correction(jump_distance_light, hmc_move_object.sys.box)
                dist = torch.sum(pos_dist_heavy**2.0) + 0.2*torch.sum(pos_dist_light**2.0)
            case "movement_heavy_light03":
                jump_distance_heavy = pos[mask_heavy_atoms] - hmc_move_object.initial_pos[mask_heavy_atoms]
                pos_dist_heavy = dimos.utils.periodic_correction(jump_distance_heavy, hmc_move_object.sys.box)
                jump_distance_light = pos[~mask_heavy_atoms] - hmc_move_object.initial_pos[~mask_heavy_atoms]
                pos_dist_light = dimos.utils.periodic_correction(jump_distance_light, hmc_move_object.sys.box)
                dist = torch.sum(pos_dist_heavy**2.0) + 0.3*torch.sum(pos_dist_light**2.0)                    
            case "movement_heavy_light05":
                jump_distance_heavy = pos[mask_heavy_atoms] - hmc_move_object.initial_pos[mask_heavy_atoms]
                pos_dist_heavy = dimos.utils.periodic_correction(jump_distance_heavy, hmc_move_object.sys.box)
                jump_distance_light = pos[~mask_heavy_atoms] - hmc_move_object.initial_pos[~mask_heavy_atoms]
                pos_dist_light = dimos.utils.periodic_correction(jump_distance_light, hmc_move_object.sys.box)
                dist = torch.sum(pos_dist_heavy**2.0) + 0.5*torch.sum(pos_dist_light**2.0)
            case "movement_heavy_light08":
                jump_distance_heavy = pos[mask_heavy_atoms] - hmc_move_object.initial_pos[mask_heavy_atoms]
                pos_dist_heavy = dimos.utils.periodic_correction(jump_distance_heavy, hmc_move_object.sys.box)
                jump_distance_light = pos[~mask_heavy_atoms] - hmc_move_object.initial_pos[~mask_heavy_atoms]
                pos_dist_light = dimos.utils.periodic_correction(jump_distance_light, hmc_move_object.sys.box)
                dist = torch.sum(pos_dist_heavy**2.0) + 0.8*torch.sum(pos_dist_light**2.0)    
            case "movement_heavy_light10":
                jump_distance_heavy = pos[mask_heavy_atoms] - hmc_move_object.initial_pos[mask_heavy_atoms]
                pos_dist_heavy = dimos.utils.periodic_correction(jump_distance_heavy, hmc_move_object.sys.box)
                jump_distance_light = pos[~mask_heavy_atoms] - hmc_move_object.initial_pos[~mask_heavy_atoms]
                pos_dist_light = dimos.utils.periodic_correction(jump_distance_light, hmc_move_object.sys.box)
                dist = torch.sum(pos_dist_heavy**2.0) + 1.0*torch.sum(pos_dist_light**2.0)                          
            case "movement_heavy_protein":
                mask = torch.zeros_like(mask_heavy_atoms, dtype=bool)
                mask[protein_indices] = True
                mask_heavy_protein = mask_heavy_atoms & mask
                #print(mask_heavy_protein.sum())
                jump_distance = pos[mask_heavy_protein] - hmc_move_object.initial_pos[mask_heavy_protein]
                pos_dist = dimos.utils.periodic_correction(jump_distance, hmc_move_object.sys.box)
                dist = pos_dist**2.0                    
            case "movement_protein":
                jump_distance = pos[protein_indices] - hmc_move_object.initial_pos[protein_indices]
                pos_dist = dimos.utils.periodic_correction(jump_distance, hmc_move_object.sys.box)
                dist = pos_dist**2.0               

        return -hmc_move_object.weights[step]*actual_acc_prob*torch.sum(dist)/(step+1)
    
    optimizer = torch.optim.Adam([timestep, base_loc, protein_masses, mass_oxygen, mass_hydrogen], lr=0.01)
else:
    def loss_fn(*arg):
        return 0

    print("Loading file:", input_name)
    data = np.loadtxt(input_name)[-1]
    timestep = data[2]
    base_loc = data[3]
    folded_normal = dimos.FoldedNormal(base_loc, base_std)
    protein_masses = torch.tensor(data[4:4+len(protein_indices)]).to(torch.get_default_dtype())
    mass_oxygen = data[-2]
    mass_hydrogen = data[-1]

truncated_folded_normal = dimos.TruncatedDistribution(0.999, folded_normal)

hmc_move = dimos.monte_carlo.STHMC(300.0, timestep, truncated_folded_normal, loss_fn, dimos_system, detach=False)
dummy_simulation = dimos.MCSimulation(dimos_system, positions, [hmc_move], [1.0], 300.0, seed=123456)
print(f"Minimizing energy")
dummy_simulation.minimize_energy(20, print_details=True)
positions = dummy_simulation.pos.clone().detach_().requires_grad_()

mc_simulations = []
for batch in range(batch_size):
    hmc_move = dimos.monte_carlo.STHMC(300.0, timestep, truncated_folded_normal, loss_fn, dimos_system, detach=False)
    mc_simulations.append(dimos.MCSimulation(dimos_system, positions, [hmc_move], [1.0], 300.0, seed=batch))
    mc_simulations[-1].sys.masses[protein_indices] = protein_masses
    mc_simulations[-1].sys.masses[mask_water_hydrogen] = mass_hydrogen
    mc_simulations[-1].sys.masses[mask_water_oxygen] = mass_oxygen


loss = torch.tensor([0.0])
for iteration in range(25_000):
    print(iteration, loss.item(), timestep.item(),  base_loc.item(), *[m.item() for m in protein_masses], mass_oxygen.item(), mass_hydrogen.item())
    print(iteration, loss.item(), timestep.item(),  base_loc.item(), *[m.item() for m in protein_masses], mass_oxygen.item(), mass_hydrogen.item(), file=output_file, flush=True)
    loss = torch.tensor([0.0])
    if train:
        optimizer.zero_grad()
    
    for batch in range(batch_size):
      mc_simulations[batch].step(detach=(not train))
      loss = loss + mc_simulations[batch].move_set[0].loss
    
    if train and iteration>equilibrate:
        for batch in range(batch_size):
            loss += mc_simulations[batch].move_set[0].loss

        loss.backward()
        optimizer.step()
        
        for batch in range(batch_size):
            mc_simulations[batch].move_set[0].timestep = timestep
            mc_simulations[batch].move_set[0].truncated_distribution.discretized_distribution.base_distribution.base_loc = base_loc

            mc_simulations[batch].sys.masses.detach_()

            mc_simulations[batch].sys.masses[protein_indices] = protein_masses
            mc_simulations[batch].sys.masses[mask_water_oxygen] = mass_oxygen
            mc_simulations[batch].sys.masses[mask_water_hydrogen] = mass_hydrogen          
            
            # Apply mass scaling to maintain total mass
            mc_simulations[batch].sys.masses = mc_simulations[batch].sys.masses * (total_mass / mc_simulations[batch].sys.masses.sum())

    for batch in range(batch_size):    
        mc_simulations[batch].detach_()    
        
    for batch in range(batch_size):      
        psi = get_dihedral_angle(mc_simulations[batch].pos, psi_atoms,  mc_simulations[batch].sys.periodic,  mc_simulations[batch].sys.box)
        phi = get_dihedral_angle(mc_simulations[batch].pos, phi_atoms,  mc_simulations[batch].sys.periodic,  mc_simulations[batch].sys.box)
        pot_energy = mc_simulations[batch].measure_potential_energy()
        accept_prob = mc_simulations[batch].move_set[0].acceptance_prob

        output_files_measure[batch].write(f"{psi.item()}\t{phi.item()}\t{pot_energy.item()}\t{accept_prob.item()}\n")
        output_files_measure[batch].flush()

        mc_simulations[batch].sys.write_pdb(mc_simulations[batch].pos, output_files_config[batch])





