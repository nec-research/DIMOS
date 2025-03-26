import openmmtools as mmtools
import parmed


file_mapping = open("box_size_mapping.dat","w")
for box_size in range(10,120,5):#(10,15,20,25,30,35,40,50,60,70,80,90,100,110,120):
    waterbox = mmtools.testsystems.WaterBox(box_edge=box_size*parmed.unit.angstrom,constrained=False)

    openmm_topology = waterbox.topology
    openmm_system = waterbox.system
    openmm_positions = waterbox.positions
    print(box_size, openmm_positions.shape[0])
    print(box_size, openmm_positions.shape[0], file=file_mapping)

    structure = parmed.openmm.load_topology(openmm_topology, openmm_system, openmm_positions)

    structure.save(f'amber{box_size}.prmtop', overwrite=True)
    structure.save(f'amber{box_size}.inpcrd', overwrite=True)