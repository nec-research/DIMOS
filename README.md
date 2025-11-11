<p align="center">
<img src="./dimos-logo.svg" alt="DIMOS" width="600"/>
</p>

DIMOS (Differentiable Molecular Simulator) is a PyTorch-based framework for molecular dynamics (MD) and Monte Carlo (MC) simulations, designed to bridge the gap between traditional simulation engines and modern machine learning (ML) workflows. Built for flexibility, performance, and end-to-end differentiability, DIMOS enables seamless integration of classical force fields, machine learning interatomic potentials (MLIPs), and hybrid ML/MM approaches, empowering researchers to innovate in computational chemistry, physics, and biology.

Documentation available at: https://dimos.henrik-christiansen.net

Please cite our paper if you are using DIMOS: 
H. Christiansen, T. Maruyama, F. Errica, V. Zaverkin, M. Takamoto, and F. Alesiani, Fast, Modular, and Differentiable Framework for Machine Learning-Enhanced Molecular Simulations, J. Chem. Phys. 163, 182501 (2025).

For convenience, you may also use the following bibtex entry:
```bash
@article{10.1063/5.0277356,
    author = {Christiansen, Henrik and Maruyama, Takashi and Errica, Federico and Zaverkin, Viktor and Takamoto, Makoto and Alesiani, Francesco},
    title = {Fast, modular, and differentiable framework for machine learning-enhanced molecular simulations},
    journal = {J. Chem. Phys.},
    volume = {163},
    pages = {182501},
    year = {2025},
    url = {https://doi.org/10.1063/5.0277356},
}
```


## Installation

DIMOS can be installed using pip
```bash
pip install dimos-torch
```

Or alternatively by cloning and then installing locally, including optional dependencies, such as the MACE and ORB interatomic potentials or tools used for tests/development:
```bash
git clone https://github.com/nec-research/dimos.git; cd dimos

# install with optional dependencies
python -m pip install -e '.[dev,mmtools,mace,orb]'
```

To run the test cases based on torchMD, also this package needs to be installed. To avoid the installation of the (proprietary) moleculekit dependency, call
```bash
pip install torchmd scipy networkx pandas tqdm pyyaml --no-deps 
```

## Get started

```python
import dimos
system = dimos.AmberForceField("config.prmtop")
integrator = dimos.LangevinDynamics(dt, T, gamma, system)
simulation = dimos.MDSimulation(system, integrator, positions, T)
simulation.step(num_steps)
```
