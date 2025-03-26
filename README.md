# DIMOS

DIMOS (Differentiable Molecular Simulations) is a PyTorch-powered framework for molecular dynamics (MD) and Monte Carlo (MC) simulations, designed to bridge the gap between traditional simulation engines and modern machine learning (ML) workflows. Built for flexibility, performance, and end-to-end differentiability, DIMOS enables seamless integration of classical force fields, machine learning interatomic potentials (MLIPs), and hybrid ML/MM approaches, empowering researchers to innovate in computational chemistry, physics, and biology.

Documentation available at: https://henrik-christiansen.de/dimos

## Installation

For this, first clone the repo to a local folder.

We recommend setting up a new python environment, for example like this (using Python 3.10 because DIMOS is tested on this version):

```bash
python3.10 -m venv .venv/dimos

source .venv/dimos/bin/activate
```

Then, clone the repository and install using pip:

```bash
git clone https://github.com/nec-research/dimos.git

cd dimos

# install without optional dependencies
python -m pip install -e .
```

Additionally, it is possible to install DIMOS with a couple of optional dependencies, such as the MACE and ORB interatomic potentials or tools used for tests/development:

```bash
# install DEVELOPMENT/TEST dependencies
python -m pip install -e '.[dev,mmtools,mace,orb]'
```

To run the test cases based on torchMD, call
```bash
pip install torchmd scipy networkx pandas tqdm pyyaml --no-deps 
```
to install torchMD without the (proprietary) moleculekit dependency.
