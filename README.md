# _Enhancing cislunar proximity operations by integrating reinforcement learning into classical relative guidance methods_ #
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?logo=copyright)
![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![MATLAB](https://img.shields.io/badge/MATLAB-R2024b-orange?logo=mathworks)
![Gymnasium](https://img.shields.io/badge/Gymnasium-RL%20Env-green?logo=openai)
![PyTorch](https://img.shields.io/badge/PyTorch-ML-red?logo=pytorch)
![GitHub last commit](https://img.shields.io/github/last-commit/CarloZambaldo/relativeGuidance?logo=github)

## Overview
_relativeGuidance_ is a framework for simulating and analyzing relative guidance algorithms for space applications. This repository contains the code used in my Master's Thesis in Space Engineering at Politecnico di Milano.

### Main Features
- The main framework is coded in python
- The validation code is done with MATLAB 
- Simulation environment is based on Gymnasium, for RL applications
- Integration with Stable-Baselines3 for training RL agent
- Support for multi-phase guidance strategies and different constraints
- Visualization tools for analyzing guidance performance

The code is structured as in figure:
![Code structure and modularity](images/RLFramework.png)

the modularity in the code design allows to change any of the blocks, provided that all the others are fixed accordingly.

### Installation
Ensure you have Python 3.12 or later installed.

Clone the repository:
```
    git clone https://github.com/CarloZambaldo/relativeGuidance.git
    cd relativeGuidance
```
Install:
```
    pip install ./SimEnvRL
```

## Usage
### Configuration
The repository includes three configuration files:
- for python usage in the ```SimEnvRL/config/``` directory:
    -  defining simulation parameters ```env_conf```
    -   RL training settings ```RL_config```
- for MATLAB simulations in the ```matlabScripts/MATLAB/config/``` folder:
    - ```initializeSimulation``` allows to generate the initial conditions for the simulation

### Running a Simulation 
To run a simulation, execute the following command:

```
python3 RLEnv_MC_Eval.py -p [PHASE_ID] -m [MODEL_NAME] -n [N_OF_SIMULATIONS] -s [SEED] -r [RENDERING_BOOL]
```

> [!NOTE]
> only PHASE_ID parameter is required, the others are set to default as follows:
> ```[MODEL_NAME] = "_NO_AGENT_"```
> ```[N_OF_SIMULATIONS] = 1```
> ```[SEED] = None```
> ```[RENDERING_BOOL] = True```

> [!TIP]
> assign the seed only if reproducibility is required, otherwise avoid seeding

### Training the RL Agent
To train a reinforcement learning agent for a given environment, use:
```
python3 RLEnv_Training.py -p [PHASE_ID] -m [NEW_MODEL_NAME] -r [RENDERING_BOOL]
```

> [!WARNING]
> to continue training an agent set the parameter ```--start-from [OLD_AGENT_NAME]```
> HOWEVER: this is highly discouraged if normalisation was set to True when training the previous agent. Indeed, the new training does not load the old normalization!!

### Plotting the results
To plot the results use (MonteCarloInfo) or (MonteCarloPlots) functions available in the (matlabScripts/) folder.
If multiple simulations were run and saved, to extract only one the MATLAB function (extractSimulationData) is available.
> [!IMPORTANT]
> In order to use the MATLAB plots (e.g. see the trajectory in Synodic frame) the ASTRO class is required.
> This can be found in the dedicated GitHub repository: [github.com/CarloZambaldo/OrbitalMechanics](https://github.com/CarloZambaldo/OrbitalMechanics)




### Repository Structure
    relativeGuidance/
    ├── matlabScripts/       # MATLAB scripts for simulations and validation
    │ ├── extractSimulationData.m
    │ ├── MATLAB/            # Core MATLAB functions
    │ │ ├── config/          # Configuration scripts
    │ │ ├── plot/            # Scripts for visualization and plotting
    │ │ ├── ReferenceFrames/ # Coordinate transformation functions
    │ │ ├── relativeDynamicsModels/ # Dynamic models
    │ │ ├── Z_ModelValidationCodes/ # Model validation scripts
    │
    ├── SimEnvRL/           # Custom Gymnasium simulation environment
    │ ├── config/           # Configuration scripts for RL environment
    │ │ ├── env_config.py, RL_config.py, refTraj.mat
    │ ├── envs/             # RL environment classes
    │ │ ├── RLEnvironment.py
    │ ├── generalScripts/   # Core simulation functions
    │ │ ├── check.py, dynamicsModel.py, OBControl.py, OBGuidance.py
    │ │ ├── ReferenceFrames.py, sunPositionVersor.py, wrappers.py
    │ ├── UserDataDisplay/  # Scripts for displaying results
    │ │ ├── plots.py, printSummary.py, see.py
    │ ├── pyproject.toml    # Python environment dependencies
    │ ├── __init__.py
    │
    ├── AgentModels/        # Folder containing trained agent models (not included in this repository)
    ├── Simulations/        # Folder containing Monte Carlo simulations (not included in this repository)
    │
    ├── RLEnv_MC_Eval.py    # Script for evaluating the RL environment via Monte Carlo simulations
    ├── RLEnv_Training.py   # Script for training the RL environment
    ├── LICENSE             # Project license
    └── README.md           # Project documentation

## Contributing
Contributions are welcome! To contribute:
- Fork the repository
- Create a new branch (git checkout -b feature-branch)
- Make your changes and commit (git commit -m "Add new feature")
- Push to your branch (git push origin feature-branch)
- Open a Pull Request

## License
Copyright (c) 2025 Carlo Zambaldo
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For questions or collaborations, please contact Carlo Zambaldo.

## Citation
If you use this work, please cite:
```bibtex
@mastersthesis{Zambaldo2025,
    author    = {Zambaldo, Carlo},
    title     = {Enhancing cislunar proximity operations by integrating reinforcement learning into classical relative guidance methods},
    school    = {Politecnico di Milano},
    url       = {https://www.politesi.polimi.it/handle/10589/234866},
    month     = apr,
    year      = {2025}
}
