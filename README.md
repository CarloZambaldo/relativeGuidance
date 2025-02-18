# _Employment of Reinforcement Learning to support state of the art Relative Guidance Methods_ #

## Overview
_relativeGuidance_ is a framework for simulating and analyzing relative guidance algorithms for space applications. This repository contains the code used in my Master's Thesis in Space Engineering at Politecnico di Milano.

### Features
- The main code is coded in python
- The validation code is 
- Simulation environment based on Gymnasium for RL applications
- Integration with Stable-Baselines3 for training RL-based controllers
- Support for multi-phase guidance strategies and different constraints
- Visualization tools for analyzing guidance performance

The code is structured as in figure:
![Code structure and modularity](images/RLFramework.png){width=10}

the modularity allows to change any 

### Installation
Ensure you have Python 3.10 or later installed. Clone the repository and install dependencies:

### Clone the repository
    git clone https://github.com/CarloZambaldo/relativeGuidance.git
    cd relativeGuidance

### Install dependencies
    pip install -r requirements.txt

Usage

### Running a Simulation 
To run a simulation, execute the following command:

```python3 RLEnv_MC_Eval.py -p [PHASE_ID] -m [MODEL_NAME] -n [N_OF_SIMULATIONS] -s [SEED] -r [RENDERING_BOOL]```

note that only PHASE_ID parameter is required, the others are set to default as follows:
 [MODEL_NAME] = "_NO_AGENT_"
 [N_OF_SIMULATIONS] = 1
 [SEED] = None
 [RENDERING_BOOL] = True

# Training the RL Agent
To train a reinforcement learning agent for guidance optimization, use:

```python RLEnv_Training.py```

# Configuration
The repository includes multiple configuration files in the configs/ directory, defining simulation parameters, RL training settings, and spacecraft dynamics.


# Repository Structure
relativeGuidance/
├── **SimEnvRL/**         # Custom simulation environments
├── **matlabScripts/**    # MATLAB scripts for additional analysis
├── **configs/**          # Configuration files for different scenarios
├── **models/**           # Pre-trained models and checkpoints
├── **RLEnv_MC_Eval.py**  # Script for evaluating the RL environment
├── **RLEnv_Training.py** # Script for training the RL environment
└── **README.md**         # This file

Contributing

Contributions are welcome! To contribute:
- Fork the repository.
- Create a new branch (git checkout -b feature-branch).
- Make your changes and commit (git commit -m "Add new feature").
- Push to your branch (git push origin feature-branch).
- Open a Pull Request.

# License
This project is licensed under the MIT License - see the LICENSE file for details.

# Contact
For questions or collaborations, please contact Carlo Zambaldo.