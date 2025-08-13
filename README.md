# coupled-microstrip-DRL-shape-optimisation

## Overview
This project implements a deep reinforcement learning (DRL) agent, specifically using the Soft Actor-Critic (SAC) algorithm, to predict the shape of a potential curve across a PCB coupled microstrip line based on various PCB parameters. This work is a continuation of the previous work with a single microstrip. As in the previous work an hybrid RL-GA algorithm is used for the optimisation task. 
For more detailed information about this study, please take a look at my thesis: [ShapeOptimization-Thesis](https://github.com/pvmodayil/MasterThesis-Shape-Optimisation-Using-DRL).

## Background
The potential curve shape is determined using Thomson's theorem, which states that the potential curve of the microstrip arrangement will be the one with the least energy. This principle serves as the foundation for our optimisation approach.

Unlike the single microstrip arrangement we have two modes in the coupled strip arrangement: odd and even. The model training and ga optimisation are separately done for each mode.

## Hybrid Algorithm
The SAC RL moel is used to predict the control points to draw Bezier curves in the given parameter set. The half width of the arrangement is varied and predcited repeatedly until a clear minimum energy is obatined. The potential curve obtained from this parameter set is then given to the GA optimisation step. A final metric evaluation is done after the GA optimisation step.

The evaluations are done with 30 g-points and 1000 Fourier coefficients.

## Features
- **Deep Reinforcement Learning**: Utilises SAC to predict initial shapes.
- **Genetic Algorithm Optimization**: Implements GA in C++ for enhanced control and faster execution speeds.
  
## Getting Started
To get started with this project, clone the repository and follow these steps:

1. Clone this repository:
   ```bash
   git clone https://github.com/pvmodayil/coupled-microstrip-DRL-shape-optimisation.git
   cd coupled-microstrip-DRL-shape-optimisation
   ```
2. ```bash
    uv run ./src/$mode/agent_train.py # replace $mode with either odd or even
    uv run ./src/$mode/hybrid_algo.py
    ```
3. Build and run
    ```bash
    cd src/$mode/galib # replace $mode with either odd or even
    mkdir build
    cd build
    cmake ..
    cmake --build .
    ./coupledstrip_arrangement.exe
    ```