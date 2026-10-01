# Neural Population Dynamics RNN

An exploratory PyTorch recurrent neural network (RNN) for modeling trial-by-trial neural population dynamics from time-varying behavioral and task inputs.

## Overview

This project explores whether recurrent neural networks can learn latent dynamics that jointly capture neural population activity and behavioral outcomes during a sensory decision-making task.

The models take time-varying behavioral and task variables as inputs and predict:

- Neural activity across hundreds of simultaneously recorded neurons
- Trial-level behavioral outcomes

The project also compares unconstrained recurrent networks with biologically inspired connectivity constraints.

## Model Architecture

The model is a rate-based RNN implemented in PyTorch. Hidden states evolve over time according to the current task/behavioral inputs and recurrent network activity.

Three model variants are evaluated:

- **M0 — Behavioral RNN:** Predicts behavioral outcome from the recurrent state.
- **M1 — Neural + Behavioral RNN:** Jointly predicts behavioral outcome and time-resolved neural population activity.
- **M2 — Biologically Constrained RNN:** Extends M1 with Dale's law–inspired recurrent connectivity, constraining latent units to have consistently excitatory or inhibitory outgoing connections.

## Training

M1 and M2 are trained using a multi-objective loss combining:

- Cross-entropy loss for behavioral prediction
- Activity-weighted mean squared error for neural prediction

The activity-weighted neural objective places greater emphasis on periods containing neural activity in the sparse neural recordings.

Models are evaluated on held-out cross-validation folds.

## Evaluation

Evaluation includes:

- Neuron-wise Pearson correlation between predicted and observed neural activity
- Trial-wise neural prediction correlations
- Prediction calibration and activity bias
- Behavioral accuracy and balanced accuracy
- Confusion matrices
- Comparison of unconstrained and biologically constrained architectures across held-out folds

## Tools

- **Python**
- **PyTorch**
- **NumPy**
- **SciPy**
- **pandas**
- **scikit-learn**

## Project Status

This is an **exploratory research project under active development**. Current work focuses on cross-validated model comparison, neural prediction calibration, and understanding how biological connectivity constraints affect learned population dynamics.

## Motivation

The broader goal is to investigate whether recurrent neural networks can provide an interpretable computational model linking behavioral and sensory variables to population-level neural dynamics, while testing how biologically motivated architectural constraints affect model behavior and predictive performance.
