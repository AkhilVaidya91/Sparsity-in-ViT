# Adversarial perturbation

## Concept

An adversarial perturbation changes input pixels by a small amount to cause a
wrong prediction. This project measures how dense, pruned, and retrained
models behave under untargeted attacks.

The perturbation budget uses the unnormalized image range `[0, 1]`. The model
uses channel-normalized inputs, so the attack converts the budget with the
GTSRB channel standard deviations. Generated images are clipped to valid
normalized bounds.

## Method

The pipeline evaluates each model with both attacks at `2/255`, `4/255`, and
`8/255`.

### FGSM

Fast Gradient Sign Method uses one update:

`x_adv = x + epsilon x sign(gradient(loss, x))`

The gradient uses standard cross-entropy without label smoothing.

### PGD

Projected Gradient Descent repeats the signed-gradient update for 10 steps.
Each step uses an alpha of 25% of epsilon. A random start is enabled. After
each step, the result is projected back into the L-infinity epsilon budget
and the valid image bounds.

For each attack, the code records clean accuracy, adversarial accuracy,
accuracy drop, macro F1, ROC-AUC, and attack success rate. Attack success rate
counts clean-correct samples that become incorrect after the attack.

## Code reference

- FGSM implementation: [`src/adversarial.py`](../src/adversarial.py),
  `fgsm_attack`.
- PGD implementation: [`src/adversarial.py`](../src/adversarial.py),
  `pgd_attack`.
- Attack selection: [`src/adversarial.py`](../src/adversarial.py),
  `adversarial_attack_batch`.
- Metrics: [`src/adversarial.py`](../src/adversarial.py),
  `evaluate_adversarial_model`.
- Result row creation: [`src/adversarial.py`](../src/adversarial.py),
  `adversarial_row`.
- Attack settings: [`src/config.py`](../src/config.py), `ADV_ATTACKS`,
  `ADV_EPSILONS`, `PGD_STEPS`, and `PGD_ALPHA_RATIO`.
- Experiment calls: [`src/experiments.py`](../src/experiments.py),
  `evaluate_pruned_models` and the retraining worker.

