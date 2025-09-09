
# Linear Regression Assignment — Submission

**Student Name:** _<RAHUL KUDURU>_  
**Student ID:** _<700763240>_  
**Course / Section:** _<MACHINE LEARNING >_  
**Date:** _<09/08/2025>_

---

## What this repository contains
- `linear_regression_gd_vs_closed_form.py` — Fully commented Python script implementing the programming task (Task 7).  
- `plots/data_and_fits.png` — Scatter plot with closed-form and gradient descent fitted lines.  
- `plots/loss_curve.png` — Gradient descent loss (MSE) vs iteration plot.  

> **How to run**
> ```bash
> python linear_regression_gd_vs_closed_form.py
> ```
> This will print the learned parameters and (re)create plots in `plots/`.

---

## Task 1 — Function Approximation by Hand

Dataset: (x,y) = {(1,1), (2,2), (3,2), (4,5)}  
Model: y = θ₀ + θ₁ x

**Model θ = (1, 0):**
- Predictions: [1.0, 1.0, 1.0, 1.0]
- Residuals (y - ŷ): [0.0, 1.0, 1.0, 4.0]
- Squared residuals: [0.0, 1.0, 1.0, 16.0]
- **MSE:** 4.5000

**Model θ = (0.5, 1):**
- Predictions: [1.5, 2.5, 3.5, 4.5]
- Residuals (y - ŷ): [-0.5, -0.5, -1.5, 0.5]
- Squared residuals: [0.25, 0.25, 2.25, 0.25]
- **MSE:** 0.7500

**Better fit:** θ=(0.5, 1) has lower MSE (0.7500 vs 4.5000).

---

## Task 2 — Random Guessing Practice

The cost function text in the prompt is truncated. I assumed a simple quadratic with its minimum at (0.3, 0.7):  
**J(θ₀, θ₁) = (θ₀ − 0.3)² + (θ₁ − 0.7)².**

- J(0.1, 0.2) = 0.2900  
- J(0.5, 0.9) = 0.0800

Distance to minimum (0.3, 0.7):  
- ‖(0.1, 0.2) − (0.3, 0.7)‖ = 0.5385  
- ‖(0.5, 0.9) − (0.3, 0.7)‖ = 0.2828

**Closer guess:** (0.5, 0.9)

**Why random guessing is inefficient (2–3 sentences):**  
Random guessing ignores the shape of the cost function and provides no systematic way to improve. Each guess is independent of feedback from previous guesses, so convergence to the minimum is slow and unreliable. Gradient-based methods use local slope information to move directly downhill, requiring far fewer evaluations.

---

## Task 3 — First Gradient Descent Iteration

Dataset: (1,3), (2,4), (3,6), (4,5)  
Start θ⁽⁰⁾ = (0, 0), α = 0.01.  
Using MSE: J(θ) = (1/m) Σ (y − (θ₀ + θ₁x))², gradients:
∂J/∂θ₀ = −(2/m) Σ r,  ∂J/∂θ₁ = −(2/m) Σ (x r).

**At θ⁽⁰⁾ = (0,0):**
- Predictions: [0.0, 0.0, 0.0, 0.0]
- Residuals r = y − ŷ: [3.0, 4.0, 6.0, 5.0]
- Σr = 18.0000,  Σ(xr) = 49.0000
- ∇J(θ⁽⁰⁾) = (-9.000000, -24.500000)
- Update: θ⁽¹⁾ = θ⁽⁰⁾ − α ∇J = (0.090000, 0.245000)
- J(θ⁽⁰⁾) = 21.500000,  J(θ⁽¹⁾) = 15.256037

**Second iteration (Continue):**
- At θ⁽¹⁾ = (0.090000, 0.245000):
  - Predictions: [0.33499999999999996, 0.58, 0.825, 1.07]
  - Residuals r: [2.665, 3.42, 5.175, 3.9299999999999997]
  - Σr = 15.1900,  Σ(xr) = 40.7500
  - ∇J(θ⁽¹⁾) = (-7.595000, -20.375000)
  - Update: θ⁽²⁾ = (0.165950, 0.448750)
  - J(θ⁽¹⁾) = 15.256037,  J(θ⁽²⁾) = 10.922289

---

## Task 4 — Compare Random Guessing vs Gradient Descent

Dataset: (1,2), (2,2), (3,4), (4,6)  
MSE for random guesses:
- θ = (0.2, 0.5): J = 5.515000
- θ = (0.9, 0.1): J = 7.935000

One GD step from θ=(0,0), α=0.01 → θ⁽¹⁾ = (0.070000, 0.210000):  
- J(θ⁽¹⁾) = 10.509150

**Which is lower?** Gradient Descent after one step gives J = 10.509150, which is not lower than both random guesses.  
**Why?** GD uses gradient information to move in the direction of steepest descent, quickly reducing error compared to unguided random guesses.

---

## Task 5 — Recognizing Underfitting and Overfitting

**Observation:** Training error is very high; Test error is also very high.  
**Answer:** This is **underfitting**.  
**Why it happens:** The model is too simple (high bias) to capture the underlying relationship, or features are insufficient/poorly engineered.  
**Two fixes:** (1) Use a more expressive model (add features or increase model capacity). (2) Reduce regularization and/or perform feature engineering and data cleaning to better capture the signal.

---

## Task 6 — Comparing Models

- **Model A** (near-perfect on train, poor on test): **Overfitting**.  
  - **Bias–Variance:** Low bias, high variance.  
  - **Improve:** Add regularization (L2/L1), simplify the model, collect more data, use cross-validation, early stopping.
- **Model B** (poor on train and test): **Underfitting**.  
  - **Bias–Variance:** High bias, low variance.  
  - **Improve:** Increase model capacity/complexity, add informative features, reduce regularization.

---

## Task 7 — Programming: Closed-Form vs Gradient Descent

Run:
```bash
python linear_regression_gd_vs_closed_form.py
```

This generates the required plots in `plots/` and prints final parameters:

- Closed-form solution: intercept=2.690841, slope=4.131842
- Gradient Descent:     intercept=2.690841, slope=4.131842
- Difference (abs):     intercept=0.000000, slope=0.000000

**Comment:** In practice, Gradient Descent converges very close to the closed-form solution on this convex quadratic; small differences come from the learning rate, iteration limit, and randomness in the dataset.

---

## Submission Checklist

- [x] Source code pushed to GitHub (include this README and the `.py` file).  
- [x] README updated with **student info** and a brief explanation.  
- [x] Plots generated and included in the repo (`plots/`).  
- [x] Video walkthrough (screen recording of running the script and explaining results).  
- [x] Code commented appropriately.

