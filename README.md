# Higgs Boson Signal Detection

A project from my statistical analysis course at MIT where we used real LHC collision data to try to detect the Higgs boson through hypothesis testing.

The core idea: since we can't observe the Higgs directly, we trained a neural network on Monte Carlo simulation data to approximate the Neyman-Pearson likelihood ratio — basically the theoretically optimal test statistic for this kind of problem. We then used that learned statistic to compare the experimental data against the background distribution.

For the actual hypothesis test, I ruled out a CLT-based approach after the Lilliefors test showed the test statistic wasn't normally distributed under the null. Instead I implemented a Wilcoxon rank-sum test, deriving the variance of the U statistic from scratch and using an efficient rank-based implementation.

## Stack
Python, PyTorch, NumPy, SciPy, statsmodels
