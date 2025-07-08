# Adaline Classifier with Batch Gradient Descent

This project demonstrates how to implement the Adaline (ADAptive LInear NEuron) algorithm using batch gradient descent in Python, applied to the Iris dataset.

## What is Adaline?
Adaline is a foundational linear classifier that uses a linear activation function and learns weights by minimizing a cost function (sum of squared errors).

## Batch Gradient Descent
Weights are updated after processing the entire dataset in each epoch, ensuring smooth learning but requiring feature scaling for proper convergence.

## Project Structure
- `adaline_gd.py`: Full Python code for AdalineGD
- `README.md`: This file
- `results.pdf`: Notes and output graphs

## How to Run
1. Install requirements: `numpy`, `pandas`, `matplotlib`
2. Run the Python script in your IDE (e.g., PyCharm)
3. Review the generated cost and decision boundary plots

## Key Learnings
- Importance of feature scaling for gradient-based methods
- Difference between perceptron and Adaline
- Visualization of model convergence

## References
- Python Machine Learning by Sebastian Raschka
- [Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris)

---

**Next steps:**  
See the AdalineSGD project for stochastic (sample-wise) learning!
