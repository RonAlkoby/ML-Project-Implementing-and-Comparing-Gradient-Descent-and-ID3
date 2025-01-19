# ML-Project-Implementing-and-Comparing-Gradient-Descent-and-ID3

Overview

This project implements and compares two fundamental machine learning algorithms: a single-layer neural network using gradient descent and the ID3 algorithm for decision trees. The objective is to understand their performance and characteristics by applying them to the Wisconsin breast cancer dataset.

Project Structure

Files:

Q1_neural_network.py: Implements a single-layer neural network with gradient descent.

Q2_ID3.py: Implements the ID3 algorithm for decision tree construction.

Visualizations: Includes plots of training and testing loss curves for the neural network and the decision tree visualization (decision_tree.png).

Assignment Report: A PDF summarizing the methodology, results, and a comparison of the two algorithms.

Requirements

The following libraries are required:

Python 3.x

numpy

matplotlib

sklearn

graphviz

Instructions

Part 1: Single-Layer Neural Network with Gradient Descent

Description:

Implements a neural network with sigmoid activation and cross-entropy loss.

Trains the model using gradient descent with different learning rates and weight initializations.

Includes normalization of features and stopping conditions for optimization.

How to Run:

Execute the Q1_neural_network.py script.

Adjust parameters like learning_rate, epochs, and weight initialization in the script if needed.

Outputs:

Training and testing loss curves (visualized).

Final weights, bias, and accuracy of the model.

Results:

Loss curves are saved and displayed to analyze convergence.

Example: Testing accuracy achieved ~95% with an appropriate learning rate of 0.5.

Key Findings:

Reducing the learning rate results in more gradual loss convergence, requiring more epochs for stable results.

Uniform initialization of weights often results in faster convergence compared to normal initialization for this dataset.

Increasing the number of epochs while using a lower learning rate improves accuracy but at the cost of increased computation time.

Example configurations:

Learning Rate: 0.5, Epochs: 500, Accuracy: ~96%.

Learning Rate: 0.1, Epochs: 1000, Accuracy: ~96%.

Part 2: ID3 Algorithm

Description:

Constructs a decision tree using information gain with entropy as the splitting criterion.

Handles numerical data and visualizes the decision tree using Graphviz.

How to Run:

Execute the Q2_ID3.py script.

Outputs:

Visualization of the decision tree (decision_tree.png).

Accuracy on the test dataset.

Results:

Example: The decision tree achieved an accuracy of ~96%.

The decision tree provided interpretable results and effectively split the data using the most significant features at each level.

Part 3: Comparison

Dataset: Wisconsin breast cancer dataset loaded using sklearn.datasets.load_breast_cancer().

Train/Test Split: 80% training, 20% testing.

Evaluation:

Neural Network: Convergence of loss, sensitivity to learning rate and initialization, ability to generalize.

Decision Tree: High interpretability, potential overfitting on smaller datasets.

Observations:

The neural network achieved stable convergence with proper hyperparameter tuning.

The decision tree provided interpretable results but required careful handling to prevent overfitting.

Both algorithms reached ~96% accuracy, but the decision tree excelled in interpretability, while the neural network demonstrated robustness to hyperparameter tuning.

Outputs and Visualizations

Neural Network:

Loss curves for different configurations (e.g., learning rates, weight initializations).

Example plots: 


![005 500](https://github.com/user-attachments/assets/778e53f1-752b-479d-8792-9ebe78e47cac)
![005 200 normal](https://github.com/user-attachments/assets/5c5f917a-3389-4f7e-81d6-b91c93377ab5)
![01 500](https://github.com/user-attachments/assets/520107b4-8d2a-4a7e-9d22-d8d76d6ccd26)




Decision Tree:

Visualization of the decision tree (DT1.jpg):
![DT1](https://github.com/user-attachments/assets/a125ee31-1a9b-4a53-b1db-29db6d1b7190)


Usage Notes

Modify hyperparameters directly in the scripts (learning_rate, epochs, etc.).

Ensure that Graphviz is installed and configured properly to visualize the decision tree.

The scripts are structured for clarity, with comments explaining each step.

Limitations

Neural network implementation is single-layer and does not use libraries like PyTorch or TensorFlow.

Decision tree does not handle categorical data directly (conversion required).

References

Wisconsin Breast Cancer Dataset: Scikit-learn documentation

Gradient Descent and Neural Networks: Online resources and course materials.

ID3 Algorithm: Online tutorials and documentation.

Authors

Please include your names and IDs as per the submission guidelines.

Submission

Ensure all Python scripts, visualizations, and the report are included in the submission folder.


