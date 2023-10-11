# Sign Language Classification using Logistic Regression

## Overview

This project aims to classify sign language images representing the numbers 0 and 1 using a logistic regression model. The dataset contains images of sign language gestures for numbers 0 to 10, but for this project, we focus on the binary classification of 0 and 1.

You can contact me at my email address (erdemtahasokullu@gmail.com) or leave comments under the project if you have any questions or need assistance.

## Computation Graph of Logistic Regression

In logistic regression, we have two main parameters: weights and bias. We compute the output using the following equation:

z = (w.t)x + b

Where:
- z: The result
- w.t: Transpose of the weight matrix
- x: Input features
- b: Bias

We pass the result 'z' through a sigmoid function to obtain 'y_head', which is used for classification.

## Contents

1. Library and Input Files
2. Data Loading and Visualization
3. Data Editing
4. Train Test Split
5. Flatten Operation
6. Transpose Operation
7. Initialize Weights and Bias
8. Sigmoid Function
9. Forward and Backward Propagation
10. Updating (Learning) Parameters
11. Prediction
12. Logistic Regression Algorithm
13. Model Result

## Code Overview

- We load the necessary libraries, input files, and visualize the data.
- We reduce the dataset to images of numbers 0 and 1.
- We split the data into training and testing sets.
- We flatten the data for more efficient learning.
- We initialize weights and bias.
- We define the sigmoid function.
- We perform forward and backward propagation.
- We update the model's parameters using gradient descent.
- We make predictions and evaluate the model.
- The model achieves a test accuracy of 93.54% after 140 iterations.

## Results

The project demonstrates that a logistic regression model can classify sign language images effectively. The cost value decreased significantly as the model was trained, and the accuracy reached 93.54% after 140 iterations. Further training did not significantly improve the model's performance.

Feel free to use this project as a starting point for your own sign language classification tasks. You can also expand it to include all ten classes by modifying the dataset and labels accordingly.

---

*Note: Visual testing of the model is included, allowing you to specify the index of the test image to view the prediction.*

**Author:** [Erdem Taha Sokullu]
