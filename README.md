# Sign Language Classification
🏆 Accuracy: %93.54

## Overview

This project focuses on classifying sign language images representing the numbers 0 and 1. The model achieves an impressive accuracy rate of 93.54%. This project has applications in sign language recognition and communication.

## Dataset

The dataset used for this project contains images of hand signs for the numbers 0 and 1 in sign language. It includes two categories, one for each number.
Data Link: [Kaggle Link](https://www.kaggle.com/datasets/ardamavi/sign-language-digits-dataset)**
## Project Structure

The project is organized into the following sections:

1. **Library and Input Files:** In this section, we import the necessary Python libraries and load the dataset.

2. **Data Loading and Visualization:** We load the sign language dataset and provide visualizations of the data.

3. **Data Preprocessing:** The dataset is preprocessed and organized for model training.

4. **Train-Test Split:** We split the data into training and testing sets for model evaluation.

5. **Flatten Operation:** Images are flattened to prepare them for training.

6. **Transpose Operation:** Data is transposed to match the required dimensions.

7. **Initialize Weights and Bias:** We initialize the weights and bias for logistic regression.

8. **Sigmoid Function:** The sigmoid function is defined, which is a crucial part of logistic regression.

9. **Forward-Backward Propagation:** Forward and backward propagation steps are explained for logistic regression.

10. **Updating Parameters:** The model parameters, including weights and bias, are updated using the gradient descent algorithm.

11. **Prediction:** We make predictions using the trained model.

12. **Logistic Regression Algorithm:** The logistic regression algorithm is executed, and results are displayed.

## Requirements

To run the project, make sure you have the following Python libraries installed:

- NumPy
- matplotlib
- pandas
- scikit-learn

You can install these libraries using pip:

```bash
pip install numpy
pip install matplotlib
pip install pandas
pip install scikit-learn
```

## Getting Started

### Installation

1. Clone the project repository:

```bash
git clone [https://github.com/Prometheussx/Sign-Language-Classification-Tutorial]
cd Sign-Language-Classification
```

2. Ensure you have Python and the required libraries installed.

### Data Preparation

1. Download the sign language dataset containing images of the numbers 0 and 1 in sign language.

2. Follow the code in the "Data Loading and Visualization" section to load and visualize the dataset.

3. Data Link: [Kaggle Link](https://www.kaggle.com/datasets/ardamavi/sign-language-digits-dataset)
## Usage

### Training the Model

```python
logistic_regression(x_train, y_train, x_test, y_test,learning_rate=0.01 ,num_iterations = 150)
```

### Evaluation

The model's performance is evaluated with metrics such as accuracy, and a sample image with the predicted number is displayed.

![image](https://github.com/Prometheussx/Sign-Language-Classification-Tutorial/assets/54312783/cca6887e-1042-412a-ad1d-c695171f9420)


![image](https://github.com/Prometheussx/Sign-Language-Classification-Tutorial/assets/54312783/def7dbd2-9968-46d1-a41d-d9a86da58dae)


## Results

The project achieves an accuracy of 93.54% in classifying sign language images of numbers 0 and 1. You can monitor the training progress and results by running the provided code.

![image](https://github.com/Prometheussx/Sign-Language-Classification-Tutorial/assets/54312783/7956fe6a-b30a-4b1b-b719-15b1839a5852)


## Author

- Email: [Your_Email_Address](mailto:erdemtahasokullu@gmail.com)
- LinkedIn Profile: [Your LinkedIn Profile](https://www.linkedin.com/in/erdem-taha-sokullu/)
- GitHub Profile: [Your GitHub Profile](https://github.com/Prometheussx)
- Kaggle Profile: [@erdemtaha](https://www.kaggle.com/erdemtaha)

Feel free to reach out if you have any questions or need further information about the project.
