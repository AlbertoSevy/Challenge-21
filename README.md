# Alphabet Soup Charity Deep Learning Model

## Overview

This project utilizes deep learning techniques to create a binary classification model for Alphabet Soup, a charitable organization. The goal is to predict whether organizations funded by Alphabet Soup will be successful based on a variety of features.

The analysis is divided into four steps:

1. **Preprocessing the Data**: Preparing the dataset for training.
2. **Compiling, Training, and Evaluating the Model**: Building and evaluating a neural network.
3. **Optimizing the Model**: Enhancing model accuracy through iterative adjustments.
4. **Writing a Report**: Documenting the process and findings.

## Files in the Repository

- **AlphabetSoupCharity.ipynb**: Contains the preprocessing, model creation, training, and evaluation.
- **AlphabetSoupCharity_Optimization.ipynb**: Includes optimization attempts to improve model performance.
- **AlphabetSoupCharity.h5**: Stores the trained model.
- **AlphabetSoupCharity_Optimization.h5**: Stores the optimized model.
- **charity_data.csv**: The dataset used for analysis.

---

## Steps in the Project

### Step 1: Preprocess the Data

1. **Read and Inspect the Dataset**
   - Load `charity_data.csv` into a Pandas DataFrame.
   - Identify:
     - **Target Variable**: `IS_SUCCESSFUL`
     - **Feature Variables**: All other columns except those irrelevant to the analysis.
   - Remove irrelevant columns: `EIN` and `NAME`.

2. **Categorical Data Handling**
   - Examine unique values in categorical columns.
   - Combine rare categories into a new category, `Other`.
   - Encode categorical variables using `pd.get_dummies()`.

3. **Data Splitting and Scaling**
   - Split data into:
     - Features (`X`)
     - Target (`y`).
   - Use `train_test_split` to divide data into training and testing sets.
   - Scale the features using `StandardScaler`.

---

### Step 2: Compile, Train, and Evaluate the Model

1. **Model Architecture**
   - **Input Layer**: Match the number of input features.
   - **Hidden Layers**:
     - First layer: 80 neurons, ReLU activation.
     - Second layer: 30 neurons, ReLU activation.
   - **Output Layer**: 1 neuron, Sigmoid activation.

2. **Training and Evaluation**
   - Compile the model using `binary_crossentropy` as the loss function and `adam` optimizer.
   - Train the model for 100 epochs with a batch size of 32.
   - Save the trained model to `AlphabetSoupCharity.h5`.

---

### Step 3: Optimize the Model

1. **Optimization Strategies**
   - Adjust the input data:
     - Remove additional irrelevant columns.
     - Refine rare category binning.
   - Modify the model:
     - Increase neurons and add hidden layers.
     - Experiment with different activation functions.
   - Adjust training:
     - Modify epochs and batch sizes.

2. **Result**
   - Achieved higher accuracy (target: >75%).
   - Saved optimized model to `AlphabetSoupCharity_Optimization.h5`.

---

### Step 4: Write a Report

1. **Data Preprocessing**
   - **Target Variable**: `IS_SUCCESSFUL`.
   - **Feature Variables**: All columns except `EIN` and `NAME`.
   - **Removed Variables**: `EIN`, `NAME`.

2. **Model Design**
   - Layers:
     - Input layer with 80 neurons.
     - Two hidden layers.
     - Output layer for binary classification.
   - Activation Functions:
     - Hidden layers: ReLU.
     - Output layer: Sigmoid.

3. **Optimization Steps**
   - Increased the number of neurons.
   - Experimented with additional layers.
   - Tuned the number of epochs and batch sizes.

4. **Summary**
   - Final model achieved acceptable accuracy.
   - Recommendation: Consider alternative models (e.g., Random Forest or Gradient Boosting) to compare performance on this dataset.

---

### Step 5: Repository Submission

1. Downloaded the Colab notebooks to the local machine.
2. Added the notebooks, data files, and HDF5 models to the GitHub repository.
3. Pushed all changes to the remote repository.

---

## Tools and Libraries Used

- **Python**: Core programming language.
- **Pandas**: Data manipulation and preprocessing.
- **TensorFlow/Keras**: Model building and training.
- **scikit-learn**: Scaling and splitting the data.
- **Google Colab**: Development environment.

## Conclusion

The Alphabet Soup Charity Deep Learning Model successfully predicts the likelihood of success for funded organizations. While achieving the target accuracy, further exploration with alternative models may yield even better results. The project demonstrates effective preprocessing, model design, and optimization techniques.

