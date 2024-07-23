# deep-learning-challenge
The deep-learning-challenge project aims to classify if applicants for a funding scheme have will likely be successful (1) in  fail (0). The project will achieve this using supervised machine learning techniques-  deep learning neural networks.

## Dependencies:

TensorFlow: For building and training the neural network.
Keras Tuner: For hyperparameter tuning.
Scikit-learn: For data pre-processing and splitting.
Pandas: For reading and manipulating the data

## Installation and Run Instructions:
You may need to install the **scikit-learn** and **keras-tuner** packages before commencing:
	1. Open **Gitbash** and activate your virtual environment.
	2. Check Following installations:
		**conda list scikit-learn**
		**conda list keras-tuner**
		**conda list tensorflow**
		**conda list pandas** 

	3. For all non-available versions run :
		**conda install scikit-learn**
		**conda install keras-tuner**
		**conda install tensorflow**
		**conda install pandas**
	
	4. Confirm installation by running:
		**conda list scikit-learn**
		**conda list keras-tuner**
		**conda list tensorflow**
		**conda list pandas**



## Files:

1. **deep-learning-challenge.ipynb:
	** a python script, The analyses for the initial neural network model.

2. **AlphabetSoupCharity.h5:
	** a HDF5 binary data formatted file for storing results from the initial neural network model.

3. **AlphabetSoupCharity_Optimisation.ipynb:
	** a python script produce the auto-optimised neural network model using keras-tuner.

4. **AlphabetSoupCharity_Optimised.h5:
	** a HDF5 binary data formatted file for storing results from the optimised neural network model.

5. **my_dir**:
	** directory containing all iterations of optimisation trials.

6. **Model_Report&Results**:
	** Contains optimisation methodology and results of optimisation trials, overview summary and recommendations which can also be found below.

7. **Readme_Report**:
	** This file, contains instructions on how to use the repository code, its purpose, results of optimisation trials, overview, summary and recommendations.


## Overview of the Analysis:
The deep-learning-challenge project aims to develop a supervised machine learning model using deep learning neural networks to classify applicants for a funding scheme into those likely to succeed (i.e., 1) or fail (i.e., 0) in their ventures. The goal is to assist decision-makers within the Alphabet Soup Charity in identifying applicants with the highest chances of success, thus improving the allocation of resources and maximizing the impact of their funding efforts. The project involves pre-processing the provided data, building an initial neural network model, optimizing the model using hyperparameter tuning, and evaluating its performance to ensure its efficacy.


Data:
	* **EIN** and **NAME:** identification columns;
	* **APPLICATION_TYPE:** Alphabet Soup application type;
	* **AFFILIATION:** affiliated sector of industry;
	* **CLASSIFICATION:** Government organisation classification;
	* **USE_CASE:** use case for funding;
	* **ORGANIZATION:** organisation type;
	* **STATUS:** active status;
	* **INCOME_AMT:** income classification;
	* **SPECIAL_CONSIDERATIONS:** special considerations for application;
	* **ASK_AMT:** funding amount requested;
	* **IS_SUCCESSFUL:** whether or not the money was used effectively, coded as 1 ("yes") or 0 ("no") (i.e., the target variable).



# Model Building and Evaluation

## Initial Model Steps

1. **Load Data**: 
	Read `charity_data.csv` into a Pandas DataFrame.
2. **Define Target**: 
	Set `IS_SUCCESSFUL` as the target variable.
3. **Define Features**: 
	Exclude `IS_SUCCESSFUL` from feature variables.
4. **De-identify Data**: 
	Drop `EIN` and `NAME` columns.
5. **Analyze Unique Values**: 
	Check the number of unique values in each column.
6. **Group Rare Categories**: 
	For columns with many unique values, combine rare categories into an "Other" category.

7. **Encode Categorical Variables**: 
	Use `pd.get_dummies()` for encoding.

8. **Split Data**: 
	Separate data into features (`X`) and target (`y`), then use `train_test_split` for training and testing sets.

9. **Scale Features**: 
	Standardize features using `StandardScaler()`.

10. **Build Model**: 
	Create a neural network model with TensorFlow and Keras.

11. **Add Layers**: 
	Include hidden layers with appropriate activation functions and an output layer.

12. **Check Model Structure**: 
	Inspect the model architecture.

13. **Compile and Train**: 
	Compile and fit the model.

14. **Save Weights**: 
	Use a callback to save model weights every 5 epochs.

15. **Evaluate Model**: 
	Determine loss and accuracy on test data.

16. **Export Results**: 
	Save the model as `AlphabetSoupCharity.h5`.

## Optimized Model

2. **Keras-Tuner Model Optimisation**:
   - **Search Space**:
     - Number of neurons (1 to 200 in steps of 1)
     - Activation functions (ReLU, Tanh, ELU)
     - Learning rates (0.01, 0.001, 0.0001)
   - **Build and Compile Model**: Defined and compiled the model with hyperparameters.
   - **Tuner Setup**: Used RandomSearch for optimization with validation accuracy as the objective.
   - **Tuning**: Ran the search for optimal hyperparameters with 10 epochs and 3 executions per trial.
   - **Layers**: Built and evaluated the model with hidden layers from 1 hidden layer up to 6 hidden layers.
   - **Final Model**: Built and evaluated with the best hyperparameters.



## Results:
### Data Preprocessing:
* Target variable: 
	**IS_SUCCESSFUL:** The model target variable is the ‘IS_SUCCESSFUL’ column. This variable indicates whether a charity donation request was successful (1) or not (0).

* Features variables: 
	The features of the model are all columns from the data frame after dropping EIN', 'NAME', 'SPECIAL_CONSIDERATIONS’ and IS_SUCCESSFUL. This includes; 			Categorical variables converted to numeric through one-hot encoding and; Numerical variables (INCOME_AMT, ASK_AMT).

* Variables removed from the input data because they are neither targets nor features: **EIN**, **SPECIAL_CONSIDERATIONS** and **NAME**. 


### Overall Model Performance:

#### Initial Model:
The hyperparameters of the initial model are as follows:
	* Number of input features: len(X_train[0]);
	* Input layer: neurons = 80, activation function = relu;
	* Second hidden layer: neurons = 30, activation function = relu;
	* Output layer: neurons = 1, activation function = sigmoid.

Overall, using these hyperparameters, the accuracy of the initial model is 0.73, indicating that it correctly classifies 73% of the instances.


#### Optimised Model:
The hyperparameters of the optimised model are as follows:
	* Number of input features: len(X_train[0]);
	* Input layer: neurons = 50, activation function = tanh;
	* Second hidden layer: neurons = 50, activation function = elu;
	* Third hidden layer: neurons = 150, activation function = relu;
	* Output layer: neurons = 1, activation function = sigmoid.
	* The optimal learning rate for the optimizer is 0.001.


The accuracy of the optimised model is 0.73. Meaning despite tuning the model only correctly classifies 73% of the instances.

## Recommendations
- **Current Model Limitations**:
  - **Model Complexity**: The neural network may not be complex enough to capture intricate data patterns.
  - **Overfitting**: Increased layers and neurons might be causing overfitting, leading to poor generalization.
  - **Feature Engineering**: The current feature set might lack crucial information.

- **Proposed Model**: 
  - **Random Forest Classifier**: Recommended for potentially better accuracy and feature importance insights. This model can handle numerous features and complex interactions, possibly improving classification performance.


## Summary:
The optimised neural network model performs acceptably well in predicting successful grant applicants, with an the overall accuracy of 73%. However, despite auto-optimisation, the model performance remained at 73%, slightly lower than the target performance of 75%. Therefore using a more complex model, such as Random Forest may help improve classification accuracy and explain which variables are the most or least influential when determining whether a funding applicant is likely to succeed or not.

