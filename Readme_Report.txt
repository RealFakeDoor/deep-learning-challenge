# deep-learning-challenge
The deep-learning-challenge project aims to identify if applicants for a funding scheme can be classified into those that have the best chance of success in their ventures (i.e., 1) or failure (i.e., 0) using supervised machine learning techniques, specifically using deep learning neural networks.

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

1. **deep-learning-model.ipynb:
	** a python script, The analyses for the initial neural network model.

2. **AlphabetSoupCharity.h5:
	** a HDF5 binary data formatted file for storing results from the initial neural network model.

3. **AlphabetSoupCharity_Optimisation.ipynb:
	** a python script produce the auto-optimised neural network model using keras-tuner.

4. **AlphabetSoupCharity_Optimisation.h5:
	** a HDF5 binary data formatted file for storing results from the optimised neural network model.


## Overview of the Analysis:
The overall purposes of this analysis is to understand whether a parsimonious model can be built to assist decision-makers within the Alphabet Soup Charity with classifying potential future applicants into those likely to succeed in their ventures (i.e., successful; 1) or not (i.e., unsuccessful; 0). To assist with this, the Alphabet Soup Charity provided data on 34,000 organisations that have previously received funding, 18,261 (53.24%) of which were classified as successful (i.e., 1), and 16,038 (46.76%) of which were classified as unsuccessful (i.e., 0).   


Several potential predictors of funding success were also included in this training dataset:
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



### Initial Model:
The following steps were used to build and evaluate the performance of the initial model:
1. Read in the **charity_data.csv** to a Pandas DataFrame;
2. Define the target variable (i.e., **IS_SUCCESSFUL**);
3. Define the features variables (i.e., excluding the **IS_SUCCESSFUL** column);
4. Drop the **EIN** and **NAME** columns to deidentify the dataset;
5. Determine the number of unique values for each column;
6. For columns that have more than 10 unique values, determine the number of data points for each unique value;
7. Use the number of data points for each unique value to pick a cutoff point to combine "rare" categorical variables together in a new value, "Other";
8. Use **pd.get_dummies() function** to encode categorical variables.
9. Split the preprocessed data into a features array, X, and a target array, y;
10. Use these arrays and the **train_test_split function** to split the data into training and testing datasets;
11. Scale the training and testing features datasets by creating a **StandardScaler() function**, fitting it to the training data, then using the transformed function in subsequent steps.
12. Create a neural network model by assigning the number of input features and nodes for each layer using **TensorFlow** and **Keras**;
13. Create the first hidden layer using an appropriate activation function;
14. If necessary, create a second hidden layer using an appropriate activation function;
15. Create an output layer using an appropriate activation function;
16. Check the structure of the model;
17. Compile and train the model; 
18. Create a callback that saves the initial model's weights every five epochs;
19. Evaluate the model using the test data to determine the loss and accuracy, and;
20. Save and export results to an HDF5 file, named **AlphabetSoupCharity.h5**.

### Optimised Model:
The following steps were used to build and evaluate the performance of the initial model:
1. Follow Steps 1-11 as above;
2. Using **keras-tuner** create an auto-optimiser function to determine the best combination of hidden layers (i.e., layers), neurons per layer (i.e., inputs), and activation furnctions (i.e., activations) (i.e., the hyperparameters), ranking these by their accuracy performance;
3. Compile and train the model with the optimal combination of hyperparameters; 
18. Create a callback that saves the optimised model's weights every five epochs;
19. Evaluate the model using the test data to determine the loss and accuracy, and;
20. Save and export results to an HDF5 file, named **AlphabetSoupCharity_Optimisation.h5**.


## Results:
### Data Preprocessing:
* Target variable: **IS_SUCCESSFUL:** indicates whether or not the money was used effectively, coded as 1 ("yes") or 0 ("no").
* Features variables: listed within **Overview of the Analysis** section above.
* Variables removed from the input data because they are neither targets nor features: **EIN** and **NAME**. 


### Compiling, Training, and Evaluating the Model"
* Number of neurons, layers, and activation functions used: detailed in the **Overall Model Performance** section separately for the initial model and the optimised model.
* Ability to achieve the target model performance: as detained in the **Overal Model Performance** section, despite optimisation using **keras-tuner** the performance of the optimal model remained 73%, lower than the target model performance of 75%.
* Steps taken to increase model performance: as detailed in the **Overal Model Performance** section, after optimisation the number of hidden layers was increased from 2 to 6, the number of neurons per layer was reduced from between 30-80 to between 5-9, and the activation functions were reduced in complexity from relu to sigmoid.


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
* Input layer: neurons = 7, activation function = sigmoid;
* Second hidden layer: neurons = 9, activation function = sigmoid;
* Third hidden layer: neurons = 5, activation function = sigmoid;
* Fourth hidden layer: neurons = 7, activation function = sigmoid;
* Fifth hidden layer: neurons 9, activation function = sigmoid;
* Sixth hidden layer: neurons = 9, activation function = sigmoid;
* Output layer: neurons = 1, activation function = sigmoid.

Overall, the accuracy of the optimised model is also 0.73, indicating that, despite tuning, the model still only correctly classifies 73% of the instances.


## Summary:
The optimised neural network model performs acceptably well in predicting successful grant applicants, with an the overall accuracy of 73%. However, despite auto-optimisation, the model performance remained at 73%, slightly lower than the target performance of 75%. Given this, it may be approriate to attempt classification using a more complex model, such as Random Forest. This method not only will help improve classificaiton accuracy, it may also help in the identification of which variables are explain the most or least in determining whether a funding applicant is likely to succeed in their venture or not.


## Credits:
This code was compiled and written by me for the deep-learning-challenge project in the 2024 Data Analytics Boot Camp hosted by Monash University. Additional credits are declared below.

### Saving model outputs as HDF5 file:
https://www.tensorflow.org/tutorials/keras/save_and_load#hdf5_format (Accessed 22 July 2024).

### keras tuner for auto-optimisation of model hyperparameters:
https://keras.io/guides/keras_tuner/getting_started/ (Accessed 22 July 2024).