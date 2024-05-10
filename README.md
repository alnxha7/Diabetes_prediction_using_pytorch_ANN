# Diabetes-Prediction
Diabetes is a significant public health concern, with long-term complications that can lead to serious health issues.
Early detection and management are crucial to reduce the burden on healthcare systems and improve patient outcomes. 
Traditional diagnostic methods rely on clinical tests, but there is a growing need for data-driven approaches that can predict diabetes risk based on patient data. 
The objective is to build a robust model that can accurately predict diabetes using a set of clinical and demographic features.

## Project target
To develop a POC using Flask, HTML and CSS for predicting whether a person is suffering from Diabetes or not.

# Data collection
About the Data set:
This is a Deep learning project where we will predict whether a person is suffering from Diabetes or not. The dataset was downloaded from Kaggle.
https://www.kaggle.com/datasets/mathchi/diabetes-data-set
The datasets consists of eight medical predictor variables and one target variable, Outcome.
Predictor variables includes the number of pregnancies the patient has had, their BMI, insulin level, age, and so on. 
The target column says whether the person is having the disease or not based on the predictor variables. The target has two values 1(having the diease) and 0(not having the disease). A binary classification problem statement.

## Project Description:
### Data Preprocessing
* After loading the dataset("diabetes.csv")
* Handle missing values: zero values of every feature was calculated. outliers, and imbalanced data if needed.
* Normalize/standardize numerical features for consistency.
  
### Data Splitting
* The dataset was divided into independent(X) and dependent(y) features. Train test split was performed for getting the train and test datasets.

### Model Selection
* Implement a feedforward Artificial neural network with an appropriate architecture, including input, hidden, and output layers.
* Use activation functions like ReLU or Sigmoid and dropout layers to avoid overfitting.
  
* The Model I created is given Below

![Screenshot_20240510_110215](https://github.com/alnxha7/Diabetes_prediction_using_pytorch_ANN/assets/129566733/53d61382-f651-4392-8a33-2afea6deea96)


### Model Training
* Train the model using a suitable loss function (e.g., Binary Cross-Entropy) and an optimizer like Adam or SGD.
* Use early stopping or regularization techniques to prevent overfitting.
* The "diabetes.ipynb" file contains all these informations.

### Model Evaluation:

#### I got an Accuracy of: 76.623 %
* Perform cross-validation to ensure robustness.
* The plot figure of loss is given below

![Screenshot_20240510_111005](https://github.com/alnxha7/Diabetes_prediction_using_pytorch_ANN/assets/129566733/36dc38fe-1ec1-44a6-83bd-3d4714b8c8c6)

* The confusion matrix i got:

![Screenshot_20240510_112251](https://github.com/alnxha7/Diabetes_prediction_using_pytorch_ANN/assets/129566733/4c9b402c-fa94-4529-991f-079f6091fda9)

##### The final step was to save the model as a pickle file to reuse it again for the Deployment purpose. pickle was used to dump the model at the desired location.

### Deployment
The model was deployed locally. The backend part of the application was made using Flask and for the frotend part HTML and CSS was used. 
The file "app.py" contains the entire flask code and inside the templates folder, "diabetes.html" contains the homepage and "result.html" contains the result page.

### samples of the app

![Screenshot_20240510_111624](https://github.com/alnxha7/Diabetes_prediction_using_pytorch_ANN/assets/129566733/2582607f-d89e-4616-89f6-98b05ad276b6)

![Screenshot_20240510_111740](https://github.com/alnxha7/Diabetes_prediction_using_pytorch_ANN/assets/129566733/9fac342b-f769-4168-b9d9-b3d94008208f)

![Screenshot_20240510_111830](https://github.com/alnxha7/Diabetes_prediction_using_pytorch_ANN/assets/129566733/52a248da-e52f-42d7-83a2-e7b28e123279)


### Technologies and Tools Used
* PyTorch
* Python
* Pandas, NumPy
* Scikit-learn for data preprocessing and metrics
* Flask
* Jupyter Notebooks or other development environments

