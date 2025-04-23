import os
import sys

import numpy as np 
import pandas as pd
import dill
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.exception import CustomException
from src.logger import logging


# -------------
# def load_object(file_path):
#     try:
#         print(f"Loading object from {file_path}")
#         with open(file_path, 'rb') as file_obj: 
#             return dill.load(file_obj)
#     except Exception as e:
#         raise CustomException(f"Error loading object from {file_path}: {str(e)}", sys)


# ----------------

# To save any Python object (like your trained model or preprocessor) to a .pkl file using dill.

def save_object(file_path, obj):
    try:
        # Ensures the folder path exists where the file will be saved.

        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        


        #  Dumps (saves) the object to a file using dill.

        # 👉 Used in:

        # data_transformation.py to save the preprocessor

        # train_model.py to save the best model
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    


#     📈 evaluate_models(xtrain, ytrain, xtest, ytest, models)
# ✅ Purpose:
# To train and evaluate multiple models and return their performance on the test data.

# 🔍 What it does:
# Iterates through each model in the models dictionary

# Trains each model on training data

# Predicts results on both train & test sets

# Calculates R2 score for test set

# Stores the results in a report dictionary

    
def evaluate_models(xtrain,ytrain,xtest,ytest,models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            # Train model
            model.fit(xtrain,ytrain)

            # Predict Training data
            y_train_pred = model.predict(xtrain)

            # Predict Testing data
            y_test_pred =model.predict(xtest)

            # Get R2 scores for train and test data
            train_model_score = r2_score(ytrain,y_train_pred)
            test_model_score = r2_score(ytest,y_test_pred)

            report[list(models.keys())[i]] =  test_model_score

        return report

    except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException(e,sys)
    
    #     📊 model_metrics(true, predicted)
    # ✅ Purpose:
    # To calculate 3 important metrics:

    # MAE = Mean Absolute Error

    # RMSE = Root Mean Squared Error

    # R² Score = Goodness of fit
        
def model_metrics(true, predicted):
    try :
        mae = mean_absolute_error(true, predicted)
        mse = mean_squared_error(true, predicted)
        rmse = np.sqrt(mse)
        r2_square = r2_score(true, predicted)
        return mae, rmse, r2_square
    except Exception as e:
        logging.info('Exception Occured while evaluating metric')
        raise CustomException(e,sys)
    


        #     📋 print_evaluated_results(xtrain, ytrain, xtest, ytest, model)
        # ✅ Purpose:
        # To print a clean comparison of model performance on both train and test sets.

        # 🔍 What it does:
        # Uses model.predict() on both sets

        # Calls model_metrics() to compute errors and scores

        # Prints the metrics for both training and testing phases


    

def print_evaluated_results(xtrain,ytrain,xtest,ytest,model):
    try:
        ytrain_pred = model.predict(xtrain)
        ytest_pred = model.predict(xtest)

        # Evaluate Train and Test dataset
        model_train_mae , model_train_rmse, model_train_r2 = model_metrics(ytrain, ytrain_pred)
        model_test_mae , model_test_rmse, model_test_r2 = model_metrics(ytest, ytest_pred)

        # Printing results
        print('Model performance for Training set')
        print("- Root Mean Squared Error: {:.4f}".format(model_train_rmse))
        print("- Mean Absolute Error: {:.4f}".format(model_train_mae))
        print("- R2 Score: {:.4f}".format(model_train_r2))

        print('----------------------------------')
    
        print('Model performance for Test set')
        print("- Root Mean Squared Error: {:.4f}".format(model_test_rmse))
        print("- Mean Absolute Error: {:.4f}".format(model_test_mae))
        print("- R2 Score: {:.4f}".format(model_test_r2))
    
    except Exception as e:
        logging.info('Exception occured during printing of evaluated results')
        raise CustomException(e,sys)
    

        #     📂 load_object(file_path)
        # ✅ Purpose:
        # To load any saved object like the model or preprocessor from a .pkl file.

        # return dill.load(file_obj)
        # 👉 Used in:

        # predict_pipeline.py to load the saved preprocessor and trained model for making predictions on new user inputs.
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys) 