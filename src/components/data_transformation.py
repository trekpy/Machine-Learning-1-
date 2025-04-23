import sys
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object  # Corrected redundant import

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation.
        '''
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]



# SimpleImputer(strategy="median")
# → Fills missing values with the median of the column
# (Better than mean when dealing with outliers)

# StandardScaler()
# → Scales data to zero mean and unit variance
# (important for models like Linear Regression, SVM)
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(sparse=False)),  # Set sparse=False to avoid sparse matrix
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            # These are just log messages to help you see what your pipeline is processing.
            #  Useful for debugging.

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

#              This tells your model:
           # “Apply the numerical pipeline to numerical columns,
           #  and the categorical pipeline to categorical columns.”
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        

        # 🧠 PURPOSE:
    # This function does 4 big things:

    # Loads the training and test data from CSVs

    # Separates the input features and target column (math_score)

    # Applies preprocessing (encoding, scaling, etc.)

    # Saves the fitted preprocessor for future use (e.g., prediction)


        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")


        #  Calls the method we explained earlier that returns the ColumnTransformer
        #  (the full preprocessing logic for numerical and categorical features).
            preprocessing_obj = self.get_data_transformer_object()


            #Target variable → What we are predicting (math_score)

            # Numerical features → Used for separating and transforming input features
            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            # Separates the input features (X) and target feature (y)
            #  from both train and test datasets.
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")
 
            #fit_transform() is applied to training data (learns and applies transformation)

            # transform() is applied to test data using the same logic (no refitting)
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)


            # np.c_[] horizontally combines:
            # Transformed features (after scaling & encoding)
            # Target values (math_score)
            # Now train_arr and test_arr are ready for training/testing the model.
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")
            
#             Saves the fitted preprocessor (ColumnTransformer) to disk.
            # This is super important for later prediction — the same preprocessing
            #  must be applied to user input when making predictions.
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
