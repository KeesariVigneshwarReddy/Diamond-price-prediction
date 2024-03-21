import os
import sys
import copy
import warnings
warnings.filterwarnings('ignore')
from dataclasses import dataclass

from src.Exception import CustomException
from src.Logger import logging

import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

@dataclass
class CategoricalEncoderConfig :
    
    X_categorical_encoded_data_path : str = os.path.join(r"C:/ML Projects/Diamond Price Prediction/Artifacts/Modular intermediate datasets", "X_categorical_encoded.csv")

    categorical_encoder_obj_file_path : str = os.path.join(r"C:/ML Projects/Diamond Price Prediction/Artifacts/Models/Transform models", "categorical_encoder.pkl")

class CategoricalEncoder :
    def __init__(self):
        self.categorical_encoder_config = CategoricalEncoderConfig()

    def initiate_categorical_encoder(self, X_data_path, cat_columns, cat_nominal_columns, cat_ordinal_columns, cat_target_ordinal_columns, cut_cat, color_cat, clarity_cat) :
        logging.info("Entered the Categorical Encoder component")
        try :
            
            # Load X
            X = pd.read_csv(X_data_path)
            logging.info('Read X as dataframe')

            # Pipeline
            cat_nominal_encoder_pipeline = Pipeline(steps = [
                                                                ('OH_encoder', OneHotEncoder())
                                                            ]
                                                    )
            cat_ordinal_encoder_pipeline = Pipeline(steps = [
                                                                ('ordinalencoder', OrdinalEncoder(categories = [cut_cat, color_cat, clarity_cat])), # order should be same as in dataframe
                                                            ]
                                                    )
            categorical_encoder = ColumnTransformer([ 
                                                            ('cat_nominal_pipeline', cat_nominal_encoder_pipeline, cat_nominal_columns),
                                                            ('cat_ordinal_pipeline', cat_ordinal_encoder_pipeline, cat_ordinal_columns)  
                                                    ])
            
            X_categorical_encoded = pd.DataFrame(categorical_encoder.fit_transform(X), columns = categorical_encoder.get_feature_names_out())
            X_categorical_encoded.columns = [col.split('_')[-1] for col in X_categorical_encoded.columns]

            joblib.dump(categorical_encoder, self.categorical_encoder_config.categorical_encoder_obj_file_path)
            X.drop(columns = X_categorical_encoded.columns, axis = 1, inplace = True)
            X_categorical_encoded = pd.concat([X, X_categorical_encoded], axis = 1)
            X_categorical_encoded.to_csv(self.categorical_encoder_config.X_categorical_encoded_data_path, index = False, header = True)
            logging.info("Categorical columns of X are encoded")

            return (
                        self.categorical_encoder_config.X_categorical_encoded_data_path,
                        self.categorical_encoder_config.categorical_encoder_obj_file_path
                    )
            
        except Exception as e:
            logging.info("Exception occured in the initiate_categorical_encoder")

            raise CustomException(e,sys)