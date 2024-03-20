import os
import sys
import pandas as pd
import scipy.stats as stats
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import warnings
warnings.filterwarnings('ignore')

from src.Exception import CustomException
from src.Logger import logging

from dataclasses import dataclass

@dataclass
class MissingValueHandlerConfig :
    
    X_imputed_data_path : str = os.path.join(r"C:/ML Projects/Diamond Price Prediction/Artifacts/Modular intermediate datasets", "05_X_imputed.csv")

    imputer_obj_file_path : str = os.path.join(r"C:/ML Projects/Diamond Price Prediction/Artifacts/Models/Imputer models", "imputer.pkl")

class MissingValueHandler :
    def __init__(self):
        self.missing_value_handler_config = MissingValueHandler()
        
    def initiate_missing_value_handler(self, X_data_path, y_data_path, num_columns, cat_columns) :
        logging.info("Entered the Missing value handler component")
        try :
            
            # Load X and y
            X = pd.read_csv(X_data_path)
            y = pd.read_csv(y_data_path)
            logging.info('Read X and y as dataframe')

            # Strategy
            num_columns_strategy = {
                                        'carat' : 'mean',
                                        'depth' : 'median',
                                        'table' : 'mean',
                                        'x' : 'mean',
                                        'y' : 'median',
                                        'z' : 'mean'
                                    }
            
            # Pipeline
            num_imputer_pipeline = Pipeline(steps = [
                                                        ('imputer', ColumnTransformer([
                                                                                        ('impute' + col, SimpleImputer(strategy = strategy), [col]) for col, strategy in num_columns_strategy.items()
                                                                                    ])
                                                        )
                                                        # (imputer, SimpleImputer(strategy = 'mean'))
                                                    ]
                                            )

            cat_imputer_pipeline = Pipeline(steps = [
                                                        ('imputer', SimpleImputer(strategy = 'most_frequent')),
                                                    ]
                                            )

            imputer = ColumnTransformer([
                                            ('num_imputer_pipeline', num_imputer_pipeline, num_columns), 
                                            ('cat_imputer_pipeline', cat_imputer_pipeline, cat_columns)
                                        ])
            
            # Transformation
            X_imputed = pd.DataFrame(imputer.fit_transform(X), columns = imputer.get_feature_names_out())
            X_imputed.columns = [col.split('_')[-1] for col in X_imputed.columns]

            # Saving
            X_imputed.to_csv(self.missing_value_handler_config.X_imputed_data_path, index = False, header = True)
            joblib.dump(imputer, self.missing_value_handler_config.imputer_obj_file_path)

            logging.info('Missing value handler has done its job')

            return (
                        self.missing_value_handler_config.X_imputed_data_path,
                        self.missing_value_handler_config.imputer_obj_file_path
                    )
            
        except Exception as e:
            logging.info("Exception occured in the initiate_missing_value_handler")

            raise CustomException(e,sys)