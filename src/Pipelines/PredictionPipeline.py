import sys
import os
import joblib
from src.Exception import CustomException
from src.Logger import logging
import pandas as pd

class PredictionPipeline:
    def __init__(self):
        pass

    def predict(self, features) :
        try :
            X_test = pd.read_csv(r"C:/ML Projects/Diamond Price Prediction/Artifacts/Modular intermediate datasets/X_test.csv")

            # imputing
            imputer_obj_file_path = os.path.join(r"C:/ML Projects/Diamond Price Prediction/Artifacts/Models/Imputer models", "imputer.pkl")
            imputer = joblib.load(imputer_obj_file_path)
            features_imputed = pd.DataFrame(imputer.transform(features), columns = imputer.get_feature_names_out())
            features_imputed.columns = [col.split('_')[-1] for col in features_imputed.columns]
            features = features_imputed
            
            # Categorical encoding
            categorical_encoder_obj_file_path : str = os.path.join(r"C:/ML Projects/Diamond Price Prediction/Artifacts/Models/Transform models", "categorical_encoder.pkl")
            categorical_encoder = joblib.load(categorical_encoder_obj_file_path)
            features_categorical_encoded = pd.DataFrame(categorical_encoder.transform(features), columns = categorical_encoder.get_feature_names_out())
            features_categorical_encoded.columns = [col.split('_')[-1] for col in features_categorical_encoded.columns]
            features.drop(columns = features_categorical_encoded.columns, axis = 1, inplace = True)
            features_categorical_encoded = pd.concat([features, features_categorical_encoded], axis = 1)
            features = features_categorical_encoded

            # Eliminating some features
            features = features[X_test.columns]

            # Numerical scaling
            numerical_scaler_obj_file_path : str = os.path.join(r"C:/ML Projects/Diamond Price Prediction/Artifacts/Models/Transform models", "numerical_encoder.pkl")
            numerical_scaler = joblib.load(numerical_scaler_obj_file_path)
            features_numerical_scaled = pd.DataFrame(numerical_scaler.transform(features), columns = numerical_scaler.get_feature_names_out())
            features_numerical_scaled.columns = [col.split('_')[-1] for col in features_numerical_scaled.columns]
            features = features_numerical_scaled

            # Feed it to model
            best_model_obj_file_path : str = os.path.join(r"C:/ML Projects/Diamond Price Prediction/Artifacts/Models/Machine Learning models", "best_model.pkl")
            best_model = joblib.load(best_model_obj_file_path)
            pred = best_model.predict(features)

            return pred[0][0]

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
        
class CustomData :
    def __init__(self,
                 carat:float,
                 depth:float,
                 table:float,
                 x:float,
                 y:float,
                 z:float,
                 cut:str,
                 color:str,
                 clarity:str) :
        
        self.carat = carat
        self.depth = depth
        self.table = table
        self.x = x
        self.y = y
        self.z = z
        self.cut = cut
        self.color = color
        self.clarity = clarity

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'carat' : [self.carat],
                'depth' : [self.depth],
                'table' : [self.table],
                'x' : [self.x],
                'y' : [self.y],
                'z' : [self.z],
                'cut' : [self.cut],
                'color' : [self.color],
                'clarity' : [self.clarity]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Custom input is converted to dataframe')
            return df
        
        except Exception as e:
            logging.info('Exception Occured in get_data_as_dataframe')
            raise CustomException(e, sys)