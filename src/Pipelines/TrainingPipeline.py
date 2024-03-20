import os
import sys
from src.Logger import logging
from src.Exception import CustomException
import pandas as pd

from src.Components.DataIngestion import DataIngestion
from src.Components.DataTransformation import DataTransformation
from src.Components.ModelTrainer import ModelTrainer
from src.Components.BestModel import BestModel

if __name__ == '__main__':
    Ingestor = DataIngestion()
    raw_data_path, raw_dropped_renamed_data_path, X_data_path, y_data_path = Ingestor.initiate_data_ingestion()
    #print(raw_data_path, raw_dropped_renamed_data_path, X_data_path, y_data_path)

    Transformer =  DataTransformation()
    X_train_scaled_encoded_data_path, y_train_data_path, X_test_data_path, y_test_data_path, imputer_obj_file_path, scaler_encoder_obj_file_path = Transformer.initiate_data_transformation(X_data_path, y_data_path)
    """
    print(X_train_scaled_encoded_data_path)
    print(y_train_data_path)
    print(X_test_data_path)
    print(y_test_data_path)
    print(imputer_obj_file_path)
    print(scaler_encoder_obj_file_path)
    """
    Trainer = ModelTrainer()
    report_data_path, params_data_path, best_model, best_model_score, X_test_scaled_encoded_data_path = Trainer.initate_model_training(X_train_scaled_encoded_data_path, y_train_data_path, X_test_data_path, y_test_data_path, scaler_encoder_obj_file_path)
    
    model = BestModel()
    model_obj_file_path = model.get_best_model_path(X_train_scaled_encoded_data_path, y_train_data_path)
    print(model_obj_file_path)
    