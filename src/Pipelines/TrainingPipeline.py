import os
import sys
from src.Logger import logging
from src.Exception import CustomException

import pandas as pd
from src.Components.DataIngestor import DataIngestor
from src.Components.MissingValueHandler import MissingValueHandler
from src.Components.Resampler import Resampler
from src.Components.CategoricalEncoder import CategoricalEncoder
from src.Components.LowVarianceEliminator import LowVarianceEliminator
from src.Components.CorrelationEliminator import CorrelationEliminator
from src.Components.TrainTestSplitter import TrainTestSplitter
from src.Components.NumericalScaler import NumericalScaler
from src.Components.ModelTrainer import ModelTrainer

if __name__ == '__main__' :

    # Data ingestion
    data_ingestor = DataIngestor()
    raw_data_path, raw_dropped_renamed_data_path, X_data_path, y_data_path = data_ingestor.initiate_data_ingestor()

    # Column division
    X = pd.read_csv(X_data_path)
    y = pd.read_csv(y_data_path)
    num_columns = X.select_dtypes(include = 'number').columns.tolist()
    cat_columns =  X.select_dtypes(include = 'object').columns.tolist()
    
    cat_nominal_columns = []
    cat_ordinal_columns = ['cut', 'color', 'clarity']
    cat_target_ordinal_columns = []

    # Order is from low to high
    cut_cat = ['Fair', 'Good', 'Very Good','Ideal', 'Premium']
    color_cat = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
    clarity_cat = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

    # Handling missing values
    missing_value_handler = MissingValueHandler()
    X_imputed_data_path, imputer_obj_file_path = missing_value_handler.initiate_missing_value_handler(X_data_path, y_data_path, num_columns, cat_columns)
    X_data_path = X_imputed_data_path

    """
    # Resample the dataset
    resampler = Resampler()
    X_resampled_data_path, y_resampled_data_path = resampler.initiate_resampler(X_data_path, y_data_path)
    X_data_path = X_resampled_data_path
    y_data_path = y_resampled_data_path
    """

    # Outliers


    # Categorical Encoding
    categorical_encoder = CategoricalEncoder()
    X_categorical_encoded_data_path, categorical_encoder_obj_file_path = categorical_encoder.initiate_categorical_encoder(X_data_path, cat_columns, cat_nominal_columns, cat_ordinal_columns, cat_target_ordinal_columns, cut_cat, color_cat, clarity_cat)
    X_data_path = X_categorical_encoded_data_path

    # Eliminate low variance features
    low_variance_eliminator = LowVarianceEliminator()
    X_removed_low_variance_data_path, constant_columns  = low_variance_eliminator.initiate_low_variance_eliminator(X_data_path)
    X_data_path = X_removed_low_variance_data_path

    for col in constant_columns :
        if col in num_columns :
            num_columns.remove(col)
        elif col in cat_columns :
            cat_columns.remove(col)
        elif col in cat_nominal_columns :
            cat_nominal_columns.remove(col)
        elif col in cat_ordinal_columns :
            cat_ordinal_columns.remove(col)
        elif col in cat_target_ordinal_columns :
            cat_target_ordinal_columns.remove(col)
    
    # Eliminate features on the basis of Correlation
    correlation_eliminator = CorrelationEliminator()
    X_MIF_data_path, features_to_be_dropped = correlation_eliminator.initiate_correlation_eliminator(X_data_path, y_data_path)
    X_data_path = X_MIF_data_path

    for col in constant_columns :
        if col in num_columns :
            num_columns.remove(col)
        elif col in cat_columns :
            cat_columns.remove(col)
        elif col in cat_nominal_columns :
            cat_nominal_columns.remove(col)
        elif col in cat_ordinal_columns :
            cat_ordinal_columns.remove(col)
        elif col in cat_target_ordinal_columns :
            cat_target_ordinal_columns.remove(col)

    # Train test split
    train_test_splitter = TrainTestSplitter()
    X_train_data_path, y_train_data_path, X_test_data_path, y_test_data_path = train_test_splitter.initiate_train_test_splitter(X_data_path, y_data_path)

    # Numeric scaling
    numerical_scaler = NumericalScaler()
    X_train_scaled_data_path, numerical_scaler_obj_file_path = numerical_scaler.initiate_numerical_scaler(X_train_data_path)
    X_train_data_path = X_train_scaled_data_path

    # Model training
    model_trainer = ModelTrainer()
    X_test_scaled_data_path, report_data_path, params_data_path,  best_model, best_model_score = model_trainer.initate_model_training(X_train_data_path, y_train_data_path, X_test_data_path, y_test_data_path, numerical_scaler_obj_file_path)
    print(f"Best model = {best_model} => score = {best_model_score}")