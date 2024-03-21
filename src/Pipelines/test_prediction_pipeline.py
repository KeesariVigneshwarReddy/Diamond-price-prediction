from src.Pipelines.PredictionPipeline import PredictionPipeline, CustomData
import numpy as np
myData = CustomData(carat = 1.01, depth = 61.8, table = 58.0, x = 6.44, y = 6.37, z = 3.96, cut = "Premium", color = "G", clarity = "VVS1")
# Expected = 8701

features = myData.get_data_as_dataframe()

prediction_pipeline = PredictionPipeline()
pred = prediction_pipeline.predict(features)
print(f"Actual = 8701, Predicted = {pred}")