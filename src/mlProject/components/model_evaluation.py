import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,accuracy_score
from urllib.parse import urlparse
import numpy as np
import joblib
from mlProject.entity.config_entity import ModelEvaluationConfig
from pathlib import Path
from mlProject.utils.common import save_json


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self,actual, pred):
        acc_score=accuracy_score(actual, pred)
        return acc_score

    
    def save_results(self):

        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]
        
        predicted_qualities = model.predict(test_x)

        acc_score = self.eval_metrics(test_y, predicted_qualities)
        
        # Saving metrics as local
        scores = {"accuracy_score":acc_score}
        save_json(path=Path(self.config.metric_file_name), data=scores)