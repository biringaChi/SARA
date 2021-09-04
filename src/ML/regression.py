import os
import csv
import sys
from typing import Dict
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))
from stylometry.stylometry import Extractor

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline


class BuildData(Extractor):
    def __init__(self, FILE_PATH1: str = os.getcwd() + "/src/ML/data/addressbook_processed.csv", FILE_PATH2: str = os.getcwd() + "/data/dspace_processed.csv" ) -> None:
        super().__init__()
        
        os.makedirs(os.path.dirname(FILE_PATH1), exist_ok = True)
        os.makedirs(os.path.dirname(FILE_PATH2), exist_ok = True)
        self.FILE_PATH1 = FILE_PATH1
        self.FILE_PATH2 = FILE_PATH2
    
    def addressbook_processed(self) -> DataFrame:
        collector = []
        
        for features in self.extract():
            collector.append(features)
        collector.append(self.get_test())

        try:
            with open(self.FILE_PATH1, "w") as file:
                with file:
                    write = csv.writer(file)
                    write.writerows(collector)
        except OSError as e:
            raise e

        addressbook = pd.read_csv(self.FILE_PATH1, header = None)
        addressbook = addressbook.T.reset_index()
        addressbook = addressbook.drop("index", axis = 1)
        columns = self.get_feature_names()
        columns.append("UnitTest(sec)")

        for i in range(len(columns)):
            addressbook.rename(columns = {i : columns[i]}, inplace = True)
        addressbook = addressbook.drop("Tabs", axis = 1)

        return addressbook

    def get_addreesbook(self) -> DataFrame:
        return self.addressbook_processed()

    def dspace_processed(self) -> DataFrame:
        collector = []
        
        for features in self.extract():
            collector.append(features)
        collector.append(self.get_unit_test())
        collector.append(self.get_integration_test())
        collector.append(self.get_unit_integration_test())

        try:
            with open(self.FILE_PATH2, "w") as file:
                with file:
                    write = csv.writer(file)
                    write.writerows(collector)
        except OSError as e:
            raise e

        dspace = pd.read_csv(self.FILE_PATH2, header = None)
        dspace = dspace.T.reset_index()
        dspace = dspace.drop("index", axis = 1)
        columns = self.get_feature_names()
        columns.append("UnitTest(sec)", "IntegrationTest(sec)", "UnitIntegrationTest(sec)")

        for i in range(len(columns)):
            dspace.rename(columns = {i : columns[i]}, inplace = True)
        dspace = dspace.drop("Tabs", axis = 1)

        return dspace

    def get_dspace(self) -> DataFrame:
        return self.dspace_processed()
    
class RegressionPipeline(BuildData):
    def __init__(self, FILE_PATH1: str, FILE_PATH2: str) -> None:
        super().__init__(FILE_PATH1 = FILE_PATH1, FILE_PATH2 = FILE_PATH2)

        self.MODELS = [LinearRegression(), 
		make_pipeline(StandardScaler(), SGDRegressor(max_iter = 1000, tol = 1e-3)), 
		Ridge(alpha = 1, solver = "cholesky"),
		Lasso(alpha = 0.1), 
		DecisionTreeRegressor(max_depth = 4), 
		RandomForestRegressor(max_depth = 4)]
    
    def get_data_unit_test(data):
        X = np.array(data.drop(["UnitTest(sec)", "IntegrationTest(sec)", "UnitIntegrationTest(sec)"], axis = 1))
        y = np.array(data["UnitTest(sec)"])
        return X, y

    def get_data_integration_test(data):
        X = np.array(data.drop(["UnitTest(sec)", "IntegrationTest(sec)", "UnitIntegrationTest(sec)"], axis = 1))
        y = np.array(data["IntegrationTest(sec)"])
        return X, y

    def get_data_unit_integration_test(data):
        X = np.array(data.drop(["UnitTest(sec)", "IntegrationTest(sec)", "UnitIntegrationTest(sec)"], axis = 1))
        y = np.array(data["UnitIntegrationTest(sec)"])
        return X, y
    
    def models(self, X, y):
        MAE = []
        for model in  self.MODELS:
            scores = -cross_val_score(model, X, y, scoring = "neg_mean_absolute_error", cv = 10)
            mae = {model.__class__.__name__ : scores.mean()}
            MAE.append(mae)
        return MAE