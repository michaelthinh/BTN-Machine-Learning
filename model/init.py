from model.lstm_model import LSTMModelBuilder
from model.rnn_model import RNNModelBuilder
from model.xgboost_model import XGBModelBuilder
from model.base import Model, ModelFileService
from constant import coins, features, models
from itertools import combinations
import os
from trading_data import getAllDataToCSV
from model.factory import ModelBuilderFactory


class ModelsInitializer:
    def __init__(self, features=features, coins=coins, models=models):
        self.features = features
        self.coins = coins
        self.models = models
        self.modelFileDirectory = ModelFileService.getModelFileDirectory()

    def getFeaturesCombination(self):
        results = []
        for num_of_feature in range(1, len(self.features) + 1):
            for feature_combination in combinations(self.features, num_of_feature):
                results.append(list(feature_combination))
        return results

    def clearOldModelFiles(self):
        path = self.modelFileDirectory
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            os.remove(file_path)

    def downloadTrainDataFiles(self):
        getAllDataToCSV()

    def buildModels(self):
        for coin in self.coins:
            for features_combination in self.getFeaturesCombination():
                for model in self.models:
                    ModelBuilderFactory.getModelBuilder(
                        model, features_combination, coin
                    ).buildModel()

    def init(self):
        self.downloadTrainDataFiles()
        self.clearOldModelFiles()
        self.buildModels()
