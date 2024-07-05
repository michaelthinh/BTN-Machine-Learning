from constant import features as validFeatures, coins, windowSize
import pandas as pd
from model.train_data import TrainDataProvider
from model.utils import CoinValidator, FeatureValidator


class Model:
    def __init__(self, modelName, features, coin):
        self.modelName = modelName
        # check if features are valid
        if not FeatureValidator().areValidFeatures(features):
            raise ValueError(f"Invalid features: {features}")
        self.features = features
        self.features.sort()

        if not CoinValidator().isValidCoin(coin):
            raise ValueError(f"Invalid coin: {coin}")
        self.coin = coin


class ModelInputExtractor:
    def __init__(self, model: Model, windowSize=windowSize):
        self.model = model
        self.windowSize = windowSize

    def extractData(self, data):
        data = data[self.model.features]
        data = data[-self.windowSize :]
        return data


class ModelInputValidator:
    def __init__(self, model: Model):
        self.model = model
        # self.featureValidator = FeatureValidator(self.model.features)

    def areValidFeatures(self, features):
        # features has all self.model.features
        featureValidator = FeatureValidator(features)
        return all(
            [
                featureValidator.isValidFeature(feature)
                for feature in self.model.features
            ]
        )

    def isValidInput(self, data):
        assert isinstance(data, pd.DataFrame)
        return self.areValidFeatures(data.columns)


class WindowedModelInputValidator(ModelInputValidator):
    def __init__(self, model: Model, windowSize=windowSize):
        super().__init__(model)
        self.windowSize = windowSize

    def hasValidRows(self, data):
        return data.shape[0] >= self.windowSize

    def isValidInput(self, data):
        return super().isValidInput(data) and self.hasValidRows(data)


class ModelLoader:
    def __init__(self, model: Model):
        self.model = model

    def loadModel(self):
        raise NotImplementedError("Subclasses must implement abstract method")


class ModelPredictService:
    def __init__(self, model: Model, inputExtractor=None, inputValidator=None):
        self.model = model
        self.inputExtractor = inputExtractor or ModelInputExtractor(model)
        self.inputValidator = inputValidator or ModelInputValidator(model)
        self.scaler = TrainDataProvider(
            coin=model.coin, features=model.features, windowSize=windowSize
        ).scaler

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        print(f"Predicting with {self.model.modelName} on data: {data}")
        raise NotImplementedError("Subclasses must implement abstract method")

    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.inputValidator.isValidInput(data):
            raise ValueError("Invalid input")
        data = self.inputExtractor.extractData(data)
        data = self.scaler.scale(data)
        # x_data = data.values.reshape(1, data.shape[0], data.shape[1])
        df_prediction = self.predict(data)
        df_inverseScaled = self.scaler.inverseScale(df_prediction)
        df_inverseScaled["high"] = max(df_inverseScaled.values.reshape(-1))
        df_inverseScaled['low'] = min(df_inverseScaled.values.reshape(-1))
        return df_inverseScaled


class SavedModelPredictService(ModelPredictService):
    def __init__(
        self,
        model: Model,
        modelLoader: ModelLoader,
        inputExtractor=None,
        inputValidator=None,
    ):
        super().__init__(model, inputExtractor, inputValidator)
        self.modelLoader = modelLoader

    def predictWithLoadedModel(self, loaded_model, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Subclasses must implement abstract method")

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        loaded_model = self.modelLoader.loadModel()
        return self.predictWithLoadedModel(loaded_model, data)


class ModelFileService:
    def __init__(self, model: Model, extenstion):
        self.model = model
        self.extenstion = extenstion

    @staticmethod
    def getModelFileDirectory():
        return "./model/built_models"

    def getModelFileName(self):
        return f"./model/built_models/{self.model.modelName}_{self.model.coin}_{'_'.join(self.model.features)}.{self.extenstion}"


class KerasModelFileService(ModelFileService):
    def __init__(self, model: Model):
        super().__init__(model, "keras")


class XGBModelFileService(ModelFileService):
    def __init__(self, model: Model):
        super().__init__(model, "json")


class ModelBuilder:
    def __init__(self, model: Model, modelFileService: ModelFileService):
        self.model = model
        self.modelFileService = modelFileService

    def buildModel(self):
        raise NotImplementedError("Subclasses must implement abstract method")
