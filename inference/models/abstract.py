class AbstractModel:
    SUPPORTED_MODELS = []

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.model = None
        self.transform = None

        if self.model_name not in self.get_supported_models():
            raise ValueError(f"Model '{self.model_name}' is not supported")

        self._load_model(**kwargs)

    def _load_model(self, **kwargs):
        raise NotImplementedError("The method '_load_model' is not implemented")

    def inference(self, image, **kwargs):
        raise NotImplementedError("The method 'inference' is not implemented")

    def get_supported_models(self):
        return self.SUPPORTED_MODELS

    def __call__(self, image, **kwargs):
        return self.inference(image, **kwargs)
