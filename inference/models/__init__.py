class Model:
    SUPPORTED_TASKS = ['classify', 'detect', 'estimate', 'segment', 'track']

    def __init__(self, model_name: str, task: str):
        assert task in self.SUPPORTED_TASKS, f'Unsupported task: {task}'

        self.model_name = model_name
        self.task = task

        if task == "classify":
            from .models import Classifier
            self.model = Classifier(model_name)
        elif task == "detect":
            from .models import Detector
            self.model = Detector(model_name)
        elif task == "estimate":
            from .models import Estimator
            self.model = Estimator(model_name)
        elif task == "segment":
            from .models import Segmentor
            self.model = Segmentor(model_name)
        elif task == "track":
            from .models import Tracker
            self.model = Tracker(model_name)
        else:
            raise NotImplementedError(f'Unsupported task: {task}')

    def __call__(self, image, **kwargs):
        return self.model(image, **kwargs)
