import torch

from .abstract import AbstractModel


class Classifier(AbstractModel):
    SUPPORTED_MODELS = ["tiny_vit_21m_224"]

    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)

    def _load_model(self, **kwargs):
        if self.model_name == "tiny_vit_21m_224":
            from .classify.TinyViT.config import get_config
            from .classify.TinyViT.data import build_transform, imagenet_classnames
            from .classify.TinyViT.models.tiny_vit import tiny_vit_21m_224

            self.model = tiny_vit_21m_224(pretrained=True)
            self.model.eval()

            self.transform = build_transform(is_train=False, config=get_config())
            self.classes = imagenet_classnames
        else:
            raise NotImplementedError(f"{self.model_name} not implemented")

    def inference(self, image, **kwargs):
        if self.model_name == "tiny_vit_21m_224":
            # (1, 3, img_size, img_size)
            batch = self.transform(image)[None]

            with torch.no_grad():
                logits = self.model(batch)

            # print top-5 classification names
            probs = torch.softmax(logits, -1)
            scores, inds = probs.topk(5, largest=True, sorted=True)
            class_names = [self.classes[ind] for ind in inds[0].numpy()]
            print('=' * 30)
            for score, class_name in zip(scores[0].numpy(), class_names):
                print(f'{class_name}: {score:.2f}')
        else:
            raise NotImplementedError(f"{self.model_name} not implemented")

        return class_names, scores[0].numpy()


class Detector(AbstractModel):
    SUPPORTED_MODELS = []

    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)

    def _load_model(self, **kwargs):
        pass

    def inference(self, image, **kwargs):
        pass


class Estimator(AbstractModel):
    SUPPORTED_MODELS = []

    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)

    def _load_model(self, **kwargs):
        pass

    def inference(self, image, **kwargs):
        pass


class Segmentor(AbstractModel):
    SUPPORTED_MODELS = []

    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)

    def _load_model(self, **kwargs):
        pass

    def inference(self, image, **kwargs):
        pass


class Generator(AbstractModel):
    SUPPORTED_MODELS = []

    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)

    def _load_model(self, **kwargs):
        pass

    def inference(self, image, **kwargs):
        pass


class Tracker(AbstractModel):
    SUPPORTED_MODELS = []

    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)

    def _load_model(self, **kwargs):
        pass

    def inference(self, image, **kwargs):
        pass
