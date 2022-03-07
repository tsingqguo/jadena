from typing import Sequence

import torch
import torch.nn as nn

from pytorchcv.model_provider import get_model as ptcv_get_model


class ClassifierModel(nn.Module):
    def __init__(self, classifier_name: str, stages: Sequence[int]):
        super(ClassifierModel, self).__init__()
        if isinstance(stages, int):
            stages = [stages]
        self.classifier_name = classifier_name
        self.stages = stages
        self.max_stage = max(stages)
        self.features = ptcv_get_model(classifier_name, pretrained=True).features

        # See https://github.com/osmr/imgclsmob/blob/a5c5bf8d2f3777d16d0898e2cd6572e32de17f2c/pytorch/datasets/imagenet1k_cls_dataset.py#L66
        self.register_buffer('image_mean', torch.tensor((0.485, 0.456, 0.406)))
        self.register_buffer('image_std', torch.tensor((0.229, 0.224, 0.225)))

    def forward(self, x):
        # centerization
        x = x.clone()
        x = x.sub_(self.image_mean[:, None, None]).div_(self.image_std[:, None, None])

        feat = []
        last = x
        for stage, layer in enumerate(self.features):
            last = layer(last)
            if stage in self.stages:
                feat.append(last)
            if stage >= self.max_stage:
                break

        return feat