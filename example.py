#/usr/bin/env python

from pathlib import Path

import torch
import torchvision.transforms.functional as TF

from lib import BiasFieldAttack, ClassifierModel, fcon_criterion, augment
from PIL import Image


if __name__ == '__main__':
    # prepare the attack
    model = ClassifierModel('resneta50b', (1, 2, 3))
    attack = BiasFieldAttack(
        model,
        fcon_criterion,
        step=20,
        noise_mode='add',
        bias_mode='same',
        spatial_mode='optical_flow',
        noise_lr=1. / 255.,
        bias_lr=1e-1,
        spatial_lr=1e-2,
        lambda_b=1e-2,
        lambda_s=1e-2,
        momentum_decay=1.0,
        epsilon_n=16. / 255.,
        degree=10,
    )

    # prepare images
    image_folder = Path(__file__).parent / 'example_images' / 'turtles'
    images = ['turtle_1.png', 'turtle_2.jpg', 'turtle_3.jpg', 'turtle_4.jpg', 'turtle_5.jpg']
    images = [Image.open(image_folder / image) for image in images]
    images = [TF.to_tensor(image) for image in images]

    # for "group" variant
    pert, _ = attack(torch.stack(images))
    TF.to_pil_image(pert[0]).save(image_folder / 'result_group.png')

    # for "augment" variant
    pert, _ = attack(torch.stack(augment(images[0])))
    TF.to_pil_image(pert[0]).save(image_folder / 'result_augment.png')
