# Jadena

Official implementation of "Can You Spot the Chameleon? Adversarially Camouflaging Images from Co-Salient Object Detection" in CVPR 2022. [arXiv](https://arxiv.org/pdf/2009.09258.pdf)

## Usage

1. Clone this repo.
2. Install the dependencies via the command "pip install -r requirements.txt" or any other way you prefer. You should pay attention to the installation of `torch' due to the difference between cpu and gpu.
3. Run the script `example.py`. It will perturb the image `example_images/turtles/turtle_1.png` and save the result perturbed images in the same folder.
