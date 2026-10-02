# project 7: Improving DDPM

## Denoising diffusion probabilistic models (DDPM)

In this task, you will explore the impact of noise scheduling on diffusion models.

Hypothesis to verify:

> A cosine schedule allows a denoising diffusion probabilistic model to generate high-quality samples using fewer time steps than a linear schedule.

You will train two separate models to test this hypothesis. You will submit your training code, a 2-page report, and a reproducibility package containing an inference notebook that loads your pre-trained models to generate examples for all 10 digits.

### Task details

Work through the following tasks:

#### Dataset setup

* Dataset: [mnist_custom.pt](/mnist_custom.pt)

#### Custom MNIST dataset

For this challenge, you will not use the standard torchvision MNIST dataset. Instead, you are provided with a custom dataset file named mnist_custom.pt. You can load the dataset using PyTorch as follows:

```python
import torch 
data = torch.load('mnist_custom.pt') 
```

The dataset dictionary contains four keys that provide the necessary PyTorch tensors for the training images and their corresponding digit labels, as well as the separate testing images and labels.

#### Task 1: Model training

Train 2 separate diffusion models in the DDPM_CosineSchedule.ipynb notebook on the mnist_custom.pt dataset:

* One model using the linear_beta_schedule that we defined in the 9.3 Tutorial.
* One model using your own implementation of a cosine_beta_schedule.

#### Task 2: Comparison of the models (inference)

Create a separate inference: main_report.ipynb notebook that loads your two models from their saved checkpoints. Run the inference for both the Linear Schedule and the Cosine Schedule to visually compare the generation quality. Perform conditional sampling to generate images for all 10 digits (Classes 0–9). You must generate 12 samples per digit (a total of 120 images per model) to demonstrate consistency and variety.

image.png

Example of a valid output: digits are different (no mode collapse), consistent with the condition, and of good quality.

#### Report

Submit a professional report in PDF format. Your report should be clear, concise, and technically precise. It must include the following sections:

* Introduction: Explain the hypothesis and why changing to a cosine schedule might affect reconstruction quality. Include bibliographic references that attempted to prove this hypothesis and what they found.
* Methodology: Summarise the baseline training. Describe the modifications made to incorporate the cosine schedule and any corresponding changes to the training process.
* Experiments: Describe what experiments you did to investigate that hypothesis.
* Discussion: Describe the results of your experiments, and analyse whether the results support the hypothesis. Discuss the impact of the cosine schedule on both training time and image quality.

Conclusion Summarise your main findings in 2–3 sentences. Optionally, suggest further experiments or improvements.

#### Project Code

You will submit a zipped folder named project7_code.zip on this page. This folder must include:

* **DDPM_CosineSchedule.ipynb**: A Jupyter Notebook containing the full implementation of the proposed hypothesis model: a conditional diffusion model trained with a cosine beta schedule. Follow the structure of the original 9.3 Tutorial code as closely as possible.
* **main_report.ipynb**: An inference notebook that loads your two saved checkpoints and reproduces all the plots, numerical, and qualitative results presented in your report so the grader can verify reproducibility, including the 12×10 grid of generated digits (12 samples per digit, 120 images) for both schedules. Include any auxiliary files required for this process, such as model checkpoints, pickle files with losses, or other necessary data. Ensure this notebook with these auxiliary files runs in the IFN680 computing environment. This notebook will be run for grading; failure on any of the cells will result in a 0 mark for the code evaluation.
