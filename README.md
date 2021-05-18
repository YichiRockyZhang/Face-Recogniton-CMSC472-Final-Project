# Face-Verification-CMSC472-Final-Project: [Colab Link](https://colab.research.google.com/drive/1HloPmg2NdLmXcCpRPS5qcaLezRQ7oW6X?usp=sharing)

# Getting Started: Loading the Labeled Faces in the Wild (LFW) dataset and the CelebA datadest.

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import scipy
from PIL import Image
from scipy import ndimage
from tqdm import tqdm
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from skimage.transform import resize
from skimage.color import rgb2gray

import pandas as pd
import random
import csv
```

Dataset loaded from: https://www.kaggle.com/jessicali9530/lfw-dataset

1. Download kaggle.json from Kaggle containing your (Kaggle API key)[https://www.kaggle.com/docs/api].
2. Run the code block below.
3. Upload kaggle.json when prompted.
    
```python
! pip install -q kaggle
from google.colab import files
files.upload()
! mkdir ~/.kaggle
! cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json

# To download LFW dataset
! kaggle datasets download -d jessicali9530/lfw-dataset
! mkdir LFW
! unzip lfw-dataset.zip -d LFW

# To download CelebA dataset
! kaggle datasets download -d jessicali9530/celeba-dataset
! unzip celeba-dataset.zip -d CelebA
! rm -rf celeba-dataset.zip lfw-dataset.zip
```

Try to download the cropped and resized images from CelebA from [here](https://drive.google.com/file/d/1jklUKdmdNixZ5voLOyAChrb_AtfYwLUs/view?usp=sharing) by running the cell below.

```python
! pip install gdown
! gdown https://drive.google.com/uc?id=1jklUKdmdNixZ5voLOyAChrb_AtfYwLUs
! unzip resized.zip -d CelebA/img_align_celeba/
! gdown https://drive.google.com/uc?id=1tAj_dqgm7qHQd1T0NE_rS9fp655j_z8j
! sed 's/ \+/,/g' list_identity_celeba.txt > CelebA/list_identity_celeba.csv && rm list_identity_celeba.txt
! rm -f resized.zip
! ls CelebA/img_align_celeba/img_align_celeba_resized | wc -l && echo -n " resized images from CelebA have been downloaded successfully."
```

If the download fails, use the command below to crop the images from CelebA (est. 2 hours).

```python
# ! sudo apt install imagemagick
# ! mkdir CelebA/img_align_celeba/img_align_celeba_resized
# ! for i in CelebA/img_align_celeba/img_align_celeba/*.jpg;do echo "$i" | cut -d '/' -f 4 && convert "$i" -resize 250x250 -background black -gravity center -extent 250x250 "CelebA/img_align_celeba/img_align_celeba_resized/$(echo "$i" | cut -d '/' -f 4)";done
```

Verify the resized CelebA images were successfully downloaded by running the below cell. You should see two images, one with and one without black bars.

```python
from IPython.display import Image, display
# display(Image('CelebA/img_align_celeba/img_align_celeba/028768.jpg')) # original image
# display(Image('CelebA/img_align_celeba/img_align_celeba_resized/028768.jpg')) # resized to 250x250

print('Anchor and Positive (same person)')
display(Image('CelebA/img_align_celeba/img_align_celeba_resized/008073.jpg'))
display(Image('CelebA/img_align_celeba/img_align_celeba_resized/011233.jpg'))
```
