# Fine-tuning OpenAI's CLIP model Using Indian Fashion Dataset

<p>
This repository contains code for fine-tuning OpenAI's Contrastive Language–Image Pretraining (CLIP) model on a custom dataset. In this example, we use the Indian Fashion Apparel Dataset available on <a href="https://www.kaggle.com/datasets/validmodel/indo-fashion-dataset">Kaggle</a>.
</p>


## Indofashion Dataset
<p>The dataset consists of 106K images and 15 unique cloth categories. There's an equal distribution of the classes in the validation and the test set consisting of 500 samples per class for fine-grained classification of Indian ethnic clothes.
</p>


## Overview
<p>
The CLIP model is designed to understand images in context with natural language. By training the model on a large number of images and their associated texts, CLIP learns to generate meaningful embeddings for both images and texts that are aligned in semantic space.
</p>
<p>
The code in this repository demonstrates how to fine-tune the CLIP model on a Indofashion dataset. This is done to adapt the pre-trained CLIP model to better understand specific domains or types of data that may not be well-covered in its original training set.
</p>


## Prerequisites
Before you begin, ensure you have met the following requirements:

Python 3.6 or later
PyTorch 1.7.1 or later
transformers and clip libraries installed
Access to a GPU is recommended but not required

## Dataset
The dataset used in this example is the Indian Fashion Apparel Dataset, available on Kaggle. Modify the **image_path** and **json_path** with your own and you're good to go. 

Running the Code
The main script for training the model is **fine_tune_clip.py**. This script loads the CLIP model, sets up a DataLoader for the dataset, and fine-tunes the model using backpropagation and gradient descent.

To run the script, navigate to the repository directory and enter:
```
python indofashion_clip.py
```

The script will run for a number of epochs specified by the **num_epochs** variable. The batch size and learning rate can be modified by changing the **batch_size** and **lr parameters** in the DataLoader and optimizer setup respectively.


## Refrences
1. https://github.com/openai/CLIP
2. https://github.com/openai/CLIP/issues/83




