# Celebrity Face Recognition System
## Overview

A celebrity face recognition system built using ArcFace and RetinaFace, capable of detecting and recognizing celebrity faces from images. The project demonstrates end-to-end workflows from dataset preparation to model training and evaluation.

## Features

Web-scraped dataset collection and identity-based organization

Face detection and preprocessing with RetinaFace

Face recognition using ArcFace architecture

End-to-end training workflow, including dataset file generation and iterative model refinement

Performance analysis and dataset optimization for improved recognition

## Dataset

A small sample dataset is included in this repository for testing purposes.

The full dataset of celebrity images is hosted on Kaggle: Celebrity Face Recognition Dataset

Note: Please use the dataset for educational and research purposes only.

## Dataset Structure
``` /data
   /identity_1
       img1.jpg
       img2.jpg
       ...
   /identity_2
       img1.jpg
       img2.jpg
       ...
```

## Installation

Clone the repository:

``` git clone https://github.com/MarthaKJ/celebrity-face-recognition.git ```


## Install dependencies:

``` pip install -r requirements.txt ```


Set up dataset folders according to the structure above.

## Usage


Preprocess images using the provided preprocessing scripts:

``` python preprocess.py --dataset /path/to/data ```


Train the model using ArcFace:

``` python train.py --dataset /path/to/data ```


Evaluate the model on test images:

``` python evaluate.py --model /path/to/model ```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

License

MIT License



