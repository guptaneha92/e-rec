# e-rec

<h1 align="center">e-rec - Recommendation system for Amazon products using ABO dataset</h1>

<div align="center">
 
![Python version](https://img.shields.io/badge/python-3.9+-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

## Table of Contents

1. [Introduction](#introduction)
1. [Directory Structure](#directory-structure)
1. [Installation](#installation)
1. [Contributors](#contributors)
1. [References](#references)

## Introduction
e-red is a product that recommends most similar ASINs for an Amazon product using metadata, text, images and 3D-models from [ABO dataset](https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/index.html).


## Directory Structure

```
.
├── data
├── research
├── notebooks
├── erec
    ├── metadata_pipeline
    ├── text_pipeline
    ├── image_pipeline
└── app
```

### Data

Amazon Berkeley Objects (ABO) is a collection of 147,702 product listings with multilingual metadata and 398,212 unique catalog images. 8,222 listings come with turntable photography (also referred as "spin" or "360º-View" images), as sequences of 24 or 72 images, for a total of 586,584 images in 8,209 unique sequences. For 7,953 products, the collection also provides high-quality 3d models, as glTF 2.0 files.

### Research

The ABO dataset has metadata, text, images and 3-D models of Amazon products which needs extensive research and readings on appropriate ML algorithms to identify influential features and subsequently generating product recommendations.

### erec

This is a sub-repo that contains product recommendation engine implementation files.

### App

The streamlit-based app to visualize product recommendation and its efficacy.

## Installation

Create a new virtual environment using conda and install necessary packages using `requirements.txt`.

```python
conda create -n erec python=3.9
conda activate erec
pip install -f requirements.txt
```

To generate the final processed data in `parquet` format run the following set of commands, resulting file would be stored in `data/final_df_parquet`.

```python
cd data_preprocessing
python preprocessing.py
```

## Contributors

* [Neha Gupta](https://github.com/guptaneha92)
* [Vaibhav Mathur](https://github.com/vaibhavkmathur)
* [Rishav Anand](https://github.com/17rishav)
* [Esha Chouthe]()
* [Snigdha Rudola](https://github.com/SnigdhaR)

## References

### Literature

