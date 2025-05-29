# Simple Plagiarism Checker

This is a simple plagiarism detection system that uses machine learning to identify potential plagiarism between two texts.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Download the MIT plagiarism detection dataset and save it as 'plagiarism_dataset.csv' in the project directory.

## Usage

Run the main script:
```bash
python plagiarism_checker.py
```

The script will:
1. Load and preprocess the dataset
2. Train a machine learning model
3. Show the model's accuracy and classification report
4. Run an example plagiarism check

## Features

- Text preprocessing (lowercase conversion, special character removal, stopword removal)
- TF-IDF vectorization
- Logistic Regression model for classification
- Accuracy metrics and classification report
- Simple API for checking plagiarism between two texts

## Note

This is a simple implementation for educational purposes. The model's accuracy depends on the quality and size of the training dataset. 