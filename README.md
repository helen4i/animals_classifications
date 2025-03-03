# Animal Classification and Named Entity Recognition (NER) Pipeline

## Overview
This project implements a complete pipeline that verifies whether an animal mentioned in a given text corresponds to the animal detected in an image. The solution is built using a Named Entity Recognition (NER) model for extracting animal names from text and an image classification model for identifying animals in images.

## Project Structure
```
├── ner/
│   ├── train_ner.py              # Training script for the NER model
│   ├── inference_ner.py          # Inference script for extracting animal names from text
│
├── image_classifier/
│   ├── train_image.py            # Training script for the image classification model
│   ├── inference_image.py        # Inference script for predicting animals in images
│
├── pipeline/
│   ├── pipeline.py               # Main pipeline script that integrates text and image processing
│
├── demo_images/                  # Folder containing demo images
│   ├── Tiger-1.jpg               # Demo image 1
│   ├── eleph.jpeg                # Demo image 2
│
├── ner_model/                    # Folder containing trained NER model files (to be downloaded)
├── animal_classifier_model.keras # Trained image classification model (to be downloaded)
│
├── requirements.txt              # Required dependencies
├── demo_animal.ipynb             # Jupyter Notebook demonstrating the pipeline
├── README.md                     # Main project documentation
```

## Features
- **NER Model**: Uses a transformer-based model to extract animal entities from text.
- **Image Classification Model**: Trained on a dataset with 15 animal classes from Kaggle.
- **Pipeline Integration**: Combines the NER and image classifier models to verify textual and visual consistency.

## Installation
To set up the project, follow these steps:
1. Clone the Repository:
```bash
    git clone https://github.com/helen4i/animals_classifications.git
    cd animals_classifications
```
2. Create a Virtual Environment (optional but recommended):
```bash
    python -m venv venv
    source venv/bin/activate 
```
3. Install dependencies:
```bash
    pip install -r requirements.txt
```

## Pre-Trained Models
If you want to use my already trained models instead of training from scratch, please follow these steps:
1. **Download the trained image classification model** from the following link: [animal_classifier_model.keras](https://github.com/helen4i/animals_classifications/releases/download/v1.0/animal_classifier_model.keras).
   - Place the downloaded `animal_classifier_model.keras` file in the root folder.
2. **Download the trained NER model folder** from this link: [ner_model](https://github.com/helen4i/animals_classifications/releases/download/v1.0/ner_model.zip).
   - Extract and place the entire `ner_model` folder in the root directory.

At the end, your project directory should contain:
- `animal_classifier_model.keras` in the root folder.
- `ner_model/` folder in the root, containing all necessary files for the NER model.

## Dataset
The datasets used in this project are:
- **NER Dataset**: `Studeni/GUM-NER-conll` from Hugging Face, used for Named Entity Recognition (NER).
- **Image Classification Dataset**: [Animal Image Classification Dataset](https://www.kaggle.com/datasets/utkarshsaxenadn/animal-image-classification-dataset) from Kaggle, containing 15 animal classes.
- **Test Images**: Two test images stored in the `demo_images/` directory.

## Usage

### Training the NER Model
```bash
python ner/train_ner.py
```
This script trains the transformer-based NER model for animal name extraction.

### Training the Image Classification Model
```bash
python image_classifier/train_image.py
```
This script trains the MobileNetV2-based image classifier on the dataset.

## Running Inference

### Extracting Animal Names from Text
```bash
python ner/inference_ner.py --text "There is a tiger in the picture."
```

### Predicting Animal in an Image
```bash
python image_classifier/inference_image.py --image "./demo_images/eleph.jpg"
```

### Running the Full Pipeline
```bash
python pipeline/pipeline.py --text "There is a cow in the picture." --image "./demo_images/Tiger-1.jpg"
```
This script checks if the extracted animal from the text matches the predicted animal in the image.

## Demo
The `demo_animal.ipynb` notebook provides a demonstration of the entire pipeline, including:
- Exploratory data analysis of the dataset.
- Step-by-step execution of the NER and image classification models.
- Edge case testing to ensure robustness.

## Edge Cases Considered
- **Tokenization Issues with Articles**: The NER model initially misclassified articles ("a", "the") as entity beginnings, leading to incorrect extractions such as "a cat" instead of just "cat". To mitigate this, I merged "B-entity" and "I-entity" to ensure entity names are grouped correctly.
- **Mismatch Between NER and Image Classification Outputs**: When running the full pipeline, discrepancies arose when NER extracted "a cat" or "the dog", while the image classifier recognized "cat" or "dog". This resulted in false negatives. To resolve this, I implemented a matching approach where at least one word from the NER output must match the image classification output.