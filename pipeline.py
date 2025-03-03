import argparse
import os
from ner.inference_ner import extract_animal  # Imports the function from your NER inference file.
from image_classifier.inference_image import predict_animal  # Imports the function from your image classifier inference file.

def animal_pipeline(text: str, image_path: str) -> bool:

    # Extract animal entity from text using the NER model.
    extracted_animal = extract_animal(text)
    if extracted_animal is None:
        print("No animal detected in the provided text.")
        return False

    # Check if the image file exists.
    if not os.path.exists(image_path):
        print(f"Error: The image file '{image_path}' does not exist.")
        return False

    # Predict the animal present in the image.
    predicted_animal, confidence = predict_animal(image_path)
    
    # Logging extracted and predicted results.
    print(f"Extracted animal from text: '{extracted_animal}'")
    print(f"Predicted animal from image: '{predicted_animal}' (confidence: {confidence:.2f})")

    # Compare the extracted animal and predicted animal (case insensitive).
    #if extracted_animal.lower() == predicted_animal.lower():
    if predicted_animal.lower() in extracted_animal.lower().split():
        print("Match confirmed: The animal mentioned in the text matches the image.")
        return True
    else:
        print("Mismatch: The animal mentioned in the text does not match the image.")
        return False

def main():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(
        description="Pipeline: Verifies if the animal mentioned in a text matches the animal detected in an image."
    )
    parser.add_argument("--text", type=str, default="Elephant", help="Text description containing the animal entity.")
    parser.add_argument("--image", type=str, default="./demo_images/eleph.jpeg", help="Path to the image file.")
    
    args = parser.parse_args()

    # Run the pipeline and print the final boolean result.
    result = animal_pipeline(args.text, args.image)
    print("Final Boolean Output:", result)

if __name__ == "__main__":
    main()