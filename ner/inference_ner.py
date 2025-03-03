# Import necessary libraries
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Define the path to the fine-tuned NER model.
model_path = "./ner_model"

# Load the tokenizer and model from the saved directory.
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)

# Initialize a NER pipeline using the loaded model.
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

#Extracts animal entities from the input text using the NER pipeline.
def extract_animal(text):
    results = ner_pipeline(text) # Apply NER model to extract named entities.
    detected_animal = [] # List to store detected animal entity tokens.

    # Iterate through the detected entities from the NER pipeline.
    for res in results:
        entity_label = res['entity'] #Extract entity label ('B-animal', 'I-animal').
        entity_text = res['word'] #Extract the word corresponding to the entity.  
        
        #Check if the detected entity is labeled as an animal
        if entity_label in ["B-animal", "I-animal"]:
            detected_animal.append(entity_text) # Store detected animal tokens.

    # Merge consecutive tokens into a single animal entity name.
    if detected_animal:
        return " ".join(detected_animal).lower()
    
    return None

# Run the script only if executed directly.
if __name__ == "__main__":
     # Ask the user to enter a sentence containing an animal entity.
    user_text = input("Enter a sentence describing an animal: ")

    # Extract animal entity from the user-provided sentence.
    detected_animal = extract_animal(user_text)

    # Display the extracted animal entity or notify if none was found.
    if detected_animal:
        print("Extracted animal:", detected_animal)
    else:
        print("No animal detected in the text.")
