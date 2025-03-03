# Import necessary libraries.
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Define a dictionary mapping numerical labels to named entity categories.
id2label = {
    0: "O",
    1: "B-abstract",
    2: "I-abstract",
    3: "B-animal",
    4: "I-animal",
    5: "B-event",
    6: "I-event",
    7: "B-object",
    8: "I-object",
    9: "B-organization",
    10: "I-organization",
    11: "B-person",
    12: "I-person",
    13: "B-place",
    14: "I-place",
    15: "B-plant",
    16: "I-plant",
    17: "B-quantity",
    18: "I-quantity",
    19: "B-substance",
    20: "I-substance",
    21: "B-time",
    22: "I-time"
}
# Create a reverse mapping from entity names to numerical labels.
label2id = {v: k for k, v in id2label.items()}

# Tokenizes input text and aligns tokenized output with NER labels.
def tokenize_and_align_labels(examples, tokenizer):

    # Tokenize the list of words.
    tokenized_inputs = tokenizer(
        examples["words"],
        is_split_into_words=True, #Ensure tokens correspond to words
        truncation=True,
        padding="max_length" #Pad all inputs to the same length
    )
    
    labels = []
    # Iterate over each input example to align NER labels with tokens.
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i) # Map tokens to word indices
        previous_word_idx = None # Track previous word index
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100) # Assign -100 to special tokens (ignored by the loss function)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx]) # Assign NER label to the first subword
            else:
                label_ids.append(-100) # Ignore subsequent subwords
            previous_word_idx = word_idx # Update previous word index
        labels.append(label_ids) # Store aligned labels
    
    tokenized_inputs["labels"] = labels # Add labels to tokenized output
    return tokenized_inputs

#Main function to train a Named Entity Recognition (NER) model using a pre-trained transformer.
def main():
    model_name = "google-bert/bert-base-uncased" # Pre-trained transformer model for token classification.
    
    # Load the tokenizer and model.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(id2label), # Define the number of output labels
        id2label=id2label,        # Assign label mappings for model output
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
# Move the model to MPS.
    model.to("mps")
    
    # Load the Studeni/GUM-NER-conll dataset from Hugging Face.
    dataset = load_dataset("Studeni/GUM-NER-conll")
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

      # Tokenize and align labels for both training and evaluation datasets.
    tokenized_train_dataset = train_dataset.map(
        lambda examples: tokenize_and_align_labels(examples, tokenizer),
        batched=True
    )
    tokenized_eval_dataset = eval_dataset.map(
        lambda examples: tokenize_and_align_labels(examples, tokenizer),
        batched=True
    )
    # Define training parameters for the Trainer API.
    training_args = TrainingArguments(
        output_dir="./ner_model",
        num_train_epochs=7,            # Adjust number of epochs as needed.
        per_device_train_batch_size=8, # Adjust number of batch size as needed.
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",  # Evaluate at the end of each epoch
        save_strategy="epoch",
        logging_steps=10,
        logging_dir="./logs",
        optim="adamw_torch",       # Use the efficient AdamW optimizer
        lr_scheduler_type="cosine" 
    )
    
    # Initialize the Trainer for training and evaluation.
    trainer = Trainer(
        model=model,
        args=training_args,
        # train_dataset=tokenized_dataset,
        train_dataset=tokenized_train_dataset,#my
        eval_dataset=tokenized_eval_dataset
        )
    
    # Start training the model.
    trainer.train()
    
    # Save the fine-tuned model and tokenizer.
    model.save_pretrained("./ner_model")
    tokenizer.save_pretrained("./ner_model")

# Execute the main function when the script runs.
if __name__ == "__main__":
    main()
