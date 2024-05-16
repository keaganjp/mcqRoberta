import argparse
from operator import truediv
import os
import random
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForMultipleChoice
from datasets import get_dataset_config_names, load_dataset, load_metric
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
from torch.optim import AdamW,SGD, Adam
from transformers import get_scheduler
from tqdm.notebook import tqdm
import evaluate


# Set environment variables for CUDA and tokenizers
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Function to set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Set seed value
SEED = 595
set_seed(SEED)

# Determine the device to use (GPU if available, else CPU)
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# Define a data collator class for multiple choice tasks
@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    # Function to collate and pad the input features
    def __call__(self, features):

        labels = [feature.pop('labels') for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])

        # Flatten
        flattened_features = [[{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features]
        flattened_features = sum(flattened_features, [])

        # Apply Padding
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1).to(device) for k, v in batch.items()}

        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64).to(device)

        return batch

# Function to load and preprocess the dataset
def load_data(tokenizer, params):
    """
    Takes in the pretrained tokenizer and dataset parameters. Loads data from hugging face datasets and tokenizes the inputs and returns
    torch.utils.DataLoader object. 
    """

    # Map answer choices to numerical labels
    label_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, '1' : 0, '2' : 1, '3': 2, '4' : 3}

    # Convert the labels to class labels
    def convert_labels_to_class_labels(example):
      example['answerKey'] = label_mapping[example['answerKey']]
      example['choices']['label'] = [label_mapping[label] for label in example['choices']['label']]
      return example
    
    # Tokenize the input examples
    def tokenize_function(examples):
        first_sentences = [[[question] * 4] for question in examples['question']]
        second_sentences = [[text for text in item['text']] for item in examples['choices']]
        first_sentences = sum(sum(first_sentences, []),[])
        second_sentences = sum(second_sentences, [])
        tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
        tokenized_examples = {k: [v[i:i+4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
        return tokenized_examples
    # Parse dataset name and configuration
    dataset_name_parts = params.dataset.split(',')
    base_dataset_name = dataset_name_parts[0]
    config_name = dataset_name_parts[1] if len(dataset_name_parts) > 1 else None

    # Load the dataset with the specific configuration
    if config_name:
        dataset = load_dataset(base_dataset_name, config_name)
    else:
        dataset = load_dataset(base_dataset_name)

    # Filter examples with exactly 4 choices
    dataset= dataset.filter(lambda example: len(example['choices']['label']) == 4)

    # Map the labels to class labels
    dataset = dataset.map(convert_labels_to_class_labels)

    # Tokenize the dataset
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.rename_column("answerKey", "labels")

    # Remove unnecessary columns
    accepted_keys = ["input_ids", "attention_mask", "labels"]
    for key in tokenized_datasets['train'].features.keys():
      if key not in accepted_keys:
        tokenized_datasets = tokenized_datasets.remove_columns(key)
    tokenized_datasets.set_format("torch")
    
    # Define batch size and data collator
    batch_size = params.batch_size
    data_collator = DataCollatorForMultipleChoice(tokenizer)

    # Create data loaders for training, validation, and test sets
    train_dataset = tokenized_datasets["train"].shuffle(seed=SEED)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=data_collator)
    eval_dataset = tokenized_datasets["validation"]
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=data_collator)
    test_dataset = tokenized_datasets["test"]
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=data_collator)

    return train_dataloader, eval_dataloader, test_dataloader




# Function to fine-tune the model
def finetune(model, train_dataloader, eval_dataloader, params):
    num_epochs = params.num_epochs
    num_training_steps = num_epochs * len(train_dataloader)
    
    # Freeze some layers and unfreeze others based on model type
    if model.__class__.__name__ == 'BertForMultipleChoice':
        for param in model.parameters():
            param.requires_grad = False
        for param in model.bert.encoder.layer[0:4].parameters():
            param.requires_grad = True
        for param in model.bert.encoder.layer[-2:].parameters():
            param.requires_grad = True
    elif model.__class__.__name__ == 'RobertaForMultipleChoice':
        for param in model.parameters():
            param.requires_grad = False
        for param in model.roberta.encoder.layer[0:3].parameters():
            param.requires_grad = True
        for param in model.roberta.encoder.layer[-3:].parameters():
            param.requires_grad = True

    # Define optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=params.learning_rate)
    
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    
    # Load evaluation metric
    metric = evaluate.load("accuracy")
    torch.cuda.empty_cache()
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{num_epochs}")):
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            train_loss += loss.item()
            
            if step % 10 == 0:
                print(f"Step {step} - Training Loss: {loss.item()}")

        avg_train_loss = train_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} - Average Training Loss: {avg_train_loss}")

        model.eval()
        eval_loss = 0.0
        for batch in tqdm(eval_dataloader, desc=f"Evaluating Epoch {epoch+1}/{num_epochs}"):
            with torch.no_grad():
                outputs = model(**batch)
            
            logits = outputs.logits
            loss = outputs.loss
            eval_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])

        avg_eval_loss = eval_loss / len(eval_dataloader)
        score = metric.compute()

        print(f"Epoch {epoch+1} - Validation Accuracy: {score['accuracy']} - Average Eval Loss: {avg_eval_loss}")

    return model



# Function to test the model
def test(model, test_dataloader, prediction_save='predictions.torch'):
    metric = evaluate.load('accuracy')
    model.eval()
    all_predictions = []

    for batch in test_dataloader:
      batch = {k: v.to(device) for k, v in batch.items()}
      with torch.no_grad():
        outputs = model(**batch)
      logits = outputs.logits
      predictions = torch.argmax(logits, dim=-1)
      all_predictions.extend(list(predictions))
      metric.add_batch(predictions=predictions, references=batch["labels"])
    torch.save(all_predictions, 'predictions.torch')
    score = metric.compute()
    print('Test Accuracy:', score)
    torch.save(all_predictions, prediction_save)

# Main function to load data, fine-tune and test the model
def main(params):

    tokenizer = AutoTokenizer.from_pretrained(params.model)
    train_dataloader, eval_dataloader, test_dataloader = load_data(tokenizer, params)

    model = AutoModelForMultipleChoice.from_pretrained(params.model)
    model.to(device)
    model = finetune(model, train_dataloader, eval_dataloader, params)

    test(model, test_dataloader)

# Argument parser for command line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune Language Model")

    parser.add_argument("--dataset", type=str, default="piqa")
    parser.add_argument("--model", type=str, default="bert-base-cased")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type= float, default=1e-4)

    params, unknown = parser.parse_known_args()
    main(params)