from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from utils.logging_utils import logger
from utils.model_utils import setup_padding_token


def load_model_and_tokenizer(model_name, hf_token=None):
    """
    Loads the model and tokenizer based on the model name provided.
    """
    logger.info(f"Loading model and tokenizer: {model_name}")
    if hf_token is not None:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)

        return model, tokenizer


def preprocess_data(dataset, tokenizer, input_column):
    """
    Preprocesses the data for training based on the input column and tokenizer provided.
    """
    logger.info("Preprocessing data...")

    def tokenize_function(examples):
        tokenized = tokenizer(
            examples[input_column],
            padding="max_length",  # Pad to the maximum length
            truncation=True,  # Truncate to the maximum length
            max_length=512,  # set the maximum length to 512
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    logger.info("Data preprocessed successfully!")
    return tokenized_dataset
