from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from utils.logging_utils import logger


def load_model_and_tokenizer(model_name, hf_token=None):
    """
    Loads the model and tokenizer based on the model name provided.
    """
    logger.info(f"Loading model and tokenizer: {model_name}")
    if hf_token is not None:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=hf_token)

        return model, tokenizer


def preprocess_data(dataset, tokenizer, input_column, output_column):
    """
    Preprocesses the data for training based on the input column and tokenizer provided.
    """
    logger.info("Preprocessing data...")

    def tokenize_function(examples):
        inputs = tokenizer(
            examples[input_column],
            padding="max_length",
            truncation=True,
            max_length=512,
        )
        targets = tokenizer(
            examples[output_column],
            padding="max_length",
            truncation=True,
            max_length=512,
        )
        inputs["labels"] = targets["input_ids"]
        return inputs

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    logger.info("Data preprocessed successfully!")
    return tokenized_dataset
