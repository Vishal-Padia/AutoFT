from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils.logging_utils import logger


def load_model_and_tokenizer(model_name, hf_token=None):
    logger.info(f"Loading model and tokenizer: {model_name}")
    if hf_token is not None:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, token=hf_token
        )

        return model, tokenizer


def preprocess_data(dataset, tokenizer, input_column):
    logger.info("Preprocessing data...")

    def tokenize_function(examples):
        tokenized = tokenizer(
            dataset[input_column], padding=True, truncation=True, max_length=512
        )
        tokenized["labels"] = tokenized["labels"].copy()
        return tokenized

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    logger.info("Data preprocessed successfully!")
    return tokenized_dataset
