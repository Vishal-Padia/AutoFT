from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
)
from utils.logging_utils import logger


def load_model_and_tokenizer(model_name, hf_token=None):
    logger.info(f"Loading model and tokenizer: {model_name}")
    if hf_token is not None:
        try:
            # Load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
            model = AutoModelForMaskedLM.from_pretrained(
                model_name, token=hf_token, num_labels=2
            )
            return model, tokenizer

        except Exception as e:
            # Log error and raise exception
            logger.error(f"Failed to load model and tokenizer: {e}")
            raise ValueError(
                f"The model '{model_name}' is not compatible with Seq2Seq tasks. "
                "Please use a model designed for Seq2Seq tasks (e.g., T5, BART, Pegasus)."
            )
    else:
        logger.info(f"Please provide a valid Hugging Face token to load the model.")
        raise ValueError("Please provide a valid Hugging Face token to load the model.")


def preprocess_data(dataset, tokenizer, input_column):
    logger.info("Preprocessing data...")

    # def tokenize_function(examples):
    #     tokenized = tokenizer(
    #         examples[input_column],  # input_column
    #         padding="max_length",  # Pad to the maximum length
    #         truncation=True,  # Truncate to the maximum length
    #         max_length=512,  # set the maximum length to 512
    #     )
    #     tokenized["labels"] = tokenized["input_ids"].copy()
    #     return tokenized

    def preprocess_function(examples):
        return tokenizer(
            examples[input_column],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

    # Use smaller batches during preprocessing to avoid memory issues
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        batch_size=2,
    )

    # set the format to PyTorch
    tokenized_dataset.set_format("torch")

    logger.info("Data preprocessed successfully!")
    return tokenized_dataset
