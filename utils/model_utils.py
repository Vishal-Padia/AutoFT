from utils.logging_utils import logger


def setup_padding_token(tokenizer):
    """
    Adds a padding token to the tokenizer if it doesn't exist.
    """
    if tokenizer.pad_token is None:
        logger.info("Padding token not found. Adding padding token...")

        # Add a new padding token
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        logger.info("Padding token added successfully!")

        # verify that padding token is set
        logger.info("Checking if padding token is set...")
        if tokenizer.pad_token is None:
            raise ValueError("Failed to set padding token!")

    return tokenizer
