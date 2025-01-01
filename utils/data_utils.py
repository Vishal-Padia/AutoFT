from datasets import load_dataset, Dataset
from utils.logging_utils import logger


def load_dataset_from_hf(dataset_name: str, config_name: str = None):
    """
    Loads a dataset from huggingface using the name and configuration provided.
    """
    logger.info(f"Loading dataset: {dataset_name} with config: {config_name}")
    if config_name:
        dataset_name = str(dataset_name)
        config_name = str(config_name)
        return load_dataset(dataset_name, config_name)
    return load_dataset(str(dataset_name))


def load_local_dataset(file_upload, file_type):
    """
    Loads a dataset from a local file based on the file type provided.
    """
    logger.info(f"Loading dataset from file: {file_upload.name} with type: {file_type}")
    if file_type == "csv":
        return Dataset.from_csv(file_upload)
    elif file_type == "json":
        return Dataset.from_json(file_upload)
    else:
        raise ValueError(
            "Invalid file type provided. Please provide a CSV or JSON file."
        )
