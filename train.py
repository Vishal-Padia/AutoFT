import wandb
import torch
import random

import numpy as np

from transformers import (
    EarlyStoppingCallback,
    TrainingArguments,
    Trainer,
)
from utils.logging_utils import logger
from utils.model_utils import setup_padding_token
from utils.data_utils import load_dataset_from_hf, load_local_dataset
from tasks import text_generation, summarization, sequence_classification


def finetune_model(
    dataset_name,
    config_name,
    model_name,
    hf_token,
    wandb_api_key,
    wandb_project_name,
    task_type,
    num_epochs,
    batch_size,
    learning_rate,
    optimizer,
    seed,
    input_column,
    output_column,
    file_upload,
    file_type,
    wandb_run_name,
):
    """
    This function will fine-tune a pre-trained model on a given dataset based on the parameters provided.
    """
    # Set the seed for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Initialize wandb
    logger.info("Initializing wandb...")
    wandb.login(key=wandb_api_key)
    wandb.init(
        project=wandb_project_name,
        config={
            "dataset_name": dataset_name,
            "config_name": config_name,
            "model_name": model_name,
            "task_type": task_type,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "optimizer": optimizer,
            "seed": seed,
        },
    )

    # Load the dataset
    try:
        logger.info(f"Loading dataset: {dataset_name}...")
        if file_upload is not None:
            dataset = load_local_dataset(file_upload, file_type)
        else:
            dataset = load_dataset_from_hf(dataset_name, config_name)
        logger.info("Dataset loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        raise ValueError(f"Failed to load dataset: {str(e)}")

    # Load the model and tokenizer based on the task
    try:
        logger.info(f"Loading model and tokenizer for task: {task_type}...")
        if task_type == "Text Generation":
            model, tokenizer = text_generation.load_model_and_tokenizer(
                model_name, hf_token
            )
            # resizing the model
            model.resize_token_embeddings(len(tokenizer))

            # Add padding token to the tokenizer
            setup_padding_token(tokenizer)

            # Log vocabulary sizes before resizing
            logger.info(f"Original tokenizer vocab size: {len(tokenizer)}")
            logger.info(f"Original model vocab size: {model.config.vocab_size}")

            # Now resize the model embeddings to match the new tokenizer size
            model.resize_token_embeddings(len(tokenizer))

            # Verify sizes match
            logger.info(f"New tokenizer vocab size: {len(tokenizer)}")
            logger.info(f"New model vocab size: {model.config.vocab_size}")

            # Add a verification check
            if len(tokenizer) != model.config.vocab_size:
                raise ValueError(
                    f"Vocabulary size mismatch: Tokenizer={len(tokenizer)}, Model={model.config.vocab_size}"
                )

            # Preprocess the data for training
            tokenized_dataset = text_generation.preprocess_data(
                dataset, tokenizer, input_column
            )

            # Add debug check for token IDs
            sample_batch = next(iter(tokenized_dataset["train"]))
            max_token_id = max(sample_batch["input_ids"])
            if max_token_id >= len(tokenizer):
                raise ValueError(
                    f"Token ID {max_token_id} out of range (vocab size: {len(tokenizer)})"
                )

        elif task_type == "Summarization":
            model, tokenizer = summarization.load_model_and_tokenizer(
                model_name, hf_token
            )
            # resizing the model
            model.resize_token_embeddings(len(tokenizer))

            # Preprocess the data for training
            tokenized_dataset = summarization.preprocess_data(
                dataset, tokenizer, input_column, output_column
            )
        elif task_type == "Sequence Classification":
            model, tokenizer = sequence_classification.load_model_and_tokenizer(
                model_name, hf_token
            )
            # resizing the model
            model.resize_token_embeddings(len(tokenizer))

            # Preprocess the data for training
            tokenized_dataset = sequence_classification.preprocess_data(
                dataset, tokenizer, input_column
            )
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
        logger.info("Model and tokenizer loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise ValueError(f"Failed to load model: {str(e)}")

    # Prepare the dataset for training
    train_dataset = tokenized_dataset["train"]
    eval_dataset = (
        tokenized_dataset["validation"] if "validation" in tokenized_dataset else None
    )

    # Define the training arguments
    logger.info("Setting up training arguments...")
    training_args = TrainingArguments(
        run_name=wandb_run_name,
        output_dir="./results",
        eval_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to="wandb",
        fp16=True,
    )

    # Get the optimizer
    if optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Define the trainer
    logger.info("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        optimizers=(optimizer, None),
        compute_metrics=lambda eval_pred: {
            "accuracy": (eval_pred.label_ids == eval_pred.predictions.argmax(-1)).mean()
        },
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Train the model
    logger.info("Training started...")
    trainer.train()
    logger.info("Training completed!")

    # Evaluate the model
    logger.info("Evaluating model...")
    eval_results = trainer.evaluate()
    logger.info(f"Eval Results: {eval_results}")

    # Save the model
    logger.info("Saving model...")
    model.save_pretrained("./finetuned_model")
    tokenizer.save_pretrained("./finetuned_model")
    logger.info("Model saved successfully!")

    # Log the results to wandb
    wandb.log(eval_results)
    wandb.finish()

    return "Training completed successfully!"
