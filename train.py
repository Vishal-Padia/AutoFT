import os
import wandb
import torch
import random
import numpy as np
from datasets import load_dataset, Dataset
from transformers import (
    EarlyStoppingCallback,
    TrainingArguments,
    Trainer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from custom_logging import logger


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
            if file_type == "csv":
                dataset = Dataset.from_csv(file_upload)
            elif file_type == "json":
                dataset = Dataset.from_json(file_upload)
        else:
            if config_name:
                dataset_name = str(dataset_name)
                config_name = str(config_name)
                dataset = load_dataset(dataset_name, config_name)
            else:
                dataset = load_dataset(dataset_name)
        logger.info("Dataset loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        raise ValueError(f"Failed to load dataset: {str(e)}")

    # Load the pre-trained model and tokenizer
    try:
        logger.info(f"Loading the tokenizer for model: {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

        if tokenizer.pad_token is None:
            logger.info("Padding token not found. Adding padding token...")
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            logger.info("Padding token added successfully!")
        logger.info("Tokenizer loaded successfully!")

        logger.info(f"Loading model: {model_name}...")
        if task_type == "Text Generation":
            model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)
        elif task_type == "Summarization":
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=hf_token)
        elif task_type == "Sequence Classification":
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, token=hf_token
            )
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise ValueError(f"Failed to load model: {str(e)}")

    # Tokenize the dataset
    logger.info("Tokenizing dataset...")

    def tokenize_function(examples):
        tokenized = tokenizer(
            examples[input_column],
            padding="max_length",  # Pad to the maximum length
            truncation=True,  # Truncate to the maximum length
            max_length=512,  # Set a reasonable max length
        )
        tokenized["labels"] = tokenized["labels"].copy()
        return tokenized

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    logger.info("Dataset tokenized successfully!")

    # Prepare the dataset for training
    train_dataset = tokenized_dataset["train"]
    eval_dataset = (
        tokenized_dataset["validation"] if "validation" in tokenized_dataset else None
    )

    # Define the training arguments
    logger.info("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
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
        fp16=True,  # Enable mixed precision for GPU
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
