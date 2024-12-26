import json
import gradio as gr
from custom_logging import getLogger
from train import finetune_model

# Import the logger
logger = getLogger("AutoFT")


def main():
    # Title and description for the interface
    title = "AutoFT - FineTune LLMs Easily"
    description = "Easily configure and fine-tune your model with the parameters below."

    # Create input fields and dropdowns for various parameters
    with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
        gr.Markdown(f"<h1>{title}</h1>")
        gr.Markdown(description)

        with gr.Row():
            dataset_name = gr.Textbox(
                label="Dataset Name (Hugging Face Datasets)",
                placeholder="e.g., wikitext, cnn_dailymail, imdb",
                elem_id="dataset_name",
            )
            config_name = gr.Textbox(
                label="Config Name (optional)",
                placeholder="e.g., wikitext-103-v1 for wikitext",
                elem_id="config_name",
            )
            model_name = gr.Textbox(
                label="Model Name (Hugging Face Model Hub)",
                placeholder="e.g., gpt2, t5-small, bert-base-uncased",
                elem_id="model_name",
            )

        with gr.Row():
            wandb_api_key = gr.Textbox(
                label="Wandb API Key",
                placeholder="Enter your Wandb API Key",
                elem_id="wandb_api_key",
            )
            wandb_project_name = gr.Textbox(
                label="Wandb Project Name",
                placeholder="Enter your Wandb Project Name",
                elem_id="wandb_project_name",
            )

        task_type = gr.Dropdown(
            choices=["Text Generation", "Summarization", "Sequence Classification"],
            label="Task Type",
            value="Sequence Classification",
            elem_id="task_type",
        )

        with gr.Row():
            input_column = gr.Textbox(
                label="Input Column",
                placeholder="e.g., text, question, article",
                value="text",  # Default value
            )
            output_column = gr.Textbox(
                label="Output Column (optional)",
                placeholder="e.g., answer, summary",
            )

        with gr.Row():
            num_epochs = gr.Number(
                label="Number of Epochs", value=3, elem_id="num_epochs"
            )
            batch_size = gr.Number(label="Batch Size", value=2, elem_id="batch_size")

        with gr.Row():
            learning_rate = gr.Number(
                label="Learning Rate", value=0.00003, elem_id="learning_rate"
            )
            optimizer = gr.Dropdown(
                choices=["Adam", "SGD", "RMSprop"],
                label="Optimizer",
                value="Adam",
                elem_id="optimizer",
            )

        seed = gr.Number(label="Seed", value=42, elem_id="seed")

        with gr.Row():
            file_upload = gr.File(label="Upload Dataset (CSV/JSON)")
            file_type = gr.Dropdown(
                label="File Type", choices=["csv", "json"], value="csv"
            )

        submit_button = gr.Button("Fine-Tune Model")
        output = gr.Textbox(label="Training Logs", interactive=False)

        # Save and load configuration buttons
        save_button = gr.Button("Save Configuration")
        load_button = gr.Button("Load Configuration")

        # Process inputs when the button is clicked
        def process_inputs(
            dataset_name,
            config_name,
            model_name,
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
            try:
                logger.info("Starting fine-tuning process...")
                if file_upload is not None:
                    dataset_name = file_upload.name
                else:
                    dataset_name = dataset_name

                logs = finetune_model(
                    dataset_name,
                    config_name,
                    model_name,
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
                )
                logger.info("Fine-tuning completed successfully!")
                return logs
            except Exception as e:
                logger.error(f"Error during training: {str(e)}")
                return f"Error during training: {str(e)}"

        # Save configuration
        def save_config(
            dataset_name,
            config_name,
            model_name,
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
        ):
            try:
                config = {
                    "dataset_name": dataset_name,
                    "config_name": config_name,
                    "model_name": model_name,
                    "wandb_api_key": wandb_api_key,
                    "wandb_project_name": wandb_project_name,
                    "task_type": task_type,
                    "num_epochs": num_epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "optimizer": optimizer,
                    "seed": seed,
                    "input_column": input_column,
                    "output_column": output_column,
                }
                with open("config.json", "w") as f:
                    json.dump(config, f)
                logger.info("Configuration saved successfully!")
                return "Configuration saved!"
            except Exception as e:
                logger.error(f"Failed to save configuration: {str(e)}")
                return f"Failed to save configuration: {str(e)}"

        # Load configuration
        def load_config():
            try:
                with open("config.json", "r") as f:
                    config = json.load(f)
                logger.info("Configuration loaded successfully!")
                return [
                    config.get("dataset_name", ""),
                    config.get("config_name", ""),
                    config.get("model_name", ""),
                    config.get("wandb_api_key", ""),
                    config.get("wandb_project_name", ""),
                    config.get("task_type", "Sequence Classification"),
                    config.get("num_epochs", 3),
                    config.get("batch_size", 2),
                    config.get("learning_rate", 0.00003),
                    config.get("optimizer", "Adam"),
                    config.get("seed", 42),
                    config.get("input_column", "text"),
                    config.get("output_column", ""),
                ]
            except FileNotFoundError:
                logger.error("Configuration file not found!")
                return ["Configuration file not found!"]

        # Link the buttons to the processing functions
        submit_button.click(
            fn=process_inputs,
            inputs=[
                dataset_name,
                config_name,
                model_name,
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
            ],
            outputs=output,
        )

        save_button.click(
            fn=save_config,
            inputs=[
                dataset_name,
                config_name,
                model_name,
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
            ],
            outputs=output,
        )

        load_button.click(
            fn=load_config,
            outputs=[
                dataset_name,
                config_name,
                model_name,
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
            ],
        )

    demo.launch()


# Call the main function
if __name__ == "__main__":
    logger.info("Starting AutoFT application...")
    main()
