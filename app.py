import json
import gradio as gr
from train import finetune_model
from utils.logging_utils import logger

# Setting Jetbrains Mono as the default font for the interface
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
* {
    font-family: 'JetBrains Mono', monospace !important;
}
"""


def show_wip_message(task_type):
    """
    Displays a pop-up message if the selected task type is "Sequence Classification".
    """
    if task_type == "Sequence Classification":
        return gr.Info(
            "Sequence Classification is currently a work in progress. Please check back later!"
        )
    return None


def main():
    # Title and description for the interface
    title = "AutoFT - FineTune LLMs Easily"
    description = (
        "Welcome to AutoFT! This tool simplifies fine-tuning large language models (LLMs) for various tasks. "
        "Configure your model, dataset, and hyperparameters below, and start training with just one click!"
    )

    # Create input fields and dropdowns for various parameters
    with gr.Blocks(theme="allenai/gradio-theme", css=custom_css) as demo:
        gr.Markdown(f"# {title}")
        gr.Markdown(description)

        with gr.Tabs():
            # Tab 1: Model and Dataset Configuration
            with gr.Tab("Model & Dataset"):
                with gr.Row():
                    model_name = gr.Textbox(
                        label="Model Name (Hugging Face Model Hub)",
                        placeholder="e.g., gpt2, t5-small, bert-base-uncased",
                        info="Enter the name of the pre-trained model from Hugging Face.",
                    )
                    dataset_name = gr.Textbox(
                        label="Dataset Name (Hugging Face Datasets)",
                        placeholder="e.g., wikitext, cnn_dailymail, imdb",
                        info="Enter the name of the dataset from Hugging Face.",
                    )
                    config_name = gr.Textbox(
                        label="Config Name (optional)",
                        placeholder="e.g., wikitext-103-v1 for wikitext",
                        info="If the dataset has multiple configurations, specify one here.",
                    )

                with gr.Row():
                    task_type = gr.Dropdown(
                        label="Task Type",
                        choices=[
                            "Text Generation",
                            "Summarization",
                            "Sequence Classification",
                        ],
                        value="Text Generation",  # Default to Text Generation
                        info="Select the task you want to fine-tune the model for.",
                    )
                    hf_token = gr.Textbox(
                        label="Hugging Face Token (optional)",
                        placeholder="Enter your Hugging Face Token",
                        info="Required for private models or datasets.",
                        type="password",
                    )

            # Tab 2: Hyperparameters
            with gr.Tab("Hyperparameters"):
                with gr.Row():
                    num_epochs = gr.Number(
                        label="Number of Epochs",
                        value=3,
                        info="Number of times the model will see the entire dataset.",
                    )
                    batch_size = gr.Number(
                        label="Batch Size",
                        value=2,
                        info="Number of samples processed before the model is updated.",
                    )
                    learning_rate = gr.Number(
                        label="Learning Rate",
                        value=0.000001,
                        info="Step size for the optimizer during training. (eg: 1e-5 for 0.00001)",
                    )

                with gr.Row():
                    optimizer = gr.Dropdown(
                        label="Optimizer",
                        choices=["Adam", "SGD", "RMSprop"],
                        value="Adam",
                        info="Optimization algorithm to use during training.",
                    )
                    seed = gr.Number(
                        label="Seed",
                        value=42,
                        info="Random seed for reproducibility.",
                    )

            # Tab 3: Data Columns
            with gr.Tab("Data Columns"):
                with gr.Row():
                    input_column = gr.Textbox(
                        label="Input Column",
                        placeholder="e.g., text, question, article",
                        value="text",
                        info="Name of the column in the dataset containing the input data.",
                    )
                    output_column = gr.Textbox(
                        label="Output Column (optional)",
                        placeholder="e.g., answer, summary",
                        info="Name of the column in the dataset containing the output data (if applicable).",
                    )

            # Tab 4: Advanced Settings
            with gr.Tab("Advanced Settings"):
                with gr.Row():
                    wandb_api_key = gr.Textbox(
                        label="Wandb API Key",
                        placeholder="Enter your Wandb API Key",
                        info="Required for logging training metrics to Weights & Biases.",
                        type="password",
                    )
                    wandb_project_name = gr.Textbox(
                        label="Wandb Project Name",
                        placeholder="Enter your Wandb Project Name",
                        info="Name of the Wandb project to log results to.",
                    )
                    wandb_run_name = gr.Textbox(
                        label="Wandb Run Name (optional)",
                        placeholder="Enter your Wandb Run Name",
                        info="Name of the Wandb run to log results to.",
                    )

                # with gr.Row():
                #     file_upload = gr.File(
                #         label="Upload Dataset (CSV/JSON)",
                #     )
                #     file_type = gr.Dropdown(
                #         label="File Type",
                #         choices=["csv", "json"],
                #         value="csv",
                #         info="Select the format of the uploaded dataset.",
                #     )

        # Buttons
        with gr.Row():
            submit_button = gr.Button("Start Fine-Tuning", variant="primary")
            save_button = gr.Button("Save Configuration")
            load_button = gr.Button("Load Configuration")

        # Output
        output = gr.Textbox(label="Training Logs", interactive=False)

        # Show WIP message when task type is changed
        task_type.change(
            fn=show_wip_message,
            inputs=task_type,
            outputs=None,
        )

        # Process inputs when the button is clicked
        def process_inputs(
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
            # file_upload,
            # file_type,
            wandb_run_name,
        ):
            try:
                logger.info("Starting fine-tuning process...")
                # if file_upload is not None:
                #     # dataset_name = file_upload.name
                # else:
                dataset_name = dataset_name

                logs = finetune_model(
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
                    # file_upload,
                    # file_type,
                    wandb_run_name,
                )
                logger.info("Fine-tuning completed successfully!")
                return logs
            except Exception as e:
                logger.error(f"Error during training: {str(e)}")
                return f"Error during training: {str(e)}"

        # Save configuration
        def save_config(*args):
            try:
                config = {
                    "dataset_name": args[0],
                    "config_name": args[1],
                    "model_name": args[2],
                    "hf_token": args[3],
                    "wandb_api_key": args[4],
                    "wandb_project_name": args[5],
                    "task_type": args[6],
                    "num_epochs": args[7],
                    "batch_size": args[8],
                    "learning_rate": args[9],
                    "optimizer": args[10],
                    "seed": args[11],
                    "input_column": args[12],
                    "output_column": args[13],
                    "wandb_run_name": args[14],
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
                    config.get("hf_token", ""),
                    config.get("wandb_api_key", ""),
                    config.get("wandb_project_name", ""),
                    config.get("task_type", "Text Generation"),
                    config.get("num_epochs", 3),
                    config.get("batch_size", 2),
                    config.get("learning_rate", "1e-5"),
                    config.get("optimizer", "Adam"),
                    config.get("seed", 42),
                    config.get("input_column", "text"),
                    config.get("output_column", ""),
                    config.get("wandb_run_name", ""),
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
                # file_upload,
                # file_type,
                wandb_run_name,
            ],
            outputs=output,
        )

        save_button.click(
            fn=save_config,
            inputs=[
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
                wandb_run_name,
            ],
            outputs=output,
        )

        load_button.click(
            fn=load_config,
            outputs=[
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
                wandb_run_name,
            ],
        )

    demo.launch()


# Call the main function
if __name__ == "__main__":
    logger.info("Starting AutoFT application...")
    main()
