# Purpose of this:

Basically want to pivot AutoTrain's functionality but for finetuning LLMs. Not that great of idea but I think it's worth a shot.

# How would it work:

A streamlit or a gradio interface where the user will pass a dataset and model name (both needs to be present on Huggingface). Then user will choose the task (classification, text-generation, etc). Then the user will pass in the hyperparameters, but the user can also choose to use the default hyperparameters. And the logging will be done by Weight and Biases (wandb, The user also need to pass their wandb API key).

Then the user will click on the finetune button and then the model will be finetuned on the dataset and the logs will be stored on wandb. The user can run this on their local machine or on a cloud instance by just cloning the repository, installing the dependencies and running the main file.

# How to go ahead with this thing:

First start with just one task like text-generation, as there are tons of resources available for finetuning LLMs for text-generation. Using those resources I can figure out the hyperparameters and then creating a gradio interface for that won't be that hard. Then I can move on to other tasks like classification, etc.

# What I need to do:

- [x] Create a gradio interface
- [x] Figure out different hyperparameters for text-generation
- [x] Start implementing the finetuning part
- [ ] Figure the GPU and CPU part
- [x] Figure out the logging part
- [ ] Write the documentation

# Things I think which are difficult:

The most difficult part would be to figure out the hyperparameters for different tasks. But I think I can manage that by looking at the resources available online. Also finetuning LLMs require GPUs, so I need to figure out if I can just have a toggle button for the user to choose between CPU and GPU. If the user chooses either one of them then I need check if I can pass it in the code with device argument.