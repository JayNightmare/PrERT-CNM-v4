# Exported PrERT Zip File

Inside this zip file contains all the things you need to run the PrERT model. To ensure you run everything correctly, please follow the instructions below.

## Steps

### 1. Unzip the file

Easy step, just unzip the file into a folder of your choice. You can use any unzip tool you like, such as `unzip` on Linux or macOS, or WinRAR/7-Zip on Windows.

### 2. Setup Python Environment

Make sure you have Python 3.11 installed on your system. You can check your Python version by running:

```bash
python --version
```

Create a virtual environment to keep dependencies isolated:

```bash
python -m venv .venv && source .venv/Scripts/activate && pip install --upgrade pip && pip install -e .
```

Once the command is complete, check if the cli tool works by running:

```bash
prert doctor
```

If it prints out the version and other information, you are good to go. If not, check the PATH is correct as this is the main way to run the codebase.

### 3. Run the PrERT CLI

The PrERT CLI has everything you need to get setup and run the model. You can check the commands available by running:

```bash
prert --help
```

The CLI offers a guide and an interactive interface to help you navigate through the available commands and options.

```bash
# Guided commands
prert guide

# Interactive commands - If you encounter an error, use the commands shown in the guide instead
prert interactive
```

### 4. Run the PrERT Web App

The PrERT web app is a user-friendly interface that allows you to interact with the model. To start the web app, run:

```bash
prert web
```

## Included Documentation

If at any point you need more information, please refer to the included documentation in the `docs/` folder. The documentation provides detailed explanations of the model, its components, and how to use it effectively.

- The `Project` folder contains the main codebase and resources for the PrERT model.
  - View `README.md` for a high-level overview of the project and its components.
  - Go into `Documentation` and view the pdf file to get a detailed report of the model which includes the model architecture, training process, and evaluation metrics.
  - Go into `Execution-Playbook` and view the markdown files for a break down of each phase
- The `Standards` folder contains the standards and regulations used by the model for compliance assessment.
- The `Proposal` folder contains the proposal and design documents for the model.
