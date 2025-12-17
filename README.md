# Spam Classification Project

This project contains a Jupyter Notebook to train a Spam Classification model (IndoBERT) using the `labeled_kandidat_spam.csv` dataset.

## Prerequisites

- Python 3.8 or higher
- `pip` (Python Package Installer)

## Setup and Installation

1.  **Clone or Download** this repository/folder to your local machine.
2.  **Navigate** to the folder directory in your terminal:
    ```bash
    cd /Users/najj/Documents/PunyaBotRobot/project/NLP
    ```
    _(Or wherever you downloaded the project)_
3.  **Install Jupyter** (if not already installed):
    ```bash
    pip install notebook
    ```

## How to Run

1.  Start Jupyter Notebook:

    ```bash
    jupyter notebook
    ```

    Or open the project folder in VS Code and open `Project_UAS_NLP.ipynb`.

2.  Open **`Project_UAS_NLP.ipynb`**.

3.  Run the cells sequentially (Shift + Enter).

    - **Step 1**: Installs necessary library dependencies (`transformers`, `datasets`, `torch`, etc.).
    - **Step 2**: Loads the dataset `labeled_kandidat_spam.csv`.
    - **Step 3-5**: Tokenizes data, sets up the IndoBERT model, and trains it for 15 epochs.
    - **Step 6**: Evaluates the model accuracy (Target: >85%).
    - **Step 7**: Runs a quick demo to predict sample text.

4.  The trained model will be saved in the `indobert-spam-detection-final` directory.

## Files

- `Project_UAS_NLP.ipynb`: The main notebook containing all steps.
- `labeled_kandidat_spam.csv`: The labeled dataset.
- `requirements.txt`: List of python dependencies (optional, as the notebook installs them).
