# Setup
1. Clone the repository to local machine using `git clone`
2. Setup virtual environment inside folder `python -m venv .`
3. Activate virtual environment `.\Scripts\activate`
4. Run `pip install -r requirements.txt` to setup project
5. Rename `config-template.json` to `config.json` with actual values set in placeholder
6. Install spacy libraries by running `python -m spacy download en_core_web_sm`

# Execution
1. Execute `python .\src\main.py` to train the model with mlflow
2. To view the results, run `mlflow ui --backend-store-uri sqlite:///mlflow.db`
3. To run airflow `docker compose up`