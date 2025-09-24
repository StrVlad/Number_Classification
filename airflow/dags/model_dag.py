from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime
import os

# Default arguments for the DAG
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
}

# Get the absolute path to the project root
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

with DAG(
    "torch_pipeline",
    default_args=default_args,
    schedule='*/5 * * * *',
    start_date=datetime(2025, 1, 1),
    catchup=False,
)as dag:

    train_model = BashOperator(
        task_id = "train_model",
        bash_command= f"cd {PROJECT_DIR} && python3 -m models.save_model"
    )

    deploy = BashOperator(
        task_id = "deploy",
        bash_command = f"cd {PROJECT_DIR} && docker-compose down && docker-compose up --build"
    )

    train_model >> deploy
