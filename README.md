### Run with Docker Compose

- Clone the repository:
- ```
  git clone https://github.com/StrVlad/Number_Classification
  cd Nuber_Classification
  ```

- Set up the environment:
- ```
  python3 -m venv .venv
  source .venv/bin/activate
  ```
  ```
  export AIRFLOW_HOME="$PWD/services/airflow"
  export AIRFLOW__API__WORKERS=1
  ```
- Install dependencies
  ```
  pip install --upgrade pip setuptools wheel
  pip install "apache-airflow==2.9.3" --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.9.3/constraints-3.11.txt"
  pip install -r requirements.txt
  ```
- Start
  **airflow standalone**
  Frontend (Streamlit): **http://localhost:8501**
  Airflow: **http://localhost:8080**
