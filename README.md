# AI Infra Pipeline

## Overview

This project demonstrates a basic end-to-end AI infrastructure pipeline using Apache Airflow for orchestration. It covers ETL (data loading), a simulated quantization step, distributed training with Ray, and model serving. Built with PyTorch and related tools, it's designed as a learning tool for understanding scalable AI workflows, from data ingestion to deployment.

Key goals:
- Prototype a modular ML pipeline.
- Incorporate distributed computing (Ray) for scalability.
- Highlight best practices for enterprise AI (e.g., retries, persistent storage).

This is a starting point—ideal for expanding into real-world scenarios like cloud storage integration or custom Triton kernels.

## Tech Stack
- **Orchestration**: Apache Airflow
- **Data Handling**: Hugging Face Datasets, PyTorch
- **Distributed Training**: Ray
- **Other**: vLLM (noted for LLMs, adjusted here for simple models)

## Setup Instructions

1. **Prerequisites**:
   - Python 3.8+
   - Apache Airflow: `pip install apache-airflow`
   - Required libraries: `pip install torch datasets ray[default] vllm`

2. **Clone the Repo**:
   ```
   git clone https://github.com/avuppal/ai-infra-pipeline.git
   cd ai-infra-pipeline
   ```

3. **Initialize Airflow** (if not already set up):
   ```
   airflow db init
   airflow users create --username admin --firstname Admin --lastname User --role Admin --email admin@example.com --password admin
   ```

4. **Copy DAG to Airflow**:
   - Place `dags/pipeline_dag.py` in your Airflow DAGs folder (e.g., `~/airflow/dags`).

5. **Run Airflow**:
   ```
   airflow webserver -p 8080
   airflow scheduler
   ```
   Access the UI at `http://localhost:8080` and enable the `ai_infra_pipeline` DAG.

## Pipeline Flow

1. **ETL**: Loads MNIST dataset and saves to persistent `./data/` directory.
2. **Quantization Simulation**: Flattens data and performs a dummy matmul (placeholder for real quantization/Triton ops).
3. **Ray Training**: Distributed training of a simple linear model using Ray on 2 workers.
4. **Serving**: Loads the model and runs basic inference (expand to API for production).

Tasks are chained sequentially with 3 retries for resilience.

## Running the Pipeline

- Trigger manually via Airflow UI or CLI: `airflow dags trigger ai_infra_pipeline`.
- Outputs are saved in `./data/` for inspection.

## Potential Improvements
- Integrate real quantization with Triton kernels.
- Use cloud storage (S3/GCS) instead of local `./data/`.
- Add monitoring, tests, and CI/CD.
- Scale to larger datasets/models for enterprise simulation.

## Contributing
Fork the repo, make changes, and submit a PR. Focus on enhancing scalability or adding real-world features.

## License
MIT License. See LICENSE for details.
