from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import torch
from datasets import load_dataset
import ray
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
from torch import nn, optim
import os

default_args = {
    'retries': 3,
}

dag = DAG(
    'ai_infra_pipeline',
    default_args=default_args,
    schedule_interval='@daily',
    start_date=datetime(2024, 2, 20),  # Adjusted for testing
    catchup=False,
)

# Persistent data directory
DATA_DIR = './data'
os.makedirs(DATA_DIR, exist_ok=True)

def etl():
    print("Starting ETL...")
    ds = load_dataset("mnist", split="train")
    data_path = os.path.join(DATA_DIR, 'mnist.pt')
    torch.save(ds, data_path)
    print(f"Dataset saved to {data_path} with {len(ds)} samples")
    return data_path

def quant_sim(ti):
    path = ti.xcom_pull(task_ids='etl')
    ds = torch.load(path)
    images = torch.tensor([img for img in ds['image']], dtype=torch.float32).view(-1, 784)  # Flatten MNIST images
    labels = torch.tensor(ds['label'], dtype=torch.long)
    # Dummy quantized matmul
    weights = torch.randn(784, 10)
    output = torch.matmul(images, weights)
    quant_path = os.path.join(DATA_DIR, 'quant.pt')
    torch.save({'output': output, 'labels': labels}, quant_path)
    print(f"Quantized output saved to {quant_path}")
    return quant_path

def ray_train(ti):
    os.environ["RAY_ADDRESS"] = "auto"
    ray.init(ignore_reinit_error=True)
    path = ti.xcom_pull(task_ids='quant_sim')
    data = torch.load(path)
    inputs = data['output']
    targets = data['labels']

    def train_fn():
        model = nn.Linear(784, 10)
        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()
        for epoch in range(5):
            logits = model(inputs)
            loss = criterion(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}: Loss {loss.item()}")
        model_path = os.path.join(DATA_DIR, 'model.pt')
        torch.save(model.state_dict(), model_path)
        return model_path

    trainer = TorchTrainer(
        train_fn,
        scaling_config=ScalingConfig(num_workers=2),
    )
    result = trainer.fit()
    # Assume train_fn saves the model; in real use, handle result
    return os.path.join(DATA_DIR, 'model.pt')

def serve_model(ti):
    model_path = ti.xcom_pull(task_ids='ray_train')
    model = nn.Linear(784, 10)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    test_input = torch.randn(1, 784)
    output = model(test_input)
    print(f"Serving output for dummy input: {output.argmax().item()}")

etl_task = PythonOperator(task_id='etl', python_callable=etl, dag=dag)
quant_task = PythonOperator(task_id='quant_sim', python_callable=quant_sim, dag=dag)
train_task = PythonOperator(task_id='ray_train', python_callable=ray_train, dag=dag)
serve_task = PythonOperator(task_id='serve_model', python_callable=serve_model, dag=dag)

etl_task >> quant_task >> train_task >> serve_task
