from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.ray.operators.ray import RayOperator
from datetime import datetime
import torch
from datasets import load_dataset
import ray
from vllm import LLM
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
import os

default_args = {
    'retries': 3,
}

dag = DAG(
    'ai_infra_pipeline',
    default_args=default_args,
    schedule_interval='@daily',
    start_date=datetime(2026, 2, 20),
    catchup=False,
)

def etl():
    ds = load_dataset("mnist", split="train")
    torch.save(ds, "/tmp/mnist.pt")
    return "/tmp/mnist.pt"

def triton_quant(ti):
    path = ti.xcom_pull(task_ids='etl')
    ds = torch.load(path)
    # Dummy quant matmul
    data = torch.tensor(ds['image']).float()
    weights = torch.randn(784, 10)
    output = torch.matmul(data, weights)
    torch.save(output, "/tmp/quant.pt")
    return "/tmp/quant.pt"

def ray_train(ti):
    os.environ["RAY_ADDRESS"] = "auto"
    ray.init()
    path = ti.xcom_pull(task_ids='triton_quant')
    data = torch.load(path)

    def train_fn():
        model = nn.Linear(784, 10)
        optimizer = optim.Adam(model.parameters())
        # Dummy DDP
        model = DDP(model)
        for epoch in range(1):
            loss = torch.nn.CrossEntropyLoss()(model(data), torch.randint(0, 10, (data.shape[0],)))
            loss.backward()
            optimizer.step()
        torch.save(model.state_dict(), "/tmp/model.pt")

    trainer = TorchTrainer(train_fn, scaling_config={"num_workers": 2})
    trainer.fit()

def vllm_serve():
    llm = LLM(model="/tmp/model.pt")
    outputs = llm.generate(["test"])
    print(outputs)

etl_task = PythonOperator(task_id='etl', python_callable=etl, dag=dag)
quant_task = PythonOperator(task_id='triton_quant', python_callable=triton_quant, dag=dag)
train_task = RayOperator(task_id='ray_train', command="ray_train", env_vars={"RAY_ADDRESS": "auto"}, dag=dag)
serve_task = PythonOperator(task_id='vllm_serve', python_callable=vllm_serve, dag=dag)

etl_task >> quant_task >> train_task >> serve_task
