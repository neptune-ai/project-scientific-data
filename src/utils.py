import neptune.new as neptune
import pandas as pd
import plotly.express as px
from sklearn.metrics import precision_score, accuracy_score, recall_score


def log_raw_data(run: neptune.Run, base_namespace: str, df: pd.DataFrame):
    run[f"{base_namespace}/version"].track_files(
        "../data/covid_and_healthy_spectra.csv"
    )

    run[f"{base_namespace}/n_rows"] = df.shape[0]
    run[f"{base_namespace}/n_cols"] = df.shape[1]

    run[f"{base_namespace}/target/n_Healthy"] = df.diagnostic.value_counts()["Healthy"]
    run[f"{base_namespace}/target/n_SARS-CoV-2"] = df.diagnostic.value_counts()[
        "SARS-CoV-2"
    ]
    run[f"{base_namespace}/target/class_balance"] = neptune.types.File.as_html(
        px.histogram(df.diagnostic)
    )


def log_dataset(
    run: neptune.Run,
    base_namespace: str,
    data: pd.DataFrame,
    target: pd.DataFrame,
    sample_size: int = 10,
):
    run[f"{base_namespace}/n_rows"] = data.shape[0]
    run[f"{base_namespace}/n_cols"] = data.shape[1]
    run[f"{base_namespace}/sample"] = data.head(n=sample_size)

    run[f"{base_namespace}/target/n_Healthy"] = target.value_counts()[0]
    run[f"{base_namespace}/target/n_SARS-CoV-2"] = target.value_counts()[1]
    run[f"{base_namespace}/target/class_balance"] = neptune.types.File.as_html(
        px.histogram(target)
    )


def log_training_report(run: neptune.Run, base_namespace: str, y_data: zip):
    for dataset, y_pair in zip(["train", "valid", "test"], y_data):
        run[f"{base_namespace}/{dataset}/precision"] = precision_score(
            y_pair[0], y_pair[1]
        )
        run[f"{base_namespace}/{dataset}/accuracy"] = accuracy_score(
            y_pair[0], y_pair[1]
        )
        run[f"{base_namespace}/{dataset}/recall"] = recall_score(y_pair[0], y_pair[1])
