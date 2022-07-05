import neptune.new as neptune
import numpy as np
import pandas as pd
import plotly.express as px
import sklearn.decomposition
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

    df.head(n=30).to_csv("data_sample.csv")
    run[f"{base_namespace}/sample"].upload("data_sample.csv")


def log_dataset(
    run: neptune.Run,
    base_namespace: str,
    data: pd.DataFrame,
    target: pd.Series,
):
    run[f"{base_namespace}/n_rows"] = data.shape[0]
    run[f"{base_namespace}/n_cols"] = data.shape[1]

    run[f"{base_namespace}/target/n_Healthy"] = target.value_counts()[0]
    run[f"{base_namespace}/target/n_SARS-CoV-2"] = target.value_counts()[1]
    run[f"{base_namespace}/target/class_balance"] = neptune.types.File.as_html(
        px.histogram(target, text_auto=True)
    )


def log_training_report(run: neptune.Run, base_namespace: str, y_data: zip):
    for dataset, y_pair in zip(["train", "valid", "test"], y_data):
        run[f"{base_namespace}/{dataset}/precision"] = precision_score(
            y_pair[0], y_pair[1]
        )
        run[f"{base_namespace}/{dataset}/accuracy"] = accuracy_score(
            y_pair[0], y_pair[1]
        )
        # run[f"{base_namespace}/{dataset}/recall"] = recall_score(y_pair[0], y_pair[1])


def log_pca(run: neptune.Run, base_namespace: str, pca: sklearn.decomposition.PCA):
    run[f"{base_namespace}/explained_variance_ratio"].log(
        list(pca.explained_variance_ratio_)
    )
    run[f"{base_namespace}/singular_values"].log(list(pca.singular_values_))

    exp_var = np.cumsum(pca.explained_variance_ratio_)

    fig = px.area(
        x=range(1, exp_var.shape[0] + 1),
        y=exp_var,
        labels={"x": "# Components", "y": "Explained Variance"},
    )

    run[f"{base_namespace}/explained_variance_chart"] = neptune.types.File.as_html(fig)
