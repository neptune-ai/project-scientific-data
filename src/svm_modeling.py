import neptune.new as neptune
import pandas as pd
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from utils import log_dataset, log_raw_data, log_training_report, log_pca

DATA_PATH = "../data/covid_and_healthy_spectra.csv"

run = neptune.init_run(
    source_files=["*.py", "../environment.yml"],
    tags="svm",
)

# configuration
config = {
    "test_size": 0.40,
    "val_size": 0.50,
    "scaler": True,
    "pca": True,
    "n_components": 5,
    "seed": 2022,
    "column_select": False,
    "nth_column": 10,
    "log_model": True,
    "col_840_threshold": True,
}

run["config"] = config

# data management
df = pd.read_csv(DATA_PATH)

# (neptune) log data version, data dimensions, target occurrences
log_raw_data(run=run, base_namespace="data/raw", df=df)

if config["column_select"]:
    df = df[df.columns[:: config["nth_column"]]]

df.diagnostic = df.diagnostic.apply(lambda x: 1 if x == "SARS-CoV-2" else 0)

# use only data where col["840"] > 0.1
if config["col_840_threshold"]:
    print(f"data shape before: {df.shape}")
    df = df[df["840"] > 0.1]
    print(f"data shape after:  {df.shape}")

y = df.diagnostic
X = df[df.columns.drop("diagnostic")]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=config["test_size"], random_state=config["seed"]
)
X_test, X_val, y_test, y_val = train_test_split(
    X_test, y_test, test_size=config["val_size"], random_state=config["seed"]
)

# modeling pipeline, feature management, metrics reporting
if config["scaler"]:
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)

if config["pca"]:
    pca = PCA(n_components=config["n_components"])
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    X_val = pca.transform(X_val)

    # (neptune) log PCA results
    log_pca(run=run, base_namespace="data/pca", pca=pca)

# (neptune) log metadata for train, valid, test
log_dataset(run=run, base_namespace="data/train", data=X_train, target=y_train)
log_dataset(run=run, base_namespace="data/valid", data=X_val, target=y_val)
log_dataset(run=run, base_namespace="data/test", data=X_test, target=y_test)


clf = svm.SVC()
clf.fit(X_train, y_train)

y_train_pred = clf.predict(X_train)
y_val_pred = clf.predict(X_val)
y_test_pred = clf.predict(X_test)

y_data = zip((y_train, y_val, y_test), (y_train_pred, y_val_pred, y_test_pred))

log_training_report(run=run, base_namespace="modeling", y_data=y_data)

if config["log_model"]:
    run["modeling/pickled_model"] = neptune.types.File.as_pickle(clf)
