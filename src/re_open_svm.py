import pickle

import neptune.new as neptune
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

DATA_PATH = "../data/covid_and_healthy_spectra.csv"

###############
# Re-open run #
###############

# (neptune) fetch project
project = neptune.get_project()

# (neptune) this run
run = neptune.init(
    run="SCIDATA-26",
    capture_hardware_metrics=False,
    capture_stderr=False,
    capture_stdout=False,
)

#############################
# Fetch params from the run #
#############################
config = run["config"].fetch()

# data management
df = pd.read_csv(DATA_PATH)

if config["column_select"]:
    df = df[df.columns[:: config["nth_column"]]]

df.diagnostic = df.diagnostic.apply(lambda x: 1 if x == "SARS-CoV-2" else 0)

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

###############################
# Download model from the run #
###############################
run["modeling/pickled_model"].download("pickled_model.pkl")
with open("pickled_model.pkl", "rb") as file:
    clf = pickle.load(file)

y_train_pred = clf.predict(X_train)
y_val_pred = clf.predict(X_val)
y_test_pred = clf.predict(X_test)

run["modeling/train/recall"] = recall_score(y_train, y_train_pred)
run["modeling/valid/recall"] = recall_score(y_val, y_val_pred)
run["modeling/test/recall"] = recall_score(y_test, y_test_pred)
