import neptune.new as neptune
import numpy as np
import pandas as pd
import torch
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from nn_utils import DataBuilder, AutoEncoder
from utils import log_dataset, log_raw_data, log_training_report

DATA_PATH = "../data/covid_and_healthy_spectra.csv"

run = neptune.init_run(
    source_files=["*.py", "../environment.yml"],
    tags=["autoencoder", "svm", "embeddings"],
)

# configuration
config = {
    "test_size": 0.40,
    "val_size": 0.50,
    "scaler": False,
    "seed": 2022,
    "log_model": True,
}

run["config"] = config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run["config/device"] = device

# data management
df = pd.read_csv(DATA_PATH)

# (neptune) log data version, data dimensions, target occurrences
log_raw_data(run=run, base_namespace="data/raw", df=df)

df.diagnostic = df.diagnostic.apply(lambda x: 1 if x == "SARS-CoV-2" else 0)
y = df.diagnostic
X = df[df.columns.drop("diagnostic")]

# Data loaders
traindata_set = DataBuilder(df, X, y, config, train=True)
testdata_set = DataBuilder(df, X, y, config, train=False)
trainloader = DataLoader(dataset=traindata_set, batch_size=1000)
testloader = DataLoader(dataset=testdata_set, batch_size=1000)

# (neptune) log metadata for train, test
log_dataset(run=run, base_namespace="data/train", data=trainloader.dataset.x)
log_dataset(run=run, base_namespace="data/test", data=testloader.dataset.x)

##################################
# Re-open run and download model #
##################################

# (neptune) fetch project
project = neptune.get_project()

# (neptune) re-open "SCIDATA-38"
run_with_model = neptune.init(
    run="SCIDATA-39",
    mode="read-only",
)

# (neptune) download model from the run
run_with_model["modeling/model_weights/epoch_023"].download("model_023.pkl")
run_with_model.stop()

model = AutoEncoder().double()
model.load_state_dict(torch.load("model_023.pkl"))
model.eval()

# Generate embeddings for training and test data
for batch_idx, (inp, y_train) in enumerate(trainloader):
    data = inp.to(device).double().unsqueeze(dim=0).permute(1, 0, 2)
    out, train_emb = model(data)
for batch_idx, (inp, y_test) in enumerate(testloader):
    data = inp.to(device).double().unsqueeze(dim=0).permute(1, 0, 2)
    out, test_emb = model(data)

# Concatenate embeddings and labels
train_emb, test_emb = train_emb.cpu().detach().numpy(), test_emb.cpu().detach().numpy()
y_train, y_test = y_train.cpu().detach().numpy(), y_test.cpu().detach().numpy()
embeddings = np.concatenate([train_emb, test_emb])
y_emb = np.concatenate([y_train, y_test])

# Prepare data for training
X_train, X_test, y_train, y_test = train_test_split(
    embeddings, y_emb, test_size=config["test_size"], random_state=config["seed"]
)
X_test, X_val, y_test, y_val = train_test_split(
    X_test, y_test, test_size=config["val_size"], random_state=config["seed"]
)

if config["scaler"]:
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)

# SVC training
clf = svm.SVC()
clf.fit(X_train, y_train)

y_train_pred = clf.predict(X_train)
y_val_pred = clf.predict(X_val)
y_test_pred = clf.predict(X_test)

y_data = zip((y_train, y_val, y_test), (y_train_pred, y_val_pred, y_test_pred))

log_training_report(run=run, base_namespace="modeling", y_data=y_data)

if config["log_model"]:
    run["modeling/pickled_model"] = neptune.types.File.as_pickle(clf)
