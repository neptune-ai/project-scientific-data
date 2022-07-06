import neptune.new as neptune
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from nn_training import train, test
from nn_utils import load_and_standardize_data, DataBuilder, AutoEncoder
from utils import log_dataset, log_raw_data

DATA_PATH = "../data/covid_and_healthy_spectra.csv"

run = neptune.init_run(
    source_files=["*.py", "../environment.yml"],
    tags="autoencoder",
)

# configuration
config = {
    "test_size": 0.40,
    "val_size": 0.50,
    "seed": 2022,
    "log_model": True,
    "epochs": 3,
    "learning_rate": 1e-2,
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

X_train, X_test, y_train, y_test = load_and_standardize_data(
    df, X, y, config["test_size"], config["seed"]
)

# Data loaders
traindata_set = DataBuilder(df, X, y, config, train=True)
testdata_set = DataBuilder(df, X, y, config, train=False)
trainloader = DataLoader(dataset=traindata_set, batch_size=16)
testloader = DataLoader(dataset=testdata_set, batch_size=16)

# (neptune) log metadata for train, test
log_dataset(run=run, base_namespace="data/train", data=trainloader.dataset.x)
log_dataset(run=run, base_namespace="data/test", data=testloader.dataset.x)

# model
model = AutoEncoder().to(device).double()
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
loss_mse = nn.MSELoss(reduction="sum")

# 1st training session, 3 epochs
for epoch in range(config["epochs"]):
    train(run, "modeling", model, trainloader, device, optimizer, loss_mse)
    test(run, "modeling", model, testloader, device, optimizer, loss_mse)

# (neptune) log model weights
torch.save(model.state_dict(), "../model/ae_model_e3.pth")
if config["log_model"]:
    run["modeling/model"].upload("../model/ae_model_e3.pth")

# 2nd training session, 20 epochs
for epoch in range(20):
    train(run, "modeling", model, trainloader, device, optimizer, loss_mse)
    test(run, "modeling", model, testloader, device, optimizer, loss_mse)

# (neptune) log model weights
torch.save(model.state_dict(), "../model/ae_model_e23.pth")
if config["log_model"]:
    run["modeling/model"].upload("../model/ae_model_e23.pth")
