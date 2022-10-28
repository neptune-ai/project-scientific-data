import neptune.new as neptune
from neptune.new.types import File
import pandas as pd
import plotly.express as px

project = neptune.init_project()

data = '../data/covid_and_healthy_spectra.csv'

df = pd.read_csv(data)

# Upload data
project["data/files"].upload(data)

# Histogram of classes present in the data
project["data/class_balance"] = neptune.types.File.as_html(px.histogram(df.diagnostic))

# Upload samples
samples = df.sample(20)
project["data/samples"].upload(File.from_content(samples.to_html(), extension="html"))

