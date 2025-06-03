import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("csv/personality_dataset.csv")

counts = dataset["Personality"].value_counts()



