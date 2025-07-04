import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


dataset = pd.read_csv("csv/personality_dataset.csv")

# Extroverts = 1, Introvert = 0
dataset["Personality"] = (dataset["Personality"] == "Extrovert").astype(int)


# Yes = 1, No = 0
dataset["Drained_after_socializing"] = (dataset["Drained_after_socializing"] == "Yes").astype(int)


print(len(dataset["Personality"] == "Extrovert"))
print(len(dataset["Personality"] == "Introvert"))