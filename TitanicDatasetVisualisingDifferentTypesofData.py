import pandas as pd

# Load data from our dataset file into a pandas dataframe

dataset = pd.read_csv('datasets/titanic.csv', index_col=False, sep=",", header=0)

# Let's take a look at the data
print(dataset.head())
print(dataset.info())

import graphing

graphing.histogram(dataset, label_x='Pclass', label_y='Survived', histfunc='avg', include_boxplot=True).show()

graphing.multiple_histogram(dataset,
                            label_x='Pclass',  # group by ticket class
                            label_group="Parch",  # colour by no parents or children
                            label_y='Survived',
                            histfunc="avg").show()

graphing.box_and_whisker(dataset, label_x="Pclass", label_y="SibSp").show()

graphing.scatter_2D(dataset, label_x="Age", label_y="Fare").show()

# Plot Fare vs Survival
graphing.histogram(dataset, label_x="Fare", label_y="Survived", histfunc="avg", nbins=30, title="Fare vs Survival",
                   include_boxplot=True, show=True).show()

# Plot Age vs Survival
graphing.histogram(dataset, label_x="Age", label_y="Survived", histfunc="avg", title="Age vs Survival", nbins=30,
                   include_boxplot=True).show()

import plotly.graph_objects as go
import numpy as np


# Create some simple functions
# Read their descriptions to find out more
def get_rows(sex, port):
    '''Returns rows that match in terms of sex and embarkment port'''
    return dataset[(dataset.Embarked == port) & (dataset.Sex == sex)]


def proportion_survived(sex, port):
    '''Returns the proportion of people meeting criteria who survived'''
    survived = get_rows(sex, port).Survived
    return np.mean(survived)


# Make two columns of data - together these represent each combination
# of sex and embarkment port
sexes = ["male", "male", "male", "female", "female", "female"]
ports = ["C", "Q", "S"] * 2

# Calculate the number of passengers at each port + sex combination
passenger_count = [len(get_rows(sex, port)) for sex, port in zip(sexes, ports)]

# Calculate the proportion of passengers from each port + sex combination who survived
passenger_survival = [proportion_survived(sex, port) for sex, port in zip(sexes, ports)]

# Combine into a single data frame
table = pd.DataFrame(dict(
    sex=sexes,
    port=ports,
    passenger_count=passenger_count,
    passenger_survival_rate=passenger_survival
))

# Make a bubble plot
# This is just a scatter plot but each entry in the plot
# has a size and colour. We set colour to passenger_survival
# and size to the number of passengers
graphing.scatter_2D(table,
                    label_colour="passenger_survival_rate",
                    label_size="passenger_count",
                    size_multiplier=0.3,
                    title="Bubble Plot of Categorical Data").show()
