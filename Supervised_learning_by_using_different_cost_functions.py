import pandas
from datetime import datetime

# Load a file that contains our weather data
dataset = pandas.read_csv('datasets/seattleWeather_1948-2017.csv', parse_dates=['date'])

# Convert the dates into numbers so we can use them in our models
# We make a year column that can contain fractions. For example,
# 1948.5 is halfway through the year 1948
dataset["year"] = [(d.year + d.timetuple().tm_yday / 365.25) for d in dataset.date]


# For the sake of this exercise, let's look at February 1 for the following years:
desired_dates = [
    datetime(1950,2,1),
    datetime(1960,2,1),
    datetime(1970,2,1),
    datetime(1980,2,1),
    datetime(1990,2,1),
    datetime(2000,2,1),
    datetime(2010,2,1),
    datetime(2017,2,1),
]

dataset = dataset[dataset.date.isin(desired_dates)].copy()

# Print the dataset
print(dataset)

import numpy

def sum_of_square_differences(estimate, actual):
    # Note that with NumPy, to square each value we use **
    return numpy.sum((estimate - actual)**2)

def sum_of_absolute_differences(estimate, actual):
    return numpy.sum(numpy.abs(estimate - actual))


actual_label = numpy.array([1, 3])
model_estimate = numpy.array([2, 2])

print("SSD:", sum_of_square_differences(model_estimate, actual_label))
print("SAD:", sum_of_absolute_differences(model_estimate, actual_label))

from microsoft_custom_linear_regressor import MicrosoftCustomLinearRegressor
import graphing

# Create and fit the model
# We use a custom object that we've hidden from this notebook, because
# you don't need to understand its details. This fits a linear model
# by using a provided cost function

# Fit a model by using sum of square differences
model = MicrosoftCustomLinearRegressor().fit(X = dataset.year,
                                             y = dataset.min_temperature,
                                             cost_function = sum_of_square_differences)

# Graph the model
graphing.scatter_2D(dataset,
                    label_x="year",
                    label_y="min_temperature",
                    trendline=model.predict).show()

# Fit a model with SSD
# Fit a model by using sum of square differences
model = MicrosoftCustomLinearRegressor().fit(X = dataset.year,
                                             y = dataset.min_temperature,
                                             cost_function = sum_of_absolute_differences)

# Graph the model
graphing.scatter_2D(dataset,
                    label_x="year",
                    label_y="min_temperature",
                    trendline=model.predict).show()
