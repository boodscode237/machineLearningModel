import pandas

# Convert it into a table using pandas
dataset = pandas.read_csv("datasets/doggy-illness.csv", delimiter="\t")

# Print the data
print(dataset)

import graphing

graphing.histogram(dataset, label_x='age', nbins=10, title="Feature", show=True).show()
graphing.histogram(dataset, label_x='core_temperature', nbins=10, title="Label").show()
graphing.scatter_2D(dataset, label_x="age", label_y="core_temperature",
                    title='core temperature as a function of age').show()

import statsmodels.formula.api as smf

# First, we define our formula using a special syntax
# This says that core temperature is explained by age
formula = "core_temperature ~ age"

# Perform linear regression. This method takes care of
# the entire fitting procedure for us.
model = smf.ols(formula=formula, data=dataset).fit()

# Show a graph of the result
graphing.scatter_2D(dataset, label_x="age",
                    label_y="core_temperature",
                    trendline=lambda x: model.params[1] * x + model.params[0]
                    )

print("Intercept:", model.params[0], "Slope:", model.params[1])


def estimate_temperature(age):
    # Model param[0] is the intercepts and param[1] is the slope
    return age * model.params[1] + model.params[0]


print("Estimate temperature from age")
print(estimate_temperature(age=0))
