import pandas

# Convert it into a table using pandas
dataset = pandas.read_csv("datasets/doggy-illness.csv", delimiter="\t")

# Print the data
print(dataset)

import statsmodels.formula.api as smf
import graphing  # custom graphing code. See our GitHub repo for details

# Perform linear regression. This method takes care of
# the entire fitting procedure for us.
simple_formula = "core_temperature ~ protein_content_of_last_meal"
simple_model = smf.ols(formula=simple_formula, data=dataset).fit()

# Show a graph of the result
graphing.scatter_2D(dataset, label_x="protein_content_of_last_meal",
                    label_y="core_temperature",
                    trendline=lambda x: simple_model.params[1] * x + simple_model.params[0],
                    show=True)

print("R-squared:", simple_model.rsquared)

# Perform polynomial regression. This method takes care of
# the entire fitting procedure for us.
polynomial_formula = "core_temperature ~ protein_content_of_last_meal + I(protein_content_of_last_meal**2)"
polynomial_model = smf.ols(formula=polynomial_formula, data=dataset).fit()

# Show a graph of the result
graphing.scatter_2D(dataset, label_x="protein_content_of_last_meal",
                    label_y="core_temperature",
                    # Our trendline is the equation for the polynomial
                    trendline=lambda x: polynomial_model.params[2] * x ** 2 + polynomial_model.params[1] * x +
                                        polynomial_model.params[0])

print("R-squared:", polynomial_model.rsquared)

import numpy as np

fig = graphing.surface(
    x_values=np.array([min(dataset.protein_content_of_last_meal), max(dataset.protein_content_of_last_meal)]),
    y_values=np.array([min(dataset.protein_content_of_last_meal) ** 2, max(dataset.protein_content_of_last_meal) ** 2]),
    calc_z=lambda x, y: polynomial_model.params[0] + (polynomial_model.params[1] * x) + (
            polynomial_model.params[2] * y),
    axis_title_x="x",
    axis_title_y="x2",
    axis_title_z="Core temperature"
)
# Add our datapoints to it and display
fig.add_scatter3d(x=dataset.protein_content_of_last_meal, y=dataset.protein_content_of_last_meal ** 2,
                  z=dataset.core_temperature, mode='markers')
fig.show()

# Show an extrapolated graph of the linear model
graphing.scatter_2D(dataset, label_x="protein_content_of_last_meal",
                    label_y="core_temperature",
                    # We extrapolate over the following range
                    x_range=[0, 100],
                    trendline=lambda x: simple_model.params[1] * x + simple_model.params[0], show=True)

# Show an extrapolated graph of the polynomial model
graphing.scatter_2D(dataset, label_x="protein_content_of_last_meal",
                    label_y="core_temperature",
                    # We extrapolate over the following range
                    x_range=[0, 100],
                    trendline=lambda x: polynomial_model.params[2] * x ** 2 + polynomial_model.params[1] * x +
                                        polynomial_model.params[0], show=True)
