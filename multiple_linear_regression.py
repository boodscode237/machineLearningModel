import pandas

# Convert it into a table using pandas
dataset = pandas.read_csv("datasets/doggy-illness.csv", delimiter="\t")

# Print the data
print(dataset)

import graphing  # Custom graphing code that uses Plotly. See our GitHub repository for details

graphing.box_and_whisker(dataset, "male", "core_temperature", show=True)
graphing.box_and_whisker(dataset, "attended_training", "core_temperature", show=True)
graphing.box_and_whisker(dataset, "ate_at_tonys_steakhouse", "core_temperature", show=True)
graphing.scatter_2D(dataset, "body_fat_percentage", "core_temperature", show=True)
graphing.scatter_2D(dataset, "protein_content_of_last_meal", "core_temperature", show=True)
graphing.scatter_2D(dataset, "age", "core_temperature")

import statsmodels.formula.api as smf
import graphing  # custom graphing code. See our GitHub repo for details

for feature in ["male", "age", "protein_content_of_last_meal", "body_fat_percentage"]:
    # Perform linear regression. This method takes care of
    # the entire fitting procedure for us.
    formula = "core_temperature ~ " + feature
    simple_model = smf.ols(formula=formula, data=dataset).fit()

    print(feature)
    print("R-squared:", simple_model.rsquared)

    # Show a graph of the result
    graphing.scatter_2D(dataset, label_x=feature,
                        label_y="core_temperature",
                        title=feature,
                        trendline=lambda x: simple_model.params[1] * x + simple_model.params[0],
                        show=True)

formula = "core_temperature ~ age"
age_trained_model = smf.ols(formula=formula, data=dataset).fit()
age_naive_model = smf.ols(formula=formula, data=dataset).fit()
age_naive_model.params[0] = dataset['core_temperature'].mean()
age_naive_model.params[1] = 0

print("naive R-squared:", age_naive_model.rsquared)
print("trained R-squared:", age_trained_model.rsquared)

# Show a graph of the result
graphing.scatter_2D(dataset, label_x="age",
                    label_y="core_temperature",
                    title="Naive model",
                    trendline=lambda x: dataset['core_temperature'].mean().repeat(len(x)),
                    )
# Show a graph of the result
graphing.scatter_2D(dataset, label_x="age",
                    label_y="core_temperature",
                    title="Trained model",
                    trendline=lambda x: age_trained_model.params[1] * x + age_trained_model.params[0],
                    show=True)
model = smf.ols(formula = "core_temperature ~ age + male", data = dataset).fit()

print("R-squared:", model.rsquared)

import numpy as np
# Show a graph of the result
# this needs to be 3D, because we now have three variables in play: two features and one label

def predict(age, male):
    '''
    This converts given age and male values into a prediction from the model
    '''
    # to make a prediction with statsmodels, we need to provide a dataframe
    # so create a dataframe with just the age and male variables
    df = pandas.DataFrame(dict(age=[age], male=[male]))
    return model.predict(df)

# Create the surface graph
fig = graphing.surface(
    x_values=np.array([min(dataset.age), max(dataset.age)]),
    y_values=np.array([0, 1]),
    calc_z=predict,
    axis_title_x="Age",
    axis_title_y="Male",
    axis_title_z="Core temperature"
)

# Add our datapoints to it and display
fig.add_scatter3d(x=dataset.age, y=dataset.male, z=dataset.core_temperature, mode='markers')
fig.show()

# Print summary information
print(model.summary())
print(age_trained_model.summary())

