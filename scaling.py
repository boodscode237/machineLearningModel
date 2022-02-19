import pandas

# Convert it into a table using pandas
data = pandas.read_csv("datasets/dog-training.csv", delimiter="\t")

# Print the data
print(data)

from m1b_gradient_descent import gradient_descent
import numpy
import graphing

# Train model using Gradient Descent
# This method uses custom code that will print out progress as training advances.
# You don't need to inspect how this works for these exercises, but if you are
# curious, you can find it in out GitHub repository

model = gradient_descent(data.month_old_when_trained, data.mean_rescues_per_year, learning_rate=5E-4, number_of_iterations=8000)

# Plot the data and trendline after training
graphing.scatter_2D(data, "month_old_when_trained", "mean_rescues_per_year", trendline=model.predict)

# Add the standardized verions of "age_when_trained" to the dataset.
# Notice that it "centers" the mean age around 0
data["standardized_age_when_trained"] = (data.month_old_when_trained - numpy.mean(data.month_old_when_trained)) / (numpy.std(data.month_old_when_trained))

# Print a sample of the new dataset
print(data[:5])

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

fig = px.box(data,y=["month_old_when_trained", "standardized_age_when_trained"])
fig.show()

# Let's retrain our model, this time using the standardized feature
model_norm = gradient_descent(data.standardized_age_when_trained, data.mean_rescues_per_year, learning_rate=5E-4, number_of_iterations=8000)

# Plot the data and trendline again, after training with standardized feature
graphing.scatter_2D(data, "standardized_age_when_trained", "mean_rescues_per_year", trendline=model_norm.predict, show=True)

cost1 = model.cost_history
cost2 = model_norm.cost_history

# Creates dataframes with the cost history for each model
df1 = pandas.DataFrame({"cost": cost1, "Model":"No feature scaling"})
df1["number of iterations"] = df1.index + 1
df2 = pandas.DataFrame({"cost": cost2, "Model":"With feature scaling"})
df2["number of iterations"] = df2.index + 1

# Concatenate dataframes into a single one that we can use in our plot
df = pandas.concat([df1, df2])

# Plot cost history for both models
fig = graphing.scatter_2D(df, label_x="number of iterations", label_y="cost", title="Training Cost vs Iterations", label_colour="Model")
fig.update_traces(mode='lines')
fig.show()

