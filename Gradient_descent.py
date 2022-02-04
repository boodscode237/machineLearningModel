from datetime import datetime
import pandas
import graphing  # Custom graphing code. See our GitHub repository

# Load a file that contains weather data for Seattle
data = pandas.read_csv('datasets/seattleWeather_1948-2017.csv', parse_dates=['date'])

# Remove all dates after July 1 because we have to to plant onions before summer begins
data = data[[d.month < 7 for d in data.date]].copy()

# Convert the dates into numbers so we can use them in our models
# We make a year column that can contain fractions. For example,
# 1948.5 is halfway through the year 1948
data["year"] = [(d.year + d.timetuple().tm_yday / 365.25) for d in data.date]

# Let's take a quick look at our data
print("Visual Check:")
graphing.scatter_2D(data,
                    label_x="year",
                    label_y="min_temperature",
                    title="Temperatures over time (°F)").show()

import statsmodels.formula.api as smf

# Perform linear regression to fit a line to our data
# NB OLS uses the sum or mean of squared differences as a cost function,
# which we're familiar with from our last exercise
model = smf.ols(formula="min_temperature ~ year", data=data).fit()

# Print the model
intercept = model.params[0]
slope = model.params[1]

print(f"The model is: y = {slope:0.3f} * X + {intercept:0.3f}")


class MyModel:

    def __init__(self):
        '''
        Creates a new MyModel
        '''
        # Straight lines described by two parameters:
        # The slope is the angle of the line
        self.slope = 0
        # The intercept moves the line up or down
        self.intercept = 0

    def predict(self, date):
        '''
        Estimates the temperature from the date
        '''
        return date * self.slope + self.intercept

    def get_summary(self):
        '''
        Returns a string that summarises the model
        '''
        return f"y = {self.slope} * x + {self.intercept}"


print("Model class ready")

import numpy as np

x = data.year
temperature_true = data.min_temperature

# We'll use a prebuilt method to show a 3D plot
# This requires a range of x values, a range of y values,
# and a way to calculate z
# Here, we set:
#   x to a range of potential model intercepts
#   y to a range of potential model slopes
#   z as the cost for that combination of model parameters

# Choose a range of intercepts and slopes values
intercepts = np.linspace(-100, -70, 10)
slopes = np.linspace(0.060, 0.07, 10)


# Set a cost function. This will be the mean of squared differences
def cost_function(temperature_estimate):
    """
    Calculates cost for a given temperature estimate
    Our cost function is the mean of squared differences (a.k.a. mean squared error)
    """
    # Note that with NumPy to square each value, we use **
    return np.mean((temperature_true - temperature_estimate) ** 2)


def predict_and_calc_cost(intercept, slope):
    '''
    Uses the model to make a prediction, then calculates the cost
    '''

    # Predict temperature by using these model parameters
    temperature_estimate = x * slope + intercept

    # Calculate cost
    return cost_function(temperature_estimate)


# Call the graphing method. This will use our cost function,
# which is above. If you want to view this code in detail,
# then see this project's GitHub repository

graphing.surface(x_values=intercepts, y_values=slopes, calc_z=predict_and_calc_cost,
                 title="Cost for Different Model Parameters", axis_title_x="Model intercept",
                 axis_title_y="Model slope", axis_title_z="Cost").show()

def calculate_gradient(temperature_estimate):
    """
    This calculates the gradient for a linear regession
    by using the Mean Squared Error cost function
    """

    # The partial derivatives of MSE are as follows
    # You don't need to be able to do this just yet, but
    # it's important to note that these give you the two gradients
    # that we need to train our model
    error = temperature_estimate - temperature_true
    grad_intercept = np.mean(error) * 2
    grad_slope = (x * error).mean() * 2

    return grad_intercept, grad_slope


print("Function is ready!")


def gradient_descent(learning_rate, number_of_iterations):
    """
    Performs gradient descent for a one-variable function.

    learning_rate: Larger numbers follow the gradient more aggressively
    number_of_iterations: The maximum number of iterations to perform
    """

    # Our starting guess is y = 0 * x - 83
    # We're going to start with the correct intercept so that
    # only the line's slope is estimated. This is just to keep
    # things simple for this exercise
    model = MyModel()
    model.intercept = -83
    model.slope = 0

    for i in range(number_of_iterations):
        # Calculate the predicted values
        predicted_temperature = model.predict(x)

        # == OPTIMIZER ===
        # Calculate the gradient
        _, grad_slope = calculate_gradient(predicted_temperature)
        # Update the estimation of the line
        model.slope -= learning_rate * grad_slope

        # Print the current estimation and cost every 100 iterations
        if( i % 100 == 0):
            estimate = model.predict(x)
            cost = cost_function(estimate)
            print("Next estimate:", model.get_summary(), f"Cost: {cost}")

    # Print the final model
    print(f"Final estimate:", model.get_summary())


# Run gradient descent
gradient_descent(learning_rate=1E-9, number_of_iterations=1000)
gradient_descent(learning_rate=1E-8, number_of_iterations=200)
gradient_descent(learning_rate=5E-7, number_of_iterations=500)