import pandas

# Open our dataset from file
dataset = pandas.read_csv("datasets/titanic.csv", index_col=False, sep=",", header=0)

# Fill missing cabin information with 'Unknown'
dataset["Cabin"].fillna("Unknown", inplace=True)

# Remove rows missing Age information
dataset.dropna(subset=["Age"], inplace=True)

# Remove the Name, PassengerId, and Ticket fields
# This is optional and only to make it easier to read our print-outs
dataset.drop(["PassengerId", "Name", "Ticket"], axis=1, inplace=True)

print(dataset.head())

from m0c_logistic_regression import train_logistic_regression

features = ["Age", "Pclass", "SibSp", "Parch", "Fare"]
loss_numerical_only = train_logistic_regression(dataset, features)

print(f"Numerical-Only, Log-Loss (cost): {loss_numerical_only}")

# Swap male / female with numerical values
# We can do this because there are only two categories
dataset["IsFemale"] = dataset.Sex.replace({'male':0, 'female':1})

# Print out the first few rows of the dataset
print(dataset.head())

# Run and test the model, also using IsFemale this time
features = ["Age", "Pclass", "SibSp", "Parch", "Fare", "IsFemale"]
loss_binary_categoricals = train_logistic_regression(dataset, features)

print(f"\nNumerical + Sex, Log-Loss (cost): {loss_binary_categoricals}")

# Get all possible categories for the "PClass" column
print(f"Possible values for PClass: {dataset['Pclass'].unique()}")

# Use Pandas to One-Hot encode the PClass category
dataset_with_one_hot = pandas.get_dummies(dataset, columns=["Pclass"], drop_first=False)

# Add back in the old Pclass column, for learning purposes
dataset_with_one_hot["Pclass"] = dataset.Pclass

# Print out the first few rows
print(dataset_with_one_hot.head())

# Run and test the model, also using Pclass as a categorical feature this time
features = ["Age", "SibSp", "Parch", "Fare", "IsFemale",
            "Pclass_1", "Pclass_2", "Pclass_3"]

loss_pclass_categorical = train_logistic_regression(dataset_with_one_hot, features)

print(f"\nNumerical, Sex, Categorical Pclass, Log-Loss (cost): {loss_pclass_categorical}")


# Use Pandas to One-Hot encode the Cabin and Pclass categories
dataset_with_one_hot = pandas.get_dummies(dataset, columns=["Pclass", "Cabin"], drop_first=False)

# Find cabin column names
cabin_column_names = list(c for c in dataset_with_one_hot.columns if c.startswith("Cabin_"))

# Print out how many cabins there were
print(len(cabin_column_names), "cabins found")

# Make a list of features
features = ["Age", "SibSp", "Parch", "Fare", "IsFemale",
            "Pclass_1", "Pclass_2", "Pclass_3"] + \
            cabin_column_names

# Run the model and print the result
loss_cabin_categorical = train_logistic_regression(dataset_with_one_hot, features)

print(f"\nNumerical, Sex, Categorical Pclass, Cabin, Log-Loss (cost): {loss_cabin_categorical}")

# We have cabin names, like A31, G45. The letter refers to the deck that
# the cabin was on. Extract just the deck and save it to a column.
dataset["Deck"] = [c[0] for c in dataset.Cabin]

print("Decks: ", sorted(dataset.Deck.unique()))

# Create one-hot vectors for:
# Pclass - the class of ticket. (This could be treated as ordinal or categorical)
# Deck - the deck that the cabin was on
dataset_with_one_hot = pandas.get_dummies(dataset, columns=["Pclass", "Deck"], drop_first=False)

# Find the deck names
deck_of_cabin_column_names = list(c for c in dataset_with_one_hot.columns if c.startswith("Deck_"))

features = ["Age", "IsFemale", "SibSp", "Parch", "Fare",
            "Pclass_1", "Pclass_2", "Pclass_3",
            "Deck_A", "Deck_B", "Deck_C", "Deck_D",
            "Deck_E", "Deck_F", "Deck_G", "Deck_U", "Deck_T"]

loss_deck = train_logistic_regression(dataset_with_one_hot, features)

print(f"\nSimplifying Cabin Into Deck, Log-Loss (cost): {loss_deck}")

# Use a dataframe to create a comparison table of metrics
# Copy metrics from previous Unit

l =[["Numeric Features Only", loss_numerical_only],
    ["Adding Sex as Binary", loss_binary_categoricals],
    ["Treating Pclass as Categorical", loss_pclass_categorical],
    ["Using Cabin as Categorical", loss_cabin_categorical],
    ["Using Deck rather than Cabin", loss_deck]]

pandas.DataFrame(l, columns=["Dataset", "Log-Loss (Low is better)"])