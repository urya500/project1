import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


def simpleProcessing(datasetLocation, categories, userX):
    dataset = pd.read_csv(datasetLocation)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # Convert single index to a list if necessary
    if isinstance(categories, int):
        categories = [categories]

    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), categories)], remainder='passthrough')
    X = np.array(ct.fit_transform(X))
    userX_transformed = np.array(ct.transform([userX]))

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)
    np.set_printoptions(precision=2)
    print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

    userY_pred = regressor.predict(userX_transformed)
    np.set_printoptions(precision=2)
    print("---")
    print(userY_pred)

#simpleProcessing("uploads/4.csv", 3, [-3200, -32540, -3420, "New York"])
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder, StandardScaler

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer


def simpleProcessingtest(datasetLocation, userX):
    # Load the dataset
    dataset = pd.read_csv(datasetLocation)
    print(dataset.columns)
    # Check for missing values and handle them
    if dataset.isnull().values.any():
        dataset = dataset.dropna()

    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # Identify numerical columns
    numerical_features = list(range(X.shape[1]))

    # Handle scaling (no categorical columns in this dataset)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features)
        ]
    )

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Creating a pipeline for preprocessing and regression
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    # Fit the model
    pipeline.fit(X_train, y_train)

    # Convert userX to DataFrame to match dataset structure
    userX_df = pd.DataFrame([userX], columns=dataset.columns[:-1])
    userX_transformed = pipeline.named_steps['preprocessor'].transform(userX_df.values)

    # Predict the user input
    userY_pred = pipeline.named_steps['regressor'].predict(userX_transformed)

    # Return the first (and only) prediction
    return float(userY_pred[0])


# Example usage:
# Adjust the input array to match the number of columns minus the target column
#simpleProcessingtest("uploads/train.csv", [24,21.5494519624])

def getdata(datasetLocation):
    # Load the dataset
    dataset = pd.read_csv(datasetLocation)

    x = dataset.columns[0:len(dataset.columns)-1]
    return x.tolist()

print(getdata("uploads/train.csv"))
