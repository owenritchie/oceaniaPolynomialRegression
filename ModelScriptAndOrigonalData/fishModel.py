import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import joblib
import os

# load dataset
data = pd.read_csv('/Users/owenritchie/Desktop/Websites/Fish1/fishConsumption.csv')
data = data.drop(columns=['Code'])

# drop any rows that have na's in them
data.dropna(inplace=True)

print(data.head())
print("----------------------------------")
print(data.columns)
print("----------------------------------")

# print unique values in data to see what we're working with
print(data['Entity'].unique())

# list of oceania countries and regions included in the dataset
oceania_countries = [
    'Oceania','Australia', 'New Zealand', 'Fiji', 'Solomon Islands',
    'Vanuatu', 'Samoa', 'Kiribati',
    'Micronesia (region)', 'French Polynesia', 'New Caledonia'
]

# filter to only include oceania data
oceania_data = data[data['Entity'].isin(oceania_countries)]

# verify oceania data
print(oceania_data.head())
print(oceania_data['Entity'].unique())

# create directory if doesn't exist to store models
models_directory = '/Users/owenritchie/Desktop/Websites/Fish1/models'
os.makedirs(models_directory, exist_ok=True)

# degree of polynomial regression
degree = 2

# loop through every country/region and train a model for each one
for country in oceania_countries:
    country_data = oceania_data[oceania_data['Entity'] == country]

    if not country_data.empty:
        # assign X and y as year and consumption
        X = country_data['Year'].values.reshape(-1, 1)
        y = country_data['Consumption'].values

        # train model
        polynomial_features = PolynomialFeatures(degree=degree)
        linear_regression = LinearRegression()
        model = make_pipeline(polynomial_features, linear_regression)
        model.fit(X, y)

        # save model with naming convention
        model_filename = f'{models_directory}/{country.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").lower()}_consumption_model.pkl'
        joblib.dump(model, model_filename)

        # print the filename just to verify
        print(f"Model for {country} saved as {model_filename}")
    else:
        print(f"No data available for {country}.")
