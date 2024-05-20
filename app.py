import matplotlib
matplotlib.use('Agg')
from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

import sys
path = '/home/owenritchie1/mysite'
if path not in sys.path:
   sys.path.insert(0, path)


# initialize flask app
app = Flask(__name__)

# directory of models
models_directory = os.path.join(os.getcwd(), 'models')

# load dataset
data_path = os.path.join(os.getcwd(), 'data', 'fishConsumption.csv')
print(f"Loading dataset from {data_path}")
data = pd.read_csv(data_path)

# drop code col
data = data.drop(columns=['Code'])
print("Dropped 'Code' column")

# list of regions to include
oceania_countries = [
    'Oceania', 'Australia', 'New Zealand', 'Fiji', 'Solomon Islands',
    'Vanuatu', 'Samoa', 'Kiribati',
    'Micronesia (region)', 'French Polynesia', 'New Caledonia'
]

# filter data for just oceania countries
oceania_data = data[data['Entity'].isin(oceania_countries)]
print(f"Filtered data for Oceania countries: {oceania_countries}")

# function to load model & predict
def load_model_and_predict(country, year):
    try:
        model_filename = os.path.join(models_directory, f"{country.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').lower()}_consumption_model.pkl")
        print(f"Loading model from {model_filename}")
        model = joblib.load(model_filename)
        predicted_consumption = model.predict(np.array([[year]]))[0]
        predicted_consumption = round(predicted_consumption, 3)
        print(f"Prediction for {country} in {year}: {predicted_consumption}")
        return predicted_consumption
    except Exception as e:
        print(f"Error loading model for {country}: {e}")
        return None

# route for home page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            country = request.form['country']
            year = int(request.form['year'])
            print(f"Received prediction request for {country} in {year}")
            prediction = load_model_and_predict(country, year)

            if prediction is None:
                print(f"Error making prediction for {country} in {year}")
                return render_template('index.html', countries=oceania_countries, error=True)

            # Generate plot
            plot_url, lower_bound, upper_bound = generate_plot(country, year, prediction)
            print(f"Generated plot for {country} in {year}")

            return render_template('index.html', countries=oceania_countries, prediction=prediction, plot_url=plot_url, country=country, year=year, lower_bound=lower_bound, upper_bound=upper_bound)
        except Exception as e:
            print(f"Error in prediction route: {e}")
            return render_template('index.html', countries=oceania_countries, error=True)
    else:
        print("Rendering index page")
        return render_template('index.html', countries=oceania_countries)



# function to generate plot
def generate_plot(country, year, prediction):
    try:
        country_data = oceania_data[oceania_data['Entity'] == country]
        X = country_data['Year'].values.reshape(-1, 1)
        y = country_data['Consumption'].values

        # polynomial regression
        degree = 2
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(X, y)

        # extend the years for plotting
        year_range = np.arange(X.min(), year + 1).reshape(-1, 1)
        predictions_extended = model.predict(year_range)

        # find the residuals
        residuals = y - model.predict(X)
        residual_std = np.std(residuals)

        # find confidence intervals
        lower_bound = predictions_extended - 1.96 * residual_std
        upper_bound = predictions_extended + 1.96 * residual_std

        lower_bound_point = lower_bound[-1]
        upper_bound_point = upper_bound[-1]

        plt.rcParams.update({'font.size': 14, 'font.family': 'Arial'})

        # plot info
        plt.figure(figsize=(14, 10), dpi=120)
        plt.gca().set_facecolor('#2f3640')
        sns.scatterplot(x='Year', y='Consumption', data=country_data, label='Historical Data', color='#ffffff')
        sns.lineplot(x=year_range.flatten(), y=predictions_extended, color='#0a3d62', label='Model Prediction')
        plt.fill_between(year_range.flatten(), lower_bound, upper_bound, color='#3c6382', alpha=0.3, label='95% Confidence Interval')
        plt.scatter([year], [prediction], color='#26de81', marker='x', s=200, label=f'Prediction for {year}')
        plt.xlabel('Year', color='#ffffff', fontsize=18)
        plt.ylabel('Fish and Seafood Consumption (kg per capita)', color='#ffffff', fontsize=18)
        plt.title(f'Fish and Seafood Consumption in {country} Over Time', color='#ffffff', fontsize=21)
        plt.legend(facecolor='#2f3640', framealpha=0.3, fontsize='large', loc='upper left')
        plt.grid(True, color='#7f8c8d')
        plt.xticks(color='#ffffff', fontsize=14)
        plt.yticks(color='#ffffff', fontsize=14)

        plt.setp(plt.gca().get_legend().get_texts(), color='#ffffff', fontsize=14)  # set legend to white

        img = BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight', facecolor='#2f3640')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        print(f"Generated plot URL for {country} in {year}")
        return 'data:image/png;base64,{}'.format(plot_url), lower_bound_point, upper_bound_point
    except Exception as e:
        print(f"Error generating plot for {country}: {e}")
        return None, None, None

# run the app
if __name__ == '__main__':
    print("Starting Flask app")
    app.run(debug=True, host='127.0.0.1', port=3000)
