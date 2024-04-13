# views.py (Django)
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rest_framework.views import APIView
import pandas as pd
from rest_framework.parsers import MultiPartParser
from sklearn.linear_model import LinearRegression
from rest_framework.response import Response
import numpy as np


class SalesPredictionAPIView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request, format=None):
        file_obj = request.data['file']
        data = pd.read_csv(file_obj)

        # Selecting only the required columns
        filtered_data = data[['year', 'Sales', 'competitor sales']]

        # Extracting years, sales, and competitor sales from the filtered data
        years = filtered_data['year'].values.reshape(-1, 1)
        sales = filtered_data['Sales'].values
        competitor_sales = filtered_data['competitor sales'].values

        # Create and fit the linear regression model
        model = LinearRegression()
        model.fit(years, sales)

        # Predict sales for the next 5 years (2025 to 2029)
        future_years = np.array([2025, 2026, 2027, 2028, 2029]).reshape(-1, 1)  # Years to predict
        predicted_sales = model.predict(future_years)

        # Prepare the response data
        response_data = {
            'predicted_sales': predicted_sales.tolist(),
            'years': future_years.flatten().tolist(),
            'competitor_sales': competitor_sales[:5].tolist()
        }

        # Generate the plot
        plt.figure(figsize=(10, 6))
        plt.scatter(years, sales, color='blue', label='Your Sales')
        plt.plot(years, model.predict(years), color='red', linestyle='-', label='Your Linear Regression')
        plt.scatter(future_years, predicted_sales, color='green', label='Predicted Sales')
        plt.scatter(years, competitor_sales, color='orange', label='Competitor Sales')
        plt.xlabel('Year')
        plt.ylabel('Sales')
        plt.title('Sales Prediction and Competitor Comparison')
        plt.legend()
        plt.grid(True)

        # Save the plot to a buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)

        # Encode the plot image as base64
        plot_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        response_data['plot_image_base64'] = plot_base64

        return Response(response_data)
