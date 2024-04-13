# views.py (Django)
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

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
            'competitor_sales': competitor_sales.tolist()
        }

        return Response(response_data)
