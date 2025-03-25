# Car Selling Price Prediction

This project predicts the selling price of cars using a Random Forest Regressor. The model is trained on a dataset containing features such as year, mileage, engine size, fuel type, and more. A Streamlit app has been developed for users to input car details and get predicted selling prices.

- Trained on a dataset with features like year, mileage, engine size, fuel type, etc.
- Achieved **96.28% train accuracy** and **94.72% test accuracy**.
- Built a user-friendly Streamlit app for real-time predictions.
- Deployed the trained model using Python's `joblib`.

  ## Dataset
The dataset contains information about cars, including:
- **Numerical features**: Year, km driven, mileage, engine size, max power, age.
- **Binary Columns**: fuel type, transmission, ownership type
- **Categorical features**: Make, model.
- **Target variable**: Selling price.

Sample data:
| selling_price | year | km_driven | mileage | engine | max_power | age | make   | model                  | Individual | Trustmark Dealer | Diesel | Electric | LPG | Petrol | Manual | 5 | >5 |
|---------------|------|-----------|---------|--------|-----------|-----|--------|------------------------|------------|------------------|--------|----------|-----|--------|--------|---|----|
| 1.2           | 2012 | 120000    | 19.7    | 796    | 46.3      | 11  | MARUTI | ALTO STD              | 1          | 0                | 0      | 0        | 0   | 1      | 1      | 1 | 0  |
| 5.5           | 2016 | 20000     | 18.9    | 1197   | 82        | 7   | HYUNDAI| GRAND I10 ASTA         | 1          | 0                | 0      | 0        | 0   | 1      | 1      | 1 | 0  |

*Note: The dataset is preprocessed to encode categorical variables.*

## Model Performance
- **Train Accuracy**: 96.28%
- **Test Accuracy**: 94.72%
- **Model Used**: Random Forest Regressor


