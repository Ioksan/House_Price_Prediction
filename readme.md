# UK House Price Prediction

Machine learning project that predicts UK house prices using real transaction data from 2015 to 2024.

## Dataset

- 90,000 real UK house sale transactions
- Source: HM Land Registry via Kaggle
- Features: property type, location, date, postcode area

## Models Trained

| Model | Cross-Validation RMSE |
|-------|----------------------|
| XGBoost | £130,170 |
| Linear Regression | £131,073 |
| Gradient Boosting | £136,333 |
| Random Forest | £141,884 |

Best model: XGBoost, selected after 5-fold cross-validation.

## Tech Stack

- Python 3.14
- Pandas, NumPy
- Scikit-learn, XGBoost
- Matplotlib, Seaborn
- Streamlit
- Jupyter Notebook

## Project Structure

    AI_House_Price_Prediction/
    ├── data/               - dataset (not included in repository)
    ├── models/             - trained model files
    ├── notebooks/          - Jupyter notebook and Streamlit app
    ├── outputs/charts/     - generated visualisations
    ├── src/                - training script
    └── requirements.txt    - project dependencies

## How to Run

    python -m venv priceenv
    priceenv\Scripts\activate
    pip install -r requirements.txt
    python src/train_model.py
    cd notebooks
    streamlit run app.py

## Results

Outlier removal reduced RMSE from £580,262 to £130,170, an improvement of 77%. The final model was evaluated using 5-fold cross-validation to ensure stability across different data splits.