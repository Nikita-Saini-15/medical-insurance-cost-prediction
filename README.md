# Medical Insurance Cost Prediction Using Machine Learning Regression Techniques

Predicts individual medical insurance charges from age, sex, BMI, number of children, smoking status and region.
Built purely in Python.

## Models
Linear, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting (tuned with GridSearchCV).
Metrics: MAE, RMSE, R-squared, 5-fold cross-validation.

## Files
- `insurance_cost_prediction.py`: full pipeline (EDA, preprocessing, models, evaluation, plots)
- `app.py`: Streamlit web app for predictions
- `insurance.csv`: dataset (Kaggle, Medical Cost Personal Dataset)
- `requirements.txt`: Python libraries needed

## Run
```
pip install -r requirements.txt
python insurance_cost_prediction.py
streamlit run app.py
```

## Author
Nikita Saini (Roll No: 2554853)
