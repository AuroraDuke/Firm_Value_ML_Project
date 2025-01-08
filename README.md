# Machine Learning Model for Predicting Future Profits of BIST Companies

## Project Description

This project aims to develop a machine learning model to predict the future profits of companies listed on Borsa Istanbul (BIST). The following steps were carried out during the project:

### Project Steps

1. **Data Collection and Preparation:**
   - Financial data of companies listed on BIST was compiled.
   - Missing and inconsistent data were cleaned, and necessary preprocessing steps were applied.

2. **Feature Selection and Engineering:**
   - Financial indicators that impact company profitability were identified.
   - New features were engineered to improve the predictive power of the model.

3. **Model Development:**
   - Models were developed using various machine learning algorithms such as linear regression, decision trees, random forests, and XGBoost.
   - Hyperparameter optimization was performed for better model performance.

4. **Model Evaluation:**
   - The models were evaluated on training and test datasets.
   - Evaluation metrics included Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² Score.

---

## **Time Series Decomposition**

### **Overview**
The analysis involves decomposing a time series into three main components to better understand the underlying patterns and variations:

1. **Trend Component**: Captures the long-term upward or downward movement in the data.
2. **Seasonal Component**: Identifies repeating patterns or seasonality within the data.
3. **Residual Component**: Represents the random noise or variations that are not explained by the trend or seasonal components.

### **Visualization**
The figure below illustrates the decomposition results for the given time series data:

1. **Original Data**: The raw time series data.
2. **Trend**: The long-term trend in the data.
3. **Seasonal**: Periodic seasonal variations.
4. **Residual**: The remaining noise after removing trend and seasonal components.

![download](https://github.com/user-attachments/assets/6d9a1cf0-a3da-4c0c-bf88-2254a9f9a1e7)

### **Purpose**
This decomposition helps to:
- Understand the components of the time series.
- Prepare the data for forecasting or further analysis.


## Corraliton Matrix
![download](https://github.com/user-attachments/assets/261b8ffa-8cfb-48dd-a40a-f01e7101870f)


These relationships play a critical role in profit predictions. For example:
- **Positive Correlation:** Variables such as ROE, ROA, and Gross Profit Margin directly contribute to profit increases.
- **Negative Correlation:** Variables like Debt-to-Equity Ratio and Cost of Goods Sold Ratio negatively impact profit margins.

---

## Model Performance

The performance of the models on the test dataset is summarized below:

| **Model**             | **R² Score**  | **CV - R² Score**            | **Time (second)** |
|-----------------------|---------------|------------------------------|--------------------|
| Linear Regression     | 0.7976        | -2.5735                      | 0.1026             |
| Gaussian Regressor    | -0.0790       | -0.2045                      | 1.0654             |
| Ridge                 | 0.7782        | -1.7926                      | 0.2978             |
| LASSO                 | 0.7564        | -2.2910                      | 1.2158             |
| ElasticNet            | 0.7208        | 0.2427                       | 0.1100             |
| KNN                   | 0.6246        | 0.3773                       | 0.7537             |
| Decision Tree         | 0.8121        | 0.6800                       | 0.2857             |
| Extra Trees           | 0.8924        | 0.7200                       | 3.8681             |
| AdaBoost              | 0.8030        | 0.4357                       | 8.9596             |
| Gradient Boosting     | 0.9079        | 0.4621                       | 9.1884             |
| SGD Regressor         | 0.5033        | -44.8034                     | 0.0468             |
| Lars                  | -896278.6273  | -inf                         | 0.3328             |
| Hist Gradient Boost   | 0.7167        | 0.2437                       | 12.1309            |
| SVR-Linear            | -0.0472       | -0.1186                      | 0.4019             |
| SVR-RBF               | -0.0472       | -0.1186                      | 0.4898             |
| SVR-Poly2             | -0.0472       | -0.1186                      | 0.3745             |
| SVR-Poly3             | -0.0472       | -0.1186                      | 0.3735             |
| ANN-lbfgs             | 0.9250        | -0.5725                      | 2.1858             |
| ANN-sgd               | -             | nan                          | 0.4324             |
| ANN-adam              | -0.0791       | -0.2049                      | 3.7542             |
| XGBoost               | 0.8036        | 0.6617                       | 8.3492             |
| LightGBM              | 0.7102        | 0.2904                       | 1.3978             |
| CatBoost              | 0.7362        | 0.6746                       | 63.5908            |



## **Key Insights**

### **1. ANN-lbfgs Model**
- **R² Score:** 0.9250  
- **Test Set Metrics:**  
  - **R²:** 0.9249901348672693  
  - **MSE (Mean Squared Error):** 4.7224e+18  
  - **MAE (Mean Absolute Error):** 856,578,944.15  
- ANN-lbfgs emerged as the most effective algorithm for prediction accuracy among all the tested models.  
- It excelled at capturing complex patterns in the data, making it the top-performing model.

### **2. Limitations of Linear Models**
- **Linear Regression:**  
  - **CV - R² Score:** -2.5735  
  - Struggled with generalization, indicating its inability to capture non-linear relationships effectively.  

### **3. Decision Tree and Gradient Boosting Models**
- **Decision Tree:**  
  - **R² Score:** 0.8121  
- **Gradient Boosting:**  
  - **R² Score:** 0.9079  
- Both models delivered strong performance, establishing themselves as reliable alternatives for prediction tasks.

### **4. Execution Time Considerations**
- **CatBoost:**  
  - **R² Score:** 0.7362  
  - **Execution Time:** 63.59 seconds  
  - While achieving a reasonable R² score, its significantly longer execution time makes it less suitable for time-sensitive applications.  
- **Linear Regression and SGD Regressor:**  
  - Both models executed much faster, making them viable options for real-time scenarios despite their lower accuracy.

### **5. Feature Importance**
- Variables such as **ROE** and **ROA** were critical in enhancing prediction accuracy, especially for models adept at capturing non-linear relationships.

---

## Conclusions and Future Work

- **Profit Prediction Performance:** The model provided reliable results in predicting future profits of BIST companies.
- **Critical Variables:** Key financial indicators influencing profit predictions were identified and analyzed.
- **Future Work:** To further enhance the model, the following steps are planned:
  - Incorporating a larger dataset.
  - Utilizing deep learning algorithms.
  - Integrating macroeconomic data into the model.

---

## Contact

For inquiries and suggestions, feel free to connect with me on [LinkedIn](https://www.linkedin.com/in/ilker-yayalar-1b6ba4253/).
