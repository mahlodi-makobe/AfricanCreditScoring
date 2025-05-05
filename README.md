# African Credit Scoring Challenge

📌 **Overview**

This project develops a machine learning solution to predict loan defaults in African financial markets (Kenya and Ghana) for the Zindi African Credit Scoring Challenge. By integrating loan application data with macroeconomic indicators, the model achieves a public F1 score of 0.5712 (top 30% on Zindi), delivering a credit scoring framework (Low/Medium/High risk) for financial risk assessment.

🛠️ **Technical Approach**

🔍 **Key Challenges Addressed**

- Severe class imbalance: 1.64% default rate in training data.
- Temporal data shifts: Training (2022) vs. test (2023) data.
- Cross-country generalization: Adapting from Kenya to Ghana’s market.
- Economic volatility: Ghana’s high inflation (31-38% in 2022-2023).

🧩 **Feature Engineering Highlights**

| Feature Type         | Key Transformations                      | Business Impact                                  |
|----------------------|------------------------------------------|--------------------------------------------------|
| Loan Characteristics | Log-transformed loan amount, duration bins | Captured non-linear risk patterns                |
| Borrower Behavior    | Default history, loan frequency          | Identified high-risk borrowing patterns          |
| Economic Indicators  | `inflation_loan_impact`, `ghana_high_inflation` | Quantified macroeconomic stress                  |


🤖 **Modeling**

The final model is an XGBoost classifier optimized for the F1 metric:
```python
XGBClassifier(
    scale_pos_weight=60,      # Addresses class imbalance
    max_depth=5,              # Controls overfitting
    eval_metric='f1',         # Aligns with competition
    early_stopping_rounds=50  # Prevents overtraining
)

- **Threshold**: 0.45, balancing precision (54%) and recall (86%).
- **Validation F1**: 0.678.
- **Public F1**: 0.5712.

📂 **Repository Structure**

The repository contains the following directories and files:

- **Datasets/**:
    - `final_train.csv`: Merged training data.
    - `final_test.csv`: Merged test data.
- **Notebooks/**:
    - `AfricanScoringMerge.ipynb`: Jupyter Notebook for data merging.
    - `AfrcanScoringEDA.ipynb`: Jupyter Notebook for exploratory data analysis.
    - `AfricanScoringModel.ipynb`: Jupyter Notebook for feature engineering and modeling.
- `submission.csv`: Final predictions in CSV format.
- `requirements.txt`: List of Python dependencies.
- `README.md`: This file.

