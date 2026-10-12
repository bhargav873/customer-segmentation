# Customer Segmentation Using K-Means Clustering

## Project Overview

This project uses Machine Learning to segment mall customers into different groups based on their annual income and spending scores. The K-Means clustering algorithm identifies customer groups with similar characteristics.

The results can help businesses understand customer behavior and develop targeted marketing strategies.

## Objectives

* Analyze customer demographic and spending data.
* Explore relationships between annual income and spending scores.
* Apply K-Means clustering to segment customers.
* Visualize customer segments and cluster centers.
* Generate business insights and marketing recommendations.
* Save the trained model and scaler for future predictions.

## Dataset

**Dataset:** `Mall_Customers.csv`

The dataset contains 200 customer records with the following columns:

| Column                 | Description                              |
| ---------------------- | ---------------------------------------- |
| CustomerID             | Unique customer identifier               |
| Gender                 | Customer gender                          |
| Age                    | Customer age                             |
| Annual Income (k$)     | Annual income in thousands of dollars    |
| Spending Score (1-100) | Spending score assigned to each customer |

The `CustomerID` column was removed before clustering because it does not represent a meaningful customer characteristic for this analysis.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib

## Project Workflow

1. **Data Loading and Exploration:** Loaded the dataset and examined its structure, data types, missing values, duplicates, and descriptive statistics.
2. **Data Cleaning:** Removed duplicate records and dropped the customer ID column.
3. **Exploratory Data Analysis (EDA):** Examined customer demographics, income distribution, and spending patterns using visualizations.
4. **Feature Selection:** Selected annual income and spending score as the primary clustering features.
5. **Feature Scaling:** Applied StandardScaler to standardize the selected features.
6. **Optimal Cluster Selection:** Used the Elbow Method to compare within-cluster sum of squares (WCSS) across different cluster counts.
7. **Model Training:** Trained a K-Means clustering model with five clusters.
8. **Visualization:** Plotted the customer segments and cluster centers.
9. **Cluster Analysis:** Compared the average age, income, spending score, and customer count for each cluster.
10. **Business Insights:** Developed potential marketing recommendations based on the identified customer groups.
11. **Model Saving:** Saved the trained K-Means model and scaler using Joblib.
12. **Prediction:** Loaded the saved model and predicted the segment for a new customer.
13. **Evaluation:** Calculated the Silhouette Score to assess cluster separation.
14. **Export:** Saved the customer records with their assigned cluster labels to a CSV file.

## Customer Segments

The model identifies five customer groups:

| Cluster   | Customer Characteristics              | Potential Marketing Strategy                     |
| --------- | ------------------------------------- | ------------------------------------------------ |
| Cluster 0 | Moderate income and moderate spending | Targeted promotions and personalized campaigns   |
| Cluster 1 | High income and high spending         | Loyalty programs and premium product offers      |
| Cluster 2 | Lower income and high spending        | Popular products and suitable promotional offers |
| Cluster 3 | High income and low spending          | Personalized offers to encourage engagement      |
| Cluster 4 | Lower income and low spending         | Budget-friendly offers and discounts             |

These groups are based on the observed averages of the selected features. Marketing recommendations are suggestions that would need to be tested against actual business results.

## Model Evaluation

* **Algorithm:** K-Means Clustering
* **Number of Clusters:** 5
* **Silhouette Score:** 0.5547

The Silhouette Score measures how well-separated the clusters are. A score of 0.5547 suggests a reasonably clear clustering structure, although it does not guarantee that five clusters are the best choice for every business use case.

## Project Files

```text
customer segmentation/
├── main.py
├── Mall_Customers.csv
├── customer_segments.csv
├── requirements.txt
├── README.md
└── saved_model/
    ├── kmeans_model.pkl
    ├── scaler.pkl
    └── cluster_summary.csv
```

* `main.py`: Contains the data analysis, model training, visualization, evaluation, and prediction code.
* `Mall_Customers.csv`: Original dataset.
* `customer_segments.csv`: Customer records with assigned cluster labels.
* `requirements.txt`: Required Python packages.
* `saved_model/kmeans_model.pkl`: Trained K-Means model.
* `saved_model/scaler.pkl`: Fitted feature scaler.
* `saved_model/cluster_summary.csv`: Summary statistics for the customer segments.

## Installation and Usage

### 1. Clone the Repository

```bash
git clone <your-github-repository-url>
```

### 2. Navigate to the Project Folder

```bash
cd customer-segmentation
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Project

```bash
python main.py
```

The script loads the dataset, performs the analysis, trains the clustering model, generates visualizations, evaluates the model, and saves the resulting files.

## Future Improvements

* Compare different clustering algorithms.
* Evaluate alternative numbers of clusters.
* Build an interactive dashboard to explore customer segments.
* Test whether the segments lead to measurable improvements in marketing outcomes.

## Conclusion

This project demonstrates the application of unsupervised Machine Learning to customer segmentation. By combining exploratory data analysis, feature scaling, K-Means clustering, visualization, and cluster evaluation, it identifies groups of customers with different income and spending patterns.
