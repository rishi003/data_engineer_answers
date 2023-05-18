# End-to-End Data Engineering Task

`Objective`: The task is to create a data processing pipeline for time series analysis of stocks dataset which can be downloaded from [here](https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset). The data set needs to be downloaded, processed and then ingected into a machine-learing model the output of which is saved and loaded in a simple flask app to be served as an API.

The task is broken down in following steps:

- Download the dataset from kaggle.
- Preprocess and transform data into following format:

```
Symbol: string
Security Name: string
Date: string (YYYY-MM-DD)
Open: float
High: float
Low: float
Close: float
Adj Close: float
Volume: A suitable Number type (int or float)
```

- The above transformed dataset is then passed to a feature engineering step which calculates 30 days moving average of trading volume (`Volume`) and 30 day rolling median for adjusted close(`Adj Close`) per each stock and ETF and saving the results in `vol_moving_avg` and `adj_close_rolling_med` respectively.
- This data is saved in a parquet file and the resultant data is passd to the model training step, which after a successful train, stores the model metrics in a file. The model is also persisted in the disk.
- Lastly the above model is loaded in a flask application which serves the prediction API at `/predict?vol_moving_avg=<YOUR_VOL_MOVING_AVG>&adj_close_rolling_med=<YOUR_ADJ_ROLLING_MED>`. This API returns the predicted volume based on the provided query parameters.

## About the Stack 🛒

- Programming Language: Python
- Deployment:
  - Local: Docker Compose
  - Server: GCP Docker service
- IDE: VS Code

## Project Structure 📂

```
data_engineer_answers
| data_engineer_answers (Dagster data pipeline files)
| server (Flask Server)
|   |___Dockerfile (Docker file to build server)
|   |___main.py (Flask App)
| .env (Environment Variables)
| dockerize-dagster
|   |___Dockerfile (Docker file to build dagster server)
| README.md
| .dockerignore
| workspace.yamml (Dagster Configurations)
| setup.py
| dagster.yaml (Dagster Configurations for produciton)
| docker-compose.yaml (Orchestrate the Dagster and Server project)
```

## Project challenges ⛓

- The first challenge was processing all the individual stock and ETF csv downloaded from kaggle in the first step of pipeline.<br/>
  **Resolution**:
  - Using `joblib` python library to parallelize the above task which effectively reduced processing time by 70%.
- Training the ML model and perfroming a Baysian Grid Search CV requred huge computation resources as the resulting dataset comprised of more than 28 million rows.<br/>
  **Resolution:**
  - Exporting the preprocessed data from the pipeline to a Parquet file and uploading the file in Google Drive. - After uploading the dataset, I created a Colab notebook (it can be found [here](https://colab.research.google.com/drive/1kF3kjiSdy_5qJ3j0MAW_Tv1tLRneFGqo?usp=sharing)) to mount the GDrive and load the dataset there for analysing R^2, MSE and MAE of different models. - Performed Baysian Search CV to find the best combination of hyperparameters for modle training and implemented those parameters here.
  - The Best hyperparameters for RandomForestRegressor I obtained from there are as follows:
    - n_estimators: 100
    - max_depth: 5

## Pipeline Structure 🏗

The pipeline is executed in 4 steps as described below:

1. <u>Load Data</u>: Downloads and merges the data from kaggle using 12 threads for efficient performance.
2. <u>Transform Data</u>: Calculates the Moving average and rolling median for Volume and Adjusted Close respectively.
3. <u>Preprocess Data</u>: Performs cleaning operations which are limited to dropping null rows at the moment.
4. <u>Train Model</u>: Trains the Machine learing model.

<img src="https://i.ibb.co/30DkM81/etl-pipeline.png" 
        alt="Picture" 
        width="350" 
        height="500" 
        style="display: block; margin: 0 auto" />

## Pipeline Testing 🧪

The tests for the pipeline are located under the `data_engineer_answers_test` folder. These tests import the assets taht are defined in the `data_engineer_answers.assets` module and materialize them, the end result is then tested for the number of rows returned and the columns in the resulting preprocessed data.

command to run tests:

```shell
pytest data_engineer_answers_tests
```

## Model hosting 🏁

The Server that is hosting is a containerized flask app and has been deployed to GCP using Cloud Run Serverless Container service.

[Link for the hosted model](https://stock-analysis-web-npsjpnvppa-uc.a.run.app/predict?vol_moving_avg=12345&adj_close_rolling_med=26)

## References 🔗

- Google
- Stackoverflow
- Medium
- AI tools
- Geeks For Geeks

AI Tools output (Chat-GPT)

```
Working with a large dataset and limited memory can be challenging, but there are several strategies you can employ to perform efficient Bayesian Grid Search CV with your machine's memory constraints. Here are some suggestions:

1. **Reduce memory usage**: Since you have limited memory, it's essential to optimize your DataFrame's memory usage. You can achieve this by doing the following:
   - Use appropriate data types: Choose the most memory-efficient data types for your columns. For example, if you have categorical features, consider converting them to the categorical data type.
   - Load data in chunks: If possible, load the data in smaller chunks instead of loading the entire dataset into memory at once. This can be done using the `chunksize` parameter in `pd.read_csv()` or by using libraries like Dask, which allow for lazy loading and computation on larger-than-memory datasets.

2. **Use incremental learning**: Instead of training the model on the entire dataset in one go, you can utilize incremental learning techniques. These methods allow you to update the model parameters incrementally as you process data in smaller batches or chunks. This way, you can avoid loading the entire dataset into memory simultaneously.

3. **Reduce the parameter space**: Bayesian Grid Search CV typically involves exploring a large parameter space, which may not be feasible given your memory constraints. Consider reducing the parameter combinations to search through. You can do this by selecting a subset of parameters to tune or by using coarse-grained parameter grids.

4. **Parallelize the search**: Utilize parallel processing to distribute the computations across multiple cores or machines. Libraries like scikit-learn provide options for parallel execution during grid search. This can help speed up the process and make it more memory-efficient.

5. **Consider dimensionality reduction**: If the dataset has a high number of features, you might consider applying dimensionality reduction techniques like Principal Component Analysis (PCA) or feature selection methods to reduce the number of features. This can help in reducing memory usage and computational requirements.

6. **Use external storage**: If the dataset is too large to fit into memory, you can consider using external storage solutions like databases or distributed file systems. This way, you can load and process subsets of the data at a time, reducing the memory requirements.

Remember to monitor memory usage during the Bayesian Grid Search CV process and adjust the strategies accordingly. It might also be worth considering using cloud-based solutions with more computational resources if they are available to you.
```

```python
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from skopt import BayesSearchCV

# Load the dataset in chunks
chunksize = 1000000  # Adjust the chunk size based on your memory constraints
data_chunks = pd.read_csv('your_dataset.csv', chunksize=chunksize)

# Define the parameter grid for grid search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 10],
    # Add other parameters to tune
}

# Create an empty DataFrame to store results
results = pd.DataFrame(columns=['params', 'mean_score'])

# Iterate over the data chunks
for chunk in data_chunks:
    # Preprocess the data
    X = chunk.drop('target_column', axis=1)
    y = chunk['target_column']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Define the model pipeline
    pipeline = Pipeline([
        ('rf', RandomForestRegressor())
        # Add other preprocessing steps or models if needed
    ])

    # Perform Bayesian Grid Search CV
    grid_search = BayesSearchCV(
        estimator=pipeline,
        search_spaces=param_grid,
        scoring='neg_mean_squared_error',
        cv=5,
        n_jobs=-1  # Utilize all available cores for parallel processing
    )

    # Fit the grid search on the training data
    grid_search.fit(X_train_scaled, y_train)

    # Get the best parameters and validation score
    best_params = grid_search.best_params_
    val_score = grid_search.best_score_

    # Append the results to the DataFrame
    results = results.append({'params': best_params, 'mean_score': val_score}, ignore_index=True)

# Find the best set of parameters with the highest mean validation score
best_params = results.loc[results['mean_score'].idxmax()]['params']
print("Best Parameters:", best_params)

# Train the model with the best parameters on the entire dataset
model = pipeline.set_params(**best_params)
model.fit(X_scaled, y)  # Use the full dataset or a representative subset for training

# Evaluate the model on a separate test set or using cross-validation
test_score = cross_val_score(model, X_test_scaled, y_test, scoring='neg_mean_squared_error', cv=5)
print("Test Score:", test_score.mean())

```
