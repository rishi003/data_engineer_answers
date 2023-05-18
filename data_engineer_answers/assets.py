import pandas as pd
import dask.dataframe as dd
import os
import joblib
import kaggle
from zipfile import ZipFile
from pathlib import Path
from dagster import MetadataValue, Output, asset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, r2_score

# Directory to load metadata, stocks and etf csv files
DATA_DIR = os.path.join("datasets", "stock-market-dataset.zip")

# Name of the metadata file
METADATA_FILE = "symbols_valid_meta.csv"


def download_data():
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        "jacksoncrow/stock-market-dataset",
        path="datasets",
    )


def load_data_from_csv(file_path: str, meta_data: pd.DataFrame) -> pd.DataFrame:
    """
    Helper Function
    Loads data from a CSV file located at the given file path and returns a 
    Pandas DataFrame object. 

    Args:
        file_path (str): The path to the CSV file to be loaded.

    Returns:
        pd.DataFrame: A Pandas DataFrame object containing the loaded CSV data 
        along with additional columns for the symbol and security name.
    """
    with ZipFile(os.path.join(DATA_DIR), "r") as zip_ref:
        symbol_data = pd.read_csv(zip_ref.open(file_path))
    symbol_data["Symbol"] = file_path.split("/")[-1].replace(".csv", "")
    symbol_data = pd.merge(symbol_data, meta_data[[
        "Symbol", "Security Name"]], left_on="Symbol", right_on="Symbol", how="left")
    return symbol_data


@asset
def load_data():
    """
    Loads all csv files from the DATA_DIR, concatenates them, and preprocesses the data.
    Returns an Output object containing the processed data and metadata.

    Args:
        None

    Returns:
        An Output object with the following attributes:
            - value: a pandas DataFrame containing the processed data
            - metadata: a dictionary containing metadata about the data, with the following keys:
                - num_records: an integer representing the number of records in the DataFrame
                - preview: a Markdown-formatted string representing a preview of the DataFrame
    """
    download_data()
    with ZipFile(os.path.join(DATA_DIR), "r") as zip_ref:
        meta_data = pd.read_csv(zip_ref.open(METADATA_FILE))
        csv_files = zip_ref.namelist()
        # Ignore metadata file
        csv_files = [file for file in csv_files if file != METADATA_FILE]
    # Process csv files in parallel
    dfs = joblib.Parallel(n_jobs=12, prefer="threads")(joblib.delayed(
        load_data_from_csv)(file_path, meta_data) for file_path in csv_files)
    all_data = pd.concat(dfs, ignore_index=True)
    return Output(value=all_data, metadata={"num_records": len(all_data), "preview": MetadataValue.md(all_data.head().to_markdown())})


@asset
def transform_data(load_data: pd.DataFrame):
    """
    Preprocesses and transforms the given pandas DataFrame by converting the "Date" column to a datetime format and 
    setting it as the index. Calculates the volume moving average and the adjusted close rolling median for each 
    "Symbol" group in the DataFrame. The preprocessed DataFrame is stored in a parquet file located in the "data" 
    directory. 

    Args:
        load_data (pd.DataFrame): A pandas DataFrame containing financial data.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the preprocessed financial data.
    """
    load_data["Date"] = pd.to_datetime(load_data["Date"], format="%Y-%m-%d")
    load_data.set_index("Date", inplace=True)

    # Calculate Volume moving average and Adj Close rolling median
    load_data["vol_moving_avg"] = load_data.groupby(
        "Symbol")["Volume"].transform(lambda x: x.rolling(window=30).mean())
    load_data["adj_close_rolling_med"] = load_data.groupby(
        "Symbol")["Adj Close"].transform(lambda x: x.rolling(window=30).median())
    # Create dir if not exists
    Path("data").mkdir(parents=True, exist_ok=True)
    load_data.to_parquet(os.path.join("data", "preprocessed_data.parquet"))
    return Output(value=load_data, metadata={"preview": MetadataValue.md(load_data.head().to_markdown())})


@asset
def preprocessed_data(transform_data: pd.DataFrame):
    """
    This function takes a pandas DataFrame as input, drops any rows with missing values, and returns the preprocessed data.

    Args:
    transform_data (pd.DataFrame): The input DataFrame to be processed.

    Returns:
    Output: An Output object containing the preprocessed data as a value and a preview of the first 5 rows as metadata.
    """
    transform_data.dropna(inplace=True)
    return Output(value=transform_data, metadata={"preview": MetadataValue.md(transform_data.head().to_markdown())})


@asset
def train_model(preprocessed_data):
    features = [
        "vol_moving_avg", "adj_close_rolling_med",
    ]
    target = "Volume"
    X = preprocessed_data[features]
    y = preprocessed_data[target]

    scalar = StandardScaler()
    X = scalar.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
    )

    model = RandomForestRegressor(
        n_estimators=100, random_state=42, verbose=True, n_jobs=-1, max_depth=5)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    analysis = dict(MAE=mae, MSE=mse)
    joblib.dump(model, os.path.join("data", "model.joblib"), compress=3)
    return Output(value=analysis, metadata={
        "Model used": "Random Forest Regressor",
        "MAE": mae,
        "MSE": mse,
        "R2": r2,
    })
