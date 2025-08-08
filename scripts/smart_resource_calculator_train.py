import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
sys.path.append(str(Path(Path(__file__).resolve()).parent.parent))
import config.environment as env


SYNTHETIC_DATA_FILE_PATH = Path(env.DATA_DIR, "smart_resource_calculator_train.csv")
MODEL_EXECUTORS = Path(env.MODEL_DIR, "model_executors.joblib")
MODEL_EXECUTOR_MEMORY = Path(env.MODEL_DIR, "model_executor_memory.joblib")
MODEL_CORES_PER_EXECUTOR = Path(env.MODEL_DIR, "model_cores_per_executor.joblib")
MODEL_DRIVER_MEMORY = Path(env.MODEL_DIR, "model_driver_memory.joblib")
ENCODER = Path(env.MODEL_DIR, "encoder.joblib")


if __name__ == '__main__':
    # Set random seed for reproducibility
    np.random.seed(42)

    # Step 1: Generate Synthetic Data
    num_samples = 1000
    table_count = np.random.randint(1, 100, num_samples)
    cluster_health = np.random.choice(['healthy', 'degraded', 'critical'], num_samples)
    data_volume = np.random.randint(10, 1000, num_samples)
    job_complexity = np.random.choice(['high', 'medium', 'low'], num_samples)
    historical_runtime = np.random.uniform(10, 300, num_samples)
    expected_runtime = np.random.uniform(10, 300, num_samples)
    executors = np.random.randint(1, 10, num_samples)
    executor_memory = np.random.randint(1, 16, num_samples)
    cores_per_executor = np.random.randint(1, 8, num_samples)
    driver_memory = np.random.randint(1, 16, num_samples)

    # Create DataFrame
    data = {
        'table_count': table_count,
        'cluster_health': cluster_health,
        'data_volume': data_volume,
        'job_complexity': job_complexity,
        'historical_runtime': historical_runtime,
        'expected_runtime': expected_runtime,
        'executors': executors,
        'executor_memory': executor_memory,
        'cores_per_executor': cores_per_executor,
        'driver_memory': driver_memory
    }
    df = pd.DataFrame(data)

    # Save the synthetic data to a CSV file
    df.to_csv(SYNTHETIC_DATA_FILE_PATH, index = False)
    print(f"Synthetic data saved to '{SYNTHETIC_DATA_FILE_PATH}'")

    # Step 2: Train a Machine Learning Model
    # Load the synthetic data
    df = pd.read_csv(SYNTHETIC_DATA_FILE_PATH)

    # One-hot encode categorical features
    encoder = OneHotEncoder(sparse_output=False)
    encoded_features = encoder.fit_transform(df[['cluster_health', 'job_complexity']])
    encoded_feature_names = encoder.get_feature_names_out(['cluster_health', 'job_complexity'])
    encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names)

    # Combine encoded features with the rest of the data
    df = df.drop(['cluster_health', 'job_complexity'], axis=1)
    df = pd.concat([df, encoded_df], axis=1)

    # Define features and targets
    X = df.drop(['executors', 'executor_memory', 'cores_per_executor', 'driver_memory'], axis=1)
    y_executors = df['executors']
    y_executor_memory = df['executor_memory']
    y_cores_per_executor = df['cores_per_executor']
    y_driver_memory = df['driver_memory']

    # Split the data into training and testing sets
    X_train, X_test, y_train_executors, y_test_executors = train_test_split(X, y_executors, test_size=0.2, random_state=42)
    _, _, y_train_executor_memory, y_test_executor_memory = train_test_split(X, y_executor_memory, test_size=0.2, random_state=42)
    _, _, y_train_cores_per_executor, y_test_cores_per_executor = train_test_split(X, y_cores_per_executor, test_size=0.2, random_state=42)
    _, _, y_train_driver_memory, y_test_driver_memory = train_test_split(X, y_driver_memory, test_size=0.2, random_state=42)

    # Train RandomForestRegressor models for each target
    model_executors = RandomForestRegressor(n_estimators=100, random_state=42)
    model_executors.fit(X_train, y_train_executors)

    model_executor_memory = RandomForestRegressor(n_estimators=100, random_state=42)
    model_executor_memory.fit(X_train, y_train_executor_memory)

    model_cores_per_executor = RandomForestRegressor(n_estimators=100, random_state=42)
    model_cores_per_executor.fit(X_train, y_train_cores_per_executor)

    model_driver_memory = RandomForestRegressor(n_estimators=100, random_state=42)
    model_driver_memory.fit(X_train, y_train_driver_memory)

    joblib.dump(model_executors, MODEL_EXECUTORS)
    joblib.dump(model_executor_memory, MODEL_EXECUTOR_MEMORY)
    joblib.dump(model_cores_per_executor, MODEL_CORES_PER_EXECUTOR)
    joblib.dump(model_driver_memory, MODEL_DRIVER_MEMORY)
    joblib.dump(encoder, ENCODER)


def predict_optimal_resources(table_count, cluster_health, data_volume, job_complexity, expected_runtime, historical_runtime = None):
    df = pd.read_csv(SYNTHETIC_DATA_FILE_PATH)

    model_executors = joblib.load(MODEL_EXECUTORS)
    model_executor_memory = joblib.load(MODEL_EXECUTOR_MEMORY)
    model_cores_per_executor = joblib.load(MODEL_CORES_PER_EXECUTOR)
    model_driver_memory = joblib.load(MODEL_DRIVER_MEMORY)
    encoder = joblib.load(ENCODER)

    # One-hot encode categorical features
    encoder = OneHotEncoder(sparse_output=False)
    encoded_features = encoder.fit_transform(df[['cluster_health', 'job_complexity']])
    encoded_feature_names = encoder.get_feature_names_out(['cluster_health', 'job_complexity'])
    encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names)
    
    # Combine encoded features with the rest of the data
    df = df.drop(['cluster_health', 'job_complexity'], axis=1)
    df = pd.concat([df, encoded_df], axis=1)
    
    # Define features and targets
    X = df.drop(['executors', 'executor_memory', 'cores_per_executor', 'driver_memory'], axis=1)
    y_executors = df['executors']
    y_executor_memory = df['executor_memory']
    y_cores_per_executor = df['cores_per_executor']
    y_driver_memory = df['driver_memory']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train_executors, y_test_executors = train_test_split(X, y_executors, test_size=0.2, random_state=42)

    if historical_runtime is None:
        historical_runtime = X_train['historical_runtime'].mean()
    
    new_job_characteristics = pd.DataFrame({
        'table_count': [table_count],
        'cluster_health': [cluster_health],
        'data_volume': [data_volume],
        'job_complexity': [job_complexity],
        'historical_runtime': [historical_runtime],
        'expected_runtime': [expected_runtime]
    })
    
    encoded_new_features = encoder.transform(new_job_characteristics[['cluster_health', 'job_complexity']])
    encoded_new_feature_names = encoder.get_feature_names_out(['cluster_health', 'job_complexity'])
    encoded_new_df = pd.DataFrame(encoded_new_features, columns=encoded_new_feature_names)
    
    new_job_characteristics = new_job_characteristics.drop(['cluster_health', 'job_complexity'], axis=1)
    new_job_characteristics = pd.concat([new_job_characteristics, encoded_new_df], axis=1)
    
    predicted_executors = round(model_executors.predict(new_job_characteristics)[0])
    predicted_executor_memory = round(model_executor_memory.predict(new_job_characteristics)[0])
    predicted_cores_per_executor = round(model_cores_per_executor.predict(new_job_characteristics)[0])
    predicted_driver_memory = round(model_driver_memory.predict(new_job_characteristics)[0])
    
    return {
        'executors': predicted_executors,
        'executor_memory': predicted_executor_memory,
        'cores_per_executor': predicted_cores_per_executor,
        'driver_memory': predicted_driver_memory
    }