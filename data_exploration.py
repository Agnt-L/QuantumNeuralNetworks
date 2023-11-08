import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


def explore_data():
    # Load the Iris dataset
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target  # Add the 'target' column

    # Display the first few rows of the dataset
    print("First few rows of the dataset:")
    print(df.head())

    # Get information about the dataset
    print("\nDataset information:")
    print(df.info())

    # Summary statistics of the dataset
    print("\nSummary statistics:")
    print(df.describe())

    # Number of data points and features
    num_samples, num_features = df.shape
    print(f"\nNumber of data points: {num_samples}")
    print(f"Number of features: {num_features}")

    # Class distribution
    class_distribution = df['target'].value_counts()
    print("\nClass distribution:")
    print(class_distribution)

    # Data distribution visualization
    plt.figure(figsize=(12, 6))
    for i, feature in enumerate(df.columns[:-1]):
        plt.subplot(2, 2, i + 1)
        plt.scatter(df[feature], df['target'], label='Class')
        plt.xlabel(feature)
        plt.ylabel('Class')
        plt.legend()
    plt.show()


explore_data()
