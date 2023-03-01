import argparse

import numpy as np
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run = Run.get_context()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    heart_dataset = TabularDatasetFactory.from_delimited_files(
        "https://raw.githubusercontent.com/t0m0ffel/udacity-capstone/master/starter_file/heart.csv"
    )

    X = heart_dataset.to_pandas_dataframe()
    y = X.pop('output')

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    run.log("Accuracy", np.float(accuracy))

if __name__ == '__main__':
    main()