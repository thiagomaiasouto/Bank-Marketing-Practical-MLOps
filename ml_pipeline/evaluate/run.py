"""
Creator: Ivanovitch Silva
Date: 30 Jan. 2022
Test the inference artifact using test dataset.
"""

import argparse
import logging
import pandas as pd
import wandb
import sys
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
import imblearn.pipeline as imb_pipe
from imblearn.under_sampling import RandomUnderSampler


# configure logging
logging.basicConfig(level=logging.INFO,
                    stream = sys.stdout,
                    format="%(asctime)s %(message)s",
                    datefmt='%d-%m-%Y %H:%M:%S')

# reference for a logging obj
logger = logging.getLogger()

def process_args(args):
    
    run = wandb.init(job_type="test")
    
    columns = ['age', 'job', 'marital', 'education', 'default', 'balance',
                'housing', 'loan', 'contact', 'day', 'month', 'duration',
                'campaign', 'pdays', 'previous', 'poutcome', 'y']

    logger.info("Downloading and reading test artifact")
    test_data_path = run.use_artifact(args.test_data).file()
    df_test = pd.read_csv(test_data_path, delimiter=',', names = columns)

    # Extract the target from the features
    logger.info("Extracting target from dataframe")
    x_test = df_test.copy()
    y_test = x_test.pop("y")

    # Encoding the target variable
    logger.info("Encoding Target Variable")
    # define a categorical encoding for target variable
    le = LabelEncoder()

    # fit and transoform y_train
    y_test = le.fit_transform(y_test)
    logger.info("Classes [0, 1]: {}".format(le.inverse_transform([0, 1])))

    ## Download inference artifact
    logger.info("Downloading and reading the exported model")
    model_export_path = run.use_artifact(args.model_export).download()

    ## Load the inference pipeline
    pipe = mlflow.sklearn.load_model(model_export_path)

    ## Predict test data
    predict = pipe.predict(x_test)

    # Evaluation Metrics
    logger.info("Evaluation metrics")

    # Metric: AUC
    auc = roc_auc_score(y_test, predict, average="macro")
    run.summary["AUC"] = auc

    # Metric: Accuracy
    acc = accuracy_score(y_test, predict)
    run.summary["Accuracy"] = acc
    
    # Metric: Balanced Accuracy
    blc_acc = balanced_accuracy_score(y_test, predict)
    run.summary["Balanced_Accuracy"] = blc_acc

    # Metric: Recall
    rcll = recall_score(y_test, predict)
    run.summary["Recall"] = rcll

    
    # Metric: Confusion Matrix
    fig_confusion_matrix, ax = plt.subplots(1,1,figsize=(7,4))
    ConfusionMatrixDisplay(confusion_matrix(predict,
                                            y_test,
                                            labels=[1,0]),
                           display_labels=["yes","no"]
                          ).plot(values_format=".0f",ax=ax)
    ax.set_xlabel("True Label")
    ax.set_ylabel("Predicted Label")
    
    # Uploading figures
    logger.info("Uploading figures")
    run.log(
        {
            "confusion_matrix": wandb.Image(fig_confusion_matrix)
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the provided model on the test artifact",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--model_export",
        type=str,
        help="Fully-qualified artifact name for the exported model to evaluate",
        required=True,
    )

    parser.add_argument(
        "--test_data",
        type=str,
        help="Fully-qualified artifact name for the test data",
        required=True,
    )

    ARGS = parser.parse_args()

    process_args(ARGS)