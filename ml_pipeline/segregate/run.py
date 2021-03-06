import argparse
import logging
import os
import sys
import tempfile
import pandas as pd
import wandb
from sklearn.model_selection import train_test_split

# configure logging
logging.basicConfig(level=logging.INFO,
                    stream= sys.stdout,
                    format="%(asctime)s %(message)s",
                    datefmt='%d-%m-%Y %H:%M:%S')

# reference for a logging obj
logger = logging.getLogger()

def process_args(args):
    """
    Arguments
        args - command line arguments
        args.input_artifact: Fully qualified name for the artifact
        args.artifact_root:  Name for the W&B artifact that will be created
        args.artifact_type: Type of the artifact to create
        args.test_size: Ratio of dataset used to test
        args.random_state: Integer to use to seed the random number generator
        args.stratify: If provided, it is considered a column name to be used for stratified splitting
    """
    run = wandb.init(job_type="split_data")

    logger.info("Downloading and reading artifact")
    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()
    logger.info(f"artifact.file(): {artifact_path}")

    logger.info("Create a dataframe from the artifact path")
    df = pd.read_csv(artifact_path, delimiter=',', skiprows = 1)

    # Split first in model_dev/test, then we further divide model_dev in train and validation
    logger.info("Splitting data into train and test")
    splits = {}

   
    splits["train"], splits["test"] = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
        shuffle=True,
        stratify= df[args.stratify]
    )

    # Save the artifacts. We use a temporary directory so we do not leave
    # any trace behind
    with tempfile.TemporaryDirectory() as tmp_dir:

        for split, df in splits.items():

            # Make the artifact name from the name of the provided root plus the split
            artifact_name = f"{args.artifact_root}_{split}.csv"

            # Get the path on disk within the temp directory
            temp_path = os.path.join(tmp_dir, artifact_name)

            logger.info(f"Uploading the {split} dataset to {artifact_name}")

            # Save then upload to W&B
            df.to_csv(temp_path,index=False, header = False)

            artifact = wandb.Artifact(
                name=artifact_name,
                type=args.artifact_type,
                description=f"{split} split of dataset {args.input_artifact}",
            )
            artifact.add_file(temp_path)

            logger.info("Logging artifact")
            run.log_artifact(artifact)

            # This waits for the artifact to be uploaded to W&B. If you
            # do not add this, the temp directory might be removed before
            # W&B had a chance to upload the datasets, and the upload
            # might fail
            artifact.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split a dataset into train and test",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True,
    )

    parser.add_argument(
        "--artifact_root",
        type=str,
        help="Root for the names of the produced artifacts. The script will produce 2 artifacts: "
             "{root}_train.csv and {root}_test.csv",
        required=True,
    )

    parser.add_argument(
        "--artifact_type",
        type=str,
        help="Type for the produced artifacts",
        required=True
    )

    parser.add_argument(
        "--test_size",
        help="Fraction of dataset or number of items to include in the test split",
        type=float,
        required=True
    )

    parser.add_argument(
        "--random_state",
        help="An integer number to use to init the random number generator. It ensures repeatibility in the splitting",
        type=int,
        required=True,
        default=42
    )

    parser.add_argument(
        "--stratify",
        help="If set, it is the name of a column to use for stratified splitting",
        type=str,
        required=True,
        default="y"
    )

    ARGS = parser.parse_args()

    process_args(ARGS)