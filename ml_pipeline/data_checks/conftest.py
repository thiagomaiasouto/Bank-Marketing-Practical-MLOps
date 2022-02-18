import pytest
import pandas as pd
import wandb
import sys
import logging

run = wandb.init(job_type="data_tests")

# configure logging
logging.basicConfig(level=logging.INFO,
                    stream=sys.stdout,
                    format="%(asctime)s %(message)s",
                    datefmt='%d-%m-%Y %H:%M:%S')

# reference for a logging obj
logger = logging.getLogger()

def pytest_addoption(parser):
    parser.addoption("--reference_artifact", action="store")
    parser.addoption("--sample_artifact", action="store")
    parser.addoption("--ks_alpha", action="store")


@pytest.fixture(scope="session")
def data(request):
    reference_artifact = request.config.option.reference_artifact

    if reference_artifact is None:
        pytest.fail("--reference_artifact missing on command line")

    sample_artifact = request.config.option.sample_artifact

    if sample_artifact is None:
        pytest.fail("--sample_artifact missing on command line")

    local_path1 = run.use_artifact(reference_artifact).file()
    logger.info(f"run.use_artifact(reference_artifact).file(): {local_path1}")
    sample1 = pd.read_csv(local_path1, delimiter=';')
    logger.info(f"sample1.columns : {sample1.columns.tolist()}")

    local_path2 = run.use_artifact(sample_artifact).file()
    logger.info(f"run.use_artifact(sample_artifact).file(): {local_path2}")
    sample2 = pd.read_csv(local_path2, delimiter=';')
    logger.info(f"sample2.columns : {sample2.columns.tolist()}")

    return sample1, sample2


@pytest.fixture(scope='session')
def ks_alpha(request):
    ks_alpha = request.config.option.ks_alpha

    if ks_alpha is None:
        pytest.fail("--ks_threshold missing on command line")

    return float(ks_alpha)