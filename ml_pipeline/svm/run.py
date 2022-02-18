"""
Creator: Ivanovitch Silva
Date: 25 Jan. 2022
Implement a pipeline component to train a decision tree model.
"""

import argparse
import logging
import json
import os
import yaml
import tempfile

import pandas as pd
import matplotlib.pyplot as plt
import wandb
import imblearn.pipeline as imb_pipe
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from imblearn.under_sampling import RandomUnderSampler
from mlxtend.plotting import plot_decision_regions
from sklearn.svm import SVC
import mlflow
from mlflow.models import infer_signature
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import recall_score


# option
# from sklearn.impute import SimpleImputer

# configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s",
                    datefmt='%d-%m-%Y %H:%M:%S')

# reference for a logging obj
logger = logging.getLogger()


# Custom Transformer that extracts columns passed as argument to its constructor
"""
 As colunas enviados como argumento são enviadas ao construtor para posterior tratamento
 aqui são herdadas classes da biblioteca sklearn.base caso algun método venha a ser necessário
"""


class Seletor_colunas(BaseEstimator, TransformerMixin):
    # Construtor
    def __init__(self, nome_colunas):
        self.nome_colunas = nome_colunas

        # Método fit que não retornará nada, pois não será utilizado

    def fit(self, X, y=None):
        return self

        # Método transform caso seja necessário alguma alteração nas colunas, como as alterações não serão realizadas nessa fase, apenas retornará

    # as colunas como foi passado no argumento
    def transform(self, X, y=None):
        return X[self.nome_colunas]


# Handling categorical features
class Transformacao_categorica(BaseEstimator, TransformerMixin):
    # Construtor
    def __init__(self, new_features=True):
        self.new_features = new_features
        self.colnames = None

    def fit(self, X, y=None):
        return self

    def get_feature_names(self):
        return self.colnames.tolist()

    # Modificação das colunas do dataset
    def transform(self, X, y=None):
        df = X.copy()

        # condição para caso seja definido no argumento a mudança de colunas
        if self.new_features:
            retirar = ["contact", "poutcome"]
            df = df.drop(retirar, axis=1)

        # Atualização das colunas
        self.colnames = df.columns

        return df


# transform numerical features
'''
Um fato interessante dessa classe está no fato dela permitir alterar o tipo de 
padronização apenas se alterando o argumento que vai para a definição da classe.
Caso model=0 o modelo será o MinMaxScaler() nesse modelo os valores das colunas serão 
dispostos de forma proporcional a um máximo e um mínimo valor definido pelo usuário.
Caso model=1 será utilizada a StandardScaler(), com essa função, os valores da escala
serão divididos de forma que a média será 0 e o desvio padrão 1, de forma em que cada 
valor será normalizado subtraindo a média e dividindo pelo desvio padrão.
Se o model for qualquer outro valor, não será realizado nenhum tipo de padronização.
'''


class Transformacao_numerica(BaseEstimator, TransformerMixin):
    # Construtor
    def __init__(self, model=0, new_features=True):
        self.new_features = new_features
        self.model = model
        self.colnames = None

    def fit(self, X, y=None):
        return self

    def get_feature_names(self):
        return self.colnames

        # Método para alteração das colunas

    def transform(self, X, y=None):
        df = X.copy()
        if self.new_features:
            retirar = ["pdays"]
            df = df.drop(retirar, axis=1)
        #
        self.colnames = df.columns.tolist()

        # escolha do modelo
        if self.model == 0:
            scaler = MinMaxScaler()

            df = scaler.fit_transform(df)
        elif self.model == 1:
            scaler = StandardScaler()

            df = scaler.fit_transform(df)
        else:
            df = df.values

        return df


def process_args(args):
    # project name comes from config.yaml >> project_name: Trabalho_ivan
    run = wandb.init(job_type="train")

    # columns used
    columns = ['age', 'job', 'marital', 'education', 'default', 'balance',
                'housing', 'loan', 'contact', 'day', 'month', 'duration',
                'campaign', 'pdays', 'previous', 'poutcome', 'y']

    logger.info("Downloading and reading train artifact")
    local_path = run.use_artifact(args.train_data).file()
    logger.info("Create a dataframe from the artifact path")
    df_train = pd.read_csv(local_path, delimiter=',', names = columns)
    logger.info(f"columns: {df_train.columns}")

    # Spliting train.csv into train and validation dataset
    logger.info("Spliting data into train/val")
    # split-out train/validation and test dataset
    x_train, x_val, y_train, y_val = train_test_split(df_train.drop(labels=args.stratify,axis=1),
                                                      df_train[args.stratify],
                                                      test_size=args.val_size,
                                                      random_state=args.random_seed,
                                                      shuffle=True,
                                                      stratify=df_train[args.stratify])

    logger.info("x train: {}".format(x_train.shape))
    logger.info("y train: {}".format(y_train.shape))
    logger.info("x val: {}".format(x_val.shape))
    logger.info("y val: {}".format(y_val.shape))

    logger.info("Removal Outliers")
    # temporary variable
    x = x_train.select_dtypes("int64").copy()

    # identify outlier in the dataset
    lof = LocalOutlierFactor()
    outlier = lof.fit_predict(x)
    mask = outlier != -1

    logger.info("x_train shape [original]: {}".format(x_train.shape))
    logger.info("x_train shape [outlier removal]: {}".format(x_train.loc[mask, :].shape))

    # dataset without outlier, note this step could be done during the preprocesing stage
    x_train = x_train.loc[mask, :].copy()
    y_train = y_train[mask].copy()

    logger.info("Encoding Target Variable")
    # define a categorical encoding for target variable
    le = LabelEncoder()

    # fit and transform y_train
    y_train = le.fit_transform(y_train)

    # transform y_test (avoiding data leakage)
    y_val = le.transform(y_val)

    logger.info("Classes [0, 1]: {}".format(le.inverse_transform([0, 1])))

    # Pipeline generation
    logger.info("Pipeline generation")
    # Seleção das colunas categóricas
    categorical_features = x_train.select_dtypes("object").columns.to_list()

    # Seleção das colunas numéricas
    numerical_features = x_train.select_dtypes("int64").columns.to_list()

    # Definição do Pipeline das variáveis categóricas, e cada etapa a ser realizada
    categorical_pipeline = Pipeline(steps=[('cat_selector', Seletor_colunas(categorical_features)),
                                           ('cat_transformer', Transformacao_categorica()),
                                           ('cat_encoder', OneHotEncoder(sparse=False))
                                           ]
                                    )

    # Definição do Pipeline das variáveis numéricas, e cada etapa a ser realizada
    numerical_pipeline = Pipeline(steps=[('num_selector', Seletor_colunas(numerical_features)),
                                         ('num_transformer', Transformacao_numerica())
                                         ]
                                  )

    # União das variáveis categóricas e numéricas
    full_pipeline_preprocessing = FeatureUnion(transformer_list=[('cat_pipeline', categorical_pipeline),
                                                                 ('num_pipeline', numerical_pipeline)
                                                                 ]
                                               )

    # Instatiating the random under sampling class
    rus =  RandomUnderSampler(sampling_strategy='auto',  
                              random_state=0,  
                              replacement=True 
                             )  

    # The full pipeline
    pipe = imb_pipe.Pipeline(steps = [('full_pipeline', full_pipeline_preprocessing),
                                      ('rus', rus ),
                                      ("classifier", SVC(kernel = 'linear', gamma = 'scale'))
                                    ]
                            )

    logger.info(f"args.model_config:  {args.model_config}")

    with open(args.model_config,'rb') as fp:
        model_config = yaml.safe_load(fp)

    # Add it to the W&B configuration so the values for the hyperparams
    # are tracked
    wandb.config.update(model_config)

    # training
    logger.info("Training")
    pipe.fit(x_train, y_train)

    # predict
    logger.info("Infering")
    predict = pipe.predict(x_val)

    # Evaluation Metrics
    logger.info("Evaluation metrics")
    # Metric: AUC
    auc = roc_auc_score(y_val, predict, average="macro")
    run.summary["AUC"] = auc

    # Metric: Accuracy
    acc = accuracy_score(y_val, predict)
    run.summary["Accuracy"] = acc


    # Metric: Balanced Accuracy
    blc_acc = balanced_accuracy_score(y_val, predict)
    run.summary["Balanced_Accuracy"] = blc_acc

    # Metric: Recall
    rcll = recall_score(y_val, predict)
    run.summary["Recall"] = rcll

    # Metric: Confusion Matrix
    fig_confusion_matrix, ax = plt.subplots(1, 1, figsize=(7, 4))
    ConfusionMatrixDisplay(confusion_matrix(predict,
                                            y_val,
                                            labels=[1, 0]),
                           display_labels=["yes", "no"]
                           ).plot(values_format=".0f", ax=ax)
    ax.set_xlabel("True Label")
    ax.set_ylabel("Predicted Label")


    # Uploading figures
    logger.info("Uploading figures")
    run.log(
        {
            "confusion_matrix": wandb.Image(fig_confusion_matrix)
        }
    )

    # Export if required
    if args.export_artifact != "null":
        export_model(run, pipe, x_val, predict, args.export_artifact)



def export_model(run, pipe, x_val, val_pred, export_artifact):

    # Infer the signature of the model
    signature = infer_signature(x_val, val_pred)

    with tempfile.TemporaryDirectory() as temp_dir:

        export_path = os.path.join(temp_dir, "model_export")

        mlflow.sklearn.save_model(
            pipe, # our pipeline
            export_path, # Path to a directory for the produced package
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            signature=signature, # input and output schema
            input_example=x_val.iloc[:2], # the first few examples
        )

        artifact = wandb.Artifact(
            export_artifact,
            type="model_export",
            description="SVM pipeline export",
        )
        
        # NOTE that we use .add_dir and not .add_file
        # because the export directory contains several
        # files
        artifact.add_dir(export_path)

        run.log_artifact(artifact)

        # Make sure the artifact is uploaded before the temp dir
        # gets deleted
        artifact.wait()        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a SVM",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--train_data",
        type=str,
        help="Fully-qualified name for the training data artifact",
        required=True,
    )

    parser.add_argument(
        "--model_config",
        type=str,
        help="Path to a JSON file containing the configuration for the Decision Tree",
        required=True,
    )

    parser.add_argument(
        "--export_artifact",
        type=str,
        help="Name of the artifact for the exported model. Use 'null' for no export.",
        required=False,
        default="null",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        help="Seed for the random number generator.",
        required=False,
        default=42
    )

    parser.add_argument(
        "--val_size",
        type=float,
        help="Size for the validation set as a fraction of the training set",
        required=False,
        default=0.3
    )

    parser.add_argument(
        "--stratify",
        type=str,
        help="Name of a column to be used for stratified sampling. Default: 'null', i.e., no stratification",
        required=False,
        default="null",
    )

    ARGS = parser.parse_args()

    process_args(ARGS)