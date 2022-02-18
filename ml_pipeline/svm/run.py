"""
Creator: Ivanovitch Silva
Date: 25 Jan. 2022
Implement a pipeline component to train a decision tree model.
"""

import argparse
import logging
import json

import pandas as pd
import matplotlib.pyplot as plt
import wandb
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
from sklearn.svm import SVC

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

    logger.info("Downloading and reading train artifact")
    local_path = run.use_artifact(args.train_data).file()
    df_train = pd.read_csv(local_path)

    # Spliting train.csv into train and validation dataset
    logger.info("Spliting data into train/val")
    # split-out train/validation and test dataset
    x_train, x_val, y_train, y_val = train_test_split(df_train.drop(labels="y", axis=1),
                                                      df_train["y"],
                                                      test_size=0.30,
                                                      random_state=41,
                                                      shuffle=True,
                                                      stratify=df_train["y"])

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
    new_data = full_pipeline_preprocessing.fit_transform(x_train)
    catnames = full_pipeline_preprocessing.get_params()["cat_pipeline"][2].get_feature_names_out().tolist()
    numnames = full_pipeline_preprocessing.get_params()["num_pipeline"][1].get_feature_names()
    X_train = pd.DataFrame(new_data, columns=catnames + numnames)
    retirar = ["job_unknown", "education_unknown", "default_no", "housing_no", "loan_no"]
    X_train = X_train.drop(retirar, axis=1)

    new_data = full_pipeline_preprocessing.transform(x_val)
    catnames = full_pipeline_preprocessing.get_params()["cat_pipeline"][2].get_feature_names_out().tolist()
    numnames = full_pipeline_preprocessing.get_params()["num_pipeline"][1].get_feature_names()
    X_val = pd.DataFrame(new_data, columns=catnames + numnames)

    retirar = ["job_unknown", "education_unknown", "default_no", "housing_no", "loan_no"]
    X_val = X_val.drop(retirar, axis=1)
    # Modeling and Training
    # Get the configuration for the model

    rus = RandomUnderSampler(
        sampling_strategy='auto',  # samples only the majority class
        random_state=0,  # for reproducibility
        replacement=True  # if it should resample with replacement
    )

    # Aplicando o undersampling ao conjunto de treinamento
    X_train_sub, y_train_sub = rus.fit_resample(X_train, y_train)
    # Aplicando o undersampling ao conjunto de teste
    X_val_sub, y_val_sub = rus.fit_resample(X_val, y_val)

    with open(args.model_config) as fp:
        model_config = json.load(fp)

    # Add it to the W&B configuration so the values for the hyperparams
    # are tracked
    wandb.config.update(model_config)

    # training
    logger.info("Training")
    modelo2 = SVC(**model_config)
    modelo2.fit(X_train_sub, y_train_sub)

    # predict
    logger.info("Infering")
    y2_pred = modelo2.predict(X_val_sub)

    # Evaluation Metrics
    logger.info("Evaluation metrics")
    # Metric: AUC
    auc = roc_auc_score(y_val_sub, y2_pred, average="macro")
    run.summary["AUC"] = auc

    # Metric: Accuracy
    acc = accuracy_score(y_val_sub, y2_pred)
    run.summary["Accuracy"] = acc

    # Metric: Confusion Matrix
    fig_confusion_matrix, ax = plt.subplots(1, 1, figsize=(7, 4))
    ConfusionMatrixDisplay(confusion_matrix(y2_pred,
                                            y_val_sub,
                                            labels=[1, 0]),
                           display_labels=["yes", "no"]
                           ).plot(values_format=".0f", ax=ax)
    ax.set_xlabel("True Label")
    ax.set_ylabel("Predicted Label")




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

    ARGS = parser.parse_args()

    process_args(ARGS)