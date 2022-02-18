# About the Problem:

The problem chosen consists of a classification problem, based on the Bank Marketing Dataset, that was obtained [here](https://archive.ics.uci.edu/ml/datasets/bank+marketing). The problem rests in the correct classification if a bank customer accepts or does not subscribe to a bank term deposit based on the client data and data obtained by the marketing campaign. More details about the dataset and the problem itself can be found in the Notebook in the repository.


# Used setup

For the development of this project, the setup used was a Windows 10 machine with the wsl2 installed with the Ubuntu 20.04 Distro. In addition to that, the Anaconda3 was installed in the virtual machine and was creating the mlops environment, which was responsible to execute the entire pipeline.

# Setup and Running the project

It's assumed that Anaconda3 was already installed on the computer. With that said, the **mlops** environment can be created using the following command:

```bash
conda env create -f env.yml
```

Once then environment has been installed, it's necessary to activate it:

```bash
conda activate mlops
```

Since the **Wandb** is used during the entire execution of the pipeline, it's necessary to make a login using the **API keys**, which can be found in the account settings in the [Wandb](https://wandb.ai/). To login into the account the use the command:

```bash
wandb login --relogin
```

## A few notes before execute the mlflow

 When chaining together the steps, the output artifact of a step should be the input artifact
  of the next one (when applicable). Also use the ``artifact_type`` options so that the final
  visualization of the pipeline highlights the different steps. For example, you can use
  ``raw_data`` for the artifact containing the downloaded data, ``preprocessed_data`` for the
  artifact containing the data after the preprocessing, and so on.
  
 Parameters can be override using the parameter ``main.execute_steps`` to only execute one or
  more steps of the pipeline, instead of the entire pipeline. This is useful for debugging. 

  For example, this only executes the ``svm`` step:

  ```bash
  mlflow run . -P hydra_options="main.execute_steps='svm'"
  ```
  and this executes ``download`` and ``preprocess``:

  ```bash
  mlflow run . -P hydra_options="main.execute_steps='download,preprocess'"
  ```


 To run the entire pipeline just use the following command to execute the project with the default settings defined in the ``config.yaml`` in the folder ``ml_pipeline`` of the repository:


  ```bash
  mlflow run . 
  ```

# Authors:
- [Arthur Cunha](https://github.com/arthurfpcl22)
- [Thiago Maia](https://github.com/thiagomaiasouto)
- [Vinicius Malafaya](https://github.com/malafaya9)