# **Graph Neural Network Toolkit**

### Latest version: 0.3.1
<br/>

## Features of this package
- [x] Create custom molecular graph datasets
- [x] Classification on graphs
- [x] Regression on graphs
- [ ] Generative adversarial networks for molecular graphs

_____________________________________________________________________________________
<br/>

## **Installation**

1. Install Python3-tk and python3-dev with the command:

    ```console
    sudo apt install python3-dev python3-tk
    ```

2. Clone the repo using the following command:

    ```console
    git clone git@github.com:Takaogahara/gnn-toolkit.git
    ```

3. Create and activate your `virtualenv` with Python 3.8, for example as described [here](https://docs.python.org/3/library/venv.html).

4. Install [Pytorch](https://pytorch.org/get-started/locally/)

5. Install [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

6. Install required libs using:

    ```conseole
    pip install -r requirements.txt
    ```

_____________________________________________________________________________________
<br/>

## **Dataset Organization**

The data loader for **Classification** and **Regression** tasks expects the following structure:

```console   
data/
  ├── dataset_1/
  ├── ...
  ├── dataset_n/
      ├── raw/
      |   ├── test_file.csv/
      |   ├── val_file.csv/
      |   ├── train_file.csv/
      ├── processed/
          ├── empty
```

The `data/` folder is where your datasets are saved.  
Inside `data/datasets_n/` should contain a root folder with the .csv files in a folder named `raw/`.

The path to folder and file is defined in the configuration file.  
Molecules in SMILES format will be converted to pytorch graph objects and saved in the `processed/` folder
_____________________________________________________________________________________
<br/>

## **Configuration**

You must define the network model to be used and its parameters using a YAML configuration file.

There is an example configuration file for the **Classification** and **Regression** tasks in `configs/example_classification.yaml` and `configs/example_regression.yaml`.

Options must always be contained in a list, even when there is only one choice.

Explanation of the configuration file can be found at: [Config File Explanation](./assets/config_file.md)
_____________________________________________________________________________________
<br/>

## **Usage**
With all parameters configured correctly and with the `virtualenv` activated, you can proceed to the execution.

### **1. With MLflow**
Set `RUN_MLFLOW_URI` parameter to `http://localhost:5000` to use MLflow local tracker.  
From the root folder, open the terminal and run:
  ```console
  mlflow ui
  ```
and then in another terminal:

  ```console
  python ./gnn_toolkit/main.py --cfg ./configs/config_file.yaml
  ```

### **2. Without MLflow**
Set `RUN_MLFLOW_URI` parameter to `./mlruns` to store data "locally".  
From the root folder, open the terminal and run:
  ```console
  python ./gnn_toolkit/main.py --cfg ./configs/config_file.yaml
  ```