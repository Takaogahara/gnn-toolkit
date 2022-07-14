import yaml
import psutil
import telegram_send
from ray import tune
from datetime import datetime
from pynvml.smi import nvidia_smi


parameters = {"TASK": ["Classification"],

              "RUN_NAME": ["None"],
              "RUN_MLFLOW_URI": ["http://localhost:5000"],
              "RUN_RAY_SAMPLES": [1],
              "RUN_RAY_MAX_EPOCH": [1],
              "RUN_RAY_CPU": [2],
              "RUN_RAY_GPU": [0],
              "RUN_RAY_TIME_BUDGET_S": [None],
              "RUN_RAY_RESUME": [False],
              "RUN_TELEGRAM_VERBOSE": [0],

              "DATA_DATALOADER": ["Default"],
              "DATA_PREMADE": [True],
              "DATA_ROOT_PATH": ["path/to/data"],
              "DATA_FILE_NAME": ["filename.csv"],

              "MODEL_CHECKPOINT_PATH": [None],
              "MODEL_USE_TRANSFER": [False],
              "MODEL_TRANSFER_PATH":  ["path/to/model"],
              "MODEL_ARCHITECTURE": ["GAT"],
              "MODEL_EMBEDDING_SIZE": [64],
              "MODEL_NUM_LAYERS": [3],
              "MODEL_DROPOUT_RATE": [0.2],
              "MODEL_DENSE_NEURONS": [128],
              "MODEL_NUM_HEADS": [1],
              "MODEL_NUM_TIMESTEPS": [1],
              "MODEL_TOP_K_RATIO": [0.5],
              "MODEL_TOP_K_EVERY_N": [1],

              "SOLVER_BATCH_SIZE": [32],
              "SOLVER_NUM_EPOCH": [1],
              "SOLVER_OPTIMIZER": ["SGD"],
              "SOLVER_LEARNING_RATE": [0.01],
              "SOLVER_L2_PENALTY": [0.001],
              "SOLVER_SCHEDULER_GAMMA": [0.8],
              "SOLVER_LOSS_FN_POS_WEIGHT": ["None"],
              "SOLVER_SGD_MOMENTUM": [0.8],
              "SOLVER_ADAM_BETA_1": [0.9],
              "SOLVER_ADAM_BETA_2": [0.999]}


def get_ray_config(yaml_path: str):
    """ Open YAML file and and process to parse in Ray Tune

    Args:
        yaml_path (str): path to YAML file

    Returns:
        Dict: Processed YAML file
    """
    with open(yaml_path, "r") as f:
        file = yaml.safe_load(f)

    content = {"TASK": file["TASK"]}
    content_list = [file["RUN"], file["DATA"],
                    file["MODEL"], file["SOLVER"]]

    for value in content_list:
        for item in value:
            key, value = list(item.items())[0]
            content[key] = value

    # TODO: Fix auto complete
    # * Auto complete config file
    for key, value in list(content.items()):
        if ("MODEL_" in key) or ("SOLVER_" in key):
            parameters[key] = tune.choice(value)
        else:
            parameters[key] = value

    return parameters


def extract_configs(config_file: dict):
    """Extract YAML file

    Args:
        config_file (dict): Processed YAML file

    Returns:
        Dict: Extracted YAML file
    """

    # * Extract config file
    parameters_new = {}
    for key, value in config_file.items():
        if isinstance(value, list):
            parameters_new[key] = value[0]
        else:
            parameters_new[key] = value

    return parameters_new


class TelegramReport:
    def start_eval(device, task, iter, epochs, verbose=0):
        if verbose != 0:
            gif = "https://media.giphy.com/media/XoCI9HIAQEz1dSIvIy/giphy.gif"
            txt = ("New run detected!\n"
                   f"Run task: {task}\n"
                   f"Number of samples: {iter}\n"
                   f"Number of epochs: {epochs}\n"
                   "\n"
                   f"Running on: {device}")

            telegram_send.send(captions=[txt], animations=[gif])
        else:
            pass

    def end_eval(best_trial, verbose=0):
        if verbose != 0:
            gif = "https://media.giphy.com/media/wJ6mHEehNx3OsE3LCD/giphy.gif"
            txt = ("###################\n"
                   "Run Finished\n"
                   f"Best tial loss: {best_trial.last_result['loss']}\n"
                   f"Best tial dir: {best_trial.logdir}\n"
                   f"Best trial parameters: {best_trial.config}")

            telegram_send.send(captions=[txt], animations=[gif])
        else:
            pass

    def report_run(verbose=0):
        if verbose == 2:
            now = datetime.now()
            now = now.strftime("%d/%m/%Y %H:%M:%S")

            cpu = psutil.cpu_percent()
            # ram_free = psutil.virtual_memory().available
            # ram_free = round((ram_free * 100) / ram_total, 2)
            ram_total = psutil.virtual_memory().total
            ram_used = psutil.virtual_memory().used
            ram_used = round((ram_used * 100) / ram_total, 2)

            nvsmi = nvidia_smi.getInstance()
            gpu = nvsmi.DeviceQuery("memory.free, memory.total, memory.used")
            # gpu_free = gpu["gpu"][0]["fb_memory_usage"]["free"]
            # gpu_free = round((gpu_free * 100) / gpu_total, 2)
            gpu_total = gpu["gpu"][0]["fb_memory_usage"]["total"]
            gpu_used = gpu["gpu"][0]["fb_memory_usage"]["used"]
            gpu_used = round((gpu_used * 100) / gpu_total, 2)

            txt = ("###################\n"
                   "Still running...\n"
                   f"{now}\n"
                   " \n"
                   f"CPU usage: {cpu}%\n"
                   f"RAM usage: {ram_used}%\n"
                   f"GPU usage: {gpu_used}%\n")

            telegram_send.send(messages=[txt])
        else:
            pass
