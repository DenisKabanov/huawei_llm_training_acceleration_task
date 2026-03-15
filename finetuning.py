"""
This module provides functionality for models finetuning.
"""

# imports
import os
import time
import random
import pickle
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from dotenv import load_dotenv
from datasets import Dataset, load_dataset, load_from_disk
import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import LoraConfig, get_peft_model
import lm_eval
from lm_eval.models.huggingface import HFLM




# load settings for fine-tuning
load_dotenv()
WORKING_DIR = os.getenv("WORKING_DIR")
DATA_DIR = WORKING_DIR + os.getenv("DATA_DIR")
MODELS_DIR = WORKING_DIR + os.getenv("MODELS_DIR")
RESULTS_DIR = WORKING_DIR + os.getenv("RESULTS_DIR")
DATASET_NAME = os.getenv("DATASET_NAME")
MODEL_NAME = os.getenv("MODEL_NAME")
EVAL_BENCHMARKS_TRAINING = EVAL_BENCHMARKS_TRAINING = (
    [
        item.strip().strip('"')
        for item in os.getenv("EVAL_BENCHMARKS_TRAINING").strip("[]").split(",")
    ]
    if os.getenv("EVAL_BENCHMARKS_TRAINING") != "[]"
    else []
)
MAX_SAMPLES = int(os.getenv("MAX_SAMPLES"))
MAX_SEQUENCE_LEN = (
    int(os.getenv("MAX_SEQUENCE_LEN"))
    if os.getenv("MAX_SEQUENCE_LEN").isdigit()
    else os.getenv("MAX_SEQUENCE_LEN")
)  # int or "auto"
TEST_SIZE = float(os.getenv("TEST_SIZE"))
EPOCHS = int(os.getenv("EPOCHS"))
EPOCHS_PATIENCE = int(os.getenv("EPOCHS_PATIENCE"))
TOLERANCE = float(os.getenv("TOLERANCE"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY"))
SCHEDULER_STEP = int(os.getenv("SCHEDULER_STEP"))
SCHEDULER_GAMMA = float(os.getenv("SCHEDULER_GAMMA"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
VERBOSE = int(os.getenv("VERBOSE"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE"))

FORCE_USE_PREPROCESSED_DATA = os.getenv("FORCE_USE_PREPROCESSED_DATA") == "True"
OPTIMIZER = os.getenv("OPTIMIZER")
TRACKED_METRIC = os.getenv("TRACKED_METRIC")

DEVICE = os.getenv("DEVICE") # "cpu", "cuda" or "auto"
if DEVICE in ["cuda", "auto"]:
    DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
else:
    DEVICE = torch.device("cpu")

hyperparams = {
    "WORKING_DIR": WORKING_DIR,
    "DATA_DIR": DATA_DIR,
    "MODELS_DIR": MODELS_DIR,
    "RESULTS_DIR": RESULTS_DIR,
    "DATASET_NAME": DATASET_NAME,
    "MODEL_NAME": MODEL_NAME,
    "EVAL_BENCHMARKS_TRAINING": EVAL_BENCHMARKS_TRAINING,
    "MAX_SAMPLES": MAX_SAMPLES,
    "MAX_SEQUENCE_LEN": MAX_SEQUENCE_LEN,
    "TEST_SIZE": TEST_SIZE,
    "EPOCHS":EPOCHS,
    "EPOCHS_PATIENCE": EPOCHS_PATIENCE,
    "TOLERANCE": TOLERANCE,
    "LEARNING_RATE": LEARNING_RATE,
    "WEIGHT_DECAY": WEIGHT_DECAY,
    "SCHEDULER_STEP":SCHEDULER_STEP,
    "SCHEDULER_GAMMA": SCHEDULER_GAMMA,
    "BATCH_SIZE": BATCH_SIZE,
    "VERBOSE": VERBOSE,
    "RANDOM_STATE": RANDOM_STATE,
    "FORCE_USE_PREPROCESSED_DATA": FORCE_USE_PREPROCESSED_DATA,
    "OPTIMIZER": OPTIMIZER,
    "TRACKED_METRIC": TRACKED_METRIC,
    "DEVICE": DEVICE
}
print("Starting execution with next hyperparams:")
for key, hyperparam in hyperparams.items():
    print(f"{key}: {hyperparam}")
print(f"Device name: {torch.cuda.get_device_name() if str(DEVICE) != 'cpu' else 'CPU'}.")




# creating dirs for output if not exists
print("Creating output dirs if not exists...")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)




# setting random seed for reproducibility
def set_seed(seed: int = RANDOM_STATE):
    """
    Function for setting the random seed.
    Parameters:
        seed (int) : Fixed seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed()




# model preparation
print("Preparing model...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.save_pretrained(MODELS_DIR + MODEL_NAME, from_pt=True)
# config = LoraConfig(
#     r=32,
#     lora_alpha=64,
#     target_modules="all-linear",
#     # init_lora_weights="gaussian",
#     lora_dropout=0.05,
#     bias="none",
#     task_type="CAUSAL_LM"
# )
# model = get_peft_model(model, config)
model.to(DEVICE)
MODEL_NAME_TO_SAVE = f"{MODEL_NAME}_{OPTIMIZER}"




# tokenizer preparation
print("Preparing tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.save_pretrained(MODELS_DIR + MODEL_NAME)




# data preparation
print("Preparing data...")
if FORCE_USE_PREPROCESSED_DATA and os.path.exists(DATA_DIR + DATASET_NAME + "_processed"):
    print("\t Loading preprocesed data...")
    data = load_from_disk(DATA_DIR + DATASET_NAME + "_processed")
else:
    print("\t Preprocesed data...")
    data = load_dataset(DATASET_NAME)
    data.save_to_disk(DATA_DIR + DATASET_NAME)
    data["train"] = data["train"].select(range(MAX_SAMPLES)) # limit dataset by MAX_SAMPLES

    # calculate MAX_SEQUENCE_LEN if set to "auto"
    if MAX_SEQUENCE_LEN == "auto":
        tokens_count = []
        for i in range(MAX_SAMPLES):
            data_tokenized = tokenizer(data["train"]["text"][i])
            tokens_count.append(len(data_tokenized["input_ids"]))

        MAX_SEQUENCE_LEN = int(np.percentile(tokens_count, 95))
    print(f"The number of tokens generation is limited by the value {MAX_SEQUENCE_LEN}.")

    def preprocess_function(data: Dataset) -> Dataset:
        """
        Function for Dataset preprocessing.
        Parameters:
            data (Dataset) : Dataset for preprocess.
        Returns:
            Dataset: Processed Dataset.
        """
        model_input = tokenizer(data["text"],
                                max_length=MAX_SEQUENCE_LEN,
                                return_tensors="pt",
                                padding=True,
                                truncation=True
                                )
        model_input["labels"] = model_input["input_ids"].clone()
        return model_input

    data = data.map(preprocess_function, batched=True)
    data.set_format(type='torch', columns=["text", "input_ids", "attention_mask", "labels"])
    data = data["train"].train_test_split(test_size=TEST_SIZE, shuffle=True, seed=RANDOM_STATE)
    data.save_to_disk(DATA_DIR + DATASET_NAME + "_processed")
train_dataloader = DataLoader(data["train"], batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(data["test"], batch_size=BATCH_SIZE)




# optimizers preparation
print("Preparing optimizers...")
optimizers = []
if OPTIMIZER == "AdamW":
    adamw_params = [param for param in model.parameters() if param.requires_grad]

    optimizer_adamw = torch.optim.AdamW(
        adamw_params,
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=WEIGHT_DECAY,
        eps=1e-08,
    )
    optimizers.append(optimizer_adamw)
elif OPTIMIZER == "Muon":
    muon_params = []
    for name, param in model.named_parameters():
        if (param.ndim == 2) and ("embed" not in name):
            muon_params.append(param)
        else:
            param.requires_grad = False


    optimizer_muon = torch.optim.Muon(
        muon_params,
        lr=LEARNING_RATE,
        adjust_lr_fn="match_rms_adamw",
        momentum=0.95,
        ns_coefficients=(3.4445, -4.775, 2.0315),
        ns_steps=5,
        weight_decay=WEIGHT_DECAY,
        eps=1e-07,
    )
    optimizers.append(optimizer_muon)
elif OPTIMIZER == "Muon_with_AdamW":
    # trainable_params = [param for param in model.parameters() if param.requires_grad]
    # random.shuffle(trainable_params)
    # muon_params = trainable_params[:len(trainable_params) // 2]
    # adamw_params = trainable_params[len(trainable_params) // 2:]

    muon_params = []
    adamw_params = []
    for name, param in model.named_parameters():
        if (param.ndim == 2) and ("embed" not in name):
            muon_params.append(param)
        else:
            adamw_params.append(param)

    optimizer_muon = torch.optim.Muon(
        muon_params,
        lr=LEARNING_RATE,
        adjust_lr_fn="match_rms_adamw",
        momentum=0.95,
        ns_coefficients=(3.4445, -4.775, 2.0315),
        ns_steps=5,
        weight_decay=WEIGHT_DECAY,
        eps=1e-07,
    )

    optimizer_adamw = torch.optim.AdamW(
        adamw_params,
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=WEIGHT_DECAY,
        eps=1e-08,
    )
    optimizers.append(optimizer_muon, optimizer_adamw)
else:
    raise RuntimeError(f"Optimizer '{OPTIMIZER}' is not recognized.")




# functions for training and evaluation
def clear_gpu_cache() -> None:
    """
    Function for cleaning up used and no longer reserved memory on the GPU.
    """
    if "cuda" in str(DEVICE):
        torch.cuda.empty_cache()
        torch.cuda.memory.reset_peak_memory_stats()


def log_gpu_memory() -> tuple:
    """
    Function for measuring the current GPU load.
    Returns:
        tuple: Tuple (allocated_memory, reserved_memory, max_allocated_memory_per_logging_period).
    """
    if "cuda" in str(DEVICE):
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
    else:
        allocated = 0
        reserved = 0
        max_allocated = 0
    return allocated, reserved, max_allocated


def print_trainable_parameters(model):
    """
    Function for counting trainable model's parameters.
    Parameters:
        model (Any) : Trainable model.
    """
    trainable_params = 0
    all_params = 0
    for param in model.parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || \
            all params: {all_params} || \
            trainable: {100 * trainable_params / all_params:.4f}%")


def benchmark_model(model, benchmarks: list[str]) -> pd.DataFrame:
    """
    Function for validating the model on benchmarks.
    Parameters:
        model (Any) : Testing model (path_to_model or model object).
        benchmarks (list[str]) : List of benchmarks names.
    Returns:
        pd.DataFrame: DataFrame with metrics.
    """
    if isinstance(model, str):
        model = AutoModelForCausalLM.from_pretrained(model).to(DEVICE)

    benchmarks = lm_eval.simple_evaluate(
        model=HFLM(model),
        tasks=benchmarks,
        # batch_size="auto:4",
        batch_size=BATCH_SIZE,
        device=str(DEVICE),
        random_seed=RANDOM_STATE,
        numpy_random_seed=RANDOM_STATE,
        torch_random_seed=RANDOM_STATE,
        verbosity="ERROR",
    )["results"]

    return pd.DataFrame(benchmarks)


def train_model(
    model,
    optimizers,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    model_name: str,
    tracked_metric: str = "Test loss",
) -> dict:
    """
    Function for training and testing model, stores the best found model based
    on the tracked_metric and returns training logs.
    Parameters:
        model (Any) : Trainable model.
        optimizers (Any) : List of models optimizators.
        train_dataloader (DataLoader) : Loader for training data.
        test_dataloader (DataLoader) : Loader for test data.
        model_name (str) : Model name for saving.
        tracked_metric (str) : Tracked metric for early stopping of training
            (in addition to convergence of weights).
    Returns:
        dict: Dict with logs of the training and testing process.
    """
    history = {
        "Hyperparams": hyperparams,
        "Train loss": {},
        "Test loss": {},
        "Weight convergence": {},
        "Time start": None,
        "Time end": None,
        "Time epoch": {},
        "GPU memory": {},
        "Learning rate": {},
        "Train benchmarks": {},
        "AVG benchmarks acc": {},
        "Tracked metric": tracked_metric,
    }

    schedulers = [
        lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_STEP, gamma=SCHEDULER_GAMMA)
        for optimizer in optimizers
    ]

    clear_gpu_cache()
    history["GPU memory"]["init"] = log_gpu_memory()


    # estimation before training
    model.eval()
    with torch.no_grad():
        # loss on training data
        loss_train_total = 0
        for batch in tqdm(train_dataloader):
            input_ids = batch["input_ids"].to(DEVICE)
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss_train = outputs.loss # specific CrossEntropyLoss

            loss_train_total += loss_train.item()
        loss_train_avg = loss_train_total / len(train_dataloader)

        # loss on test data
        loss_test_total = 0
        for batch in tqdm(test_dataloader):
            input_ids = batch["input_ids"].to(DEVICE)
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss_test = outputs.loss # specific CrossEntropyLoss

            loss_test_total += loss_test.item()
        loss_test_avg = loss_test_total / len(test_dataloader)

        # validation on benchmarks
        if len(EVAL_BENCHMARKS_TRAINING) > 0:
            history["Train benchmarks"]["init"] = benchmark_model(model, EVAL_BENCHMARKS_TRAINING)
            history["AVG benchmarks acc"]["init"] = (
                history["Train benchmarks"]["init"].loc["acc,none"].mean().item()
            )
        else:
            history["Train benchmarks"]["init"] = None
            history["AVG benchmarks acc"]["init"] = 0
    history["Train loss"]["init"] = loss_train_avg
    history["Test loss"]["init"] = loss_test_avg
    print(
        f"Initial metrics:\n \
        Average train Loss: {loss_train_avg:.4f}, \
        Average test Loss: {loss_test_avg:.4f}, \
        Tracked metric ({tracked_metric}): {history[tracked_metric]['init']:.6f}."
    )

    # values for early stopping
    epochs_without_improve = 0
    last_best_metric = -float("inf")
    previous_weights = [param.clone() for param in model.parameters() if param.requires_grad]

    # training-evaluation loop
    time_start = time.time()
    for epoch in range(EPOCHS):
        clear_gpu_cache()

        # training
        model.train()
        time_start_epoch = time.time()
        loss_train_total = 0
        for batch in tqdm(train_dataloader):
            for optimizer in optimizers:
                optimizer.zero_grad()

            input_ids = batch["input_ids"].to(DEVICE)

            outputs = model(input_ids=input_ids, labels=input_ids)
            loss_train = outputs.loss # specific CrossEntropyLoss

            loss_train.backward()
            for optimizer in optimizers:
                optimizer.step()

            loss_train_total += loss_train.item()
        loss_train_avg = loss_train_total / len(train_dataloader)

        # update learning rate
        current_lr = np.mean([optimizer.param_groups[0]["lr"] for optimizer in optimizers]).item()
        for scheduler in schedulers:
            scheduler.step()


        history["Time epoch"][epoch + 1] = time.time() - time_start_epoch
        history["GPU memory"][epoch + 1] = log_gpu_memory()
        history["Learning rate"][epoch + 1] = current_lr


        # evaluation
        model.eval()
        loss_test_total = 0
        with torch.no_grad():
            # loss on test data
            for batch in tqdm(test_dataloader):
                input_ids = batch["input_ids"].to(DEVICE)
                outputs = model(input_ids=input_ids, labels=input_ids)
                loss_test = outputs.loss # specific CrossEntropyLoss

                loss_test_total += loss_test.item()
            loss_test_avg = loss_test_total / len(test_dataloader)


            # weight convergence
            current_weights = [param for param in model.parameters() if param.requires_grad]
            weight_change = sum(
                torch.norm(curr_param - prev_param, p=2).item()
                for curr_param, prev_param in zip(current_weights, previous_weights)
            )
            previous_weights = [
                param.clone() for param in model.parameters() if param.requires_grad
            ]


            # validation on benchmarks
            if len(EVAL_BENCHMARKS_TRAINING) > 0:
                history["Train benchmarks"][epoch + 1] = benchmark_model(
                    model, EVAL_BENCHMARKS_TRAINING
                )
                history["AVG benchmarks acc"][epoch + 1] = (
                    history["Train benchmarks"][epoch + 1].loc["acc,none"].mean().item()
                )
            else:
                history["Train benchmarks"][epoch + 1] = None
                history["AVG benchmarks acc"][epoch + 1] = 0


        history["Train loss"][epoch + 1] = loss_train_avg
        history["Test loss"][epoch + 1] = loss_test_avg
        history["Weight convergence"][epoch + 1] = weight_change


        # logging training info
        if (epoch + 1) % VERBOSE == 0:
            print(
                f"Epoch [{epoch + 1}/{EPOCHS}], \
                Average train Loss: {loss_train_avg:.4f}, \
                Average test Loss: {loss_test_avg:.4f}, \
                Tracked metric ({tracked_metric}): {history[tracked_metric][epoch + 1]:.6f}, \
                Learning rate: {current_lr:.8f}, Weight change: {weight_change:.6f}."
            )


        # early stopping and model saving (by tracked_metric)
        patience_reduced = False
        if history[tracked_metric][epoch + 1] > last_best_metric:
            last_best_metric = history[tracked_metric][epoch + 1]
            epochs_without_improve = 0
            model.save_pretrained(f"{MODELS_DIR}{model_name}", from_pt=True)
            tokenizer.save_pretrained(f"{MODELS_DIR}{model_name}")
        elif epochs_without_improve < EPOCHS_PATIENCE:
            epochs_without_improve += 1
            patience_reduced = True
        else:
            print(
                f"Early stopping at epoch {epoch + 1}, \
                tracked metric: {history[tracked_metric][epoch + 1]:.6f}!"
            )
            break

        # early stopping (by weight convergence ~ speed up early stopping by tracked_metric,
        # if L2 norm of weight change is small)
        if not patience_reduced:
            if weight_change < TOLERANCE:
                if epochs_without_improve < EPOCHS_PATIENCE:
                    epochs_without_improve += 1
                else:
                    print(
                        f"Early stopping at epoch {epoch + 1}, \
                        weight change: {weight_change:.6f}!"
                    )
                    break
            else:
                epochs_without_improve = 0


    time_end = time.time()
    clear_gpu_cache()
    history["Time start"], history["Time end"] = time_start, time_end
    history["GPU memory"]["end"] = log_gpu_memory()
    print(
        f"Total training time: {time_end - time_start:.2f} sec \
        ~ {(time_end - time_start) / 60**2:.2f} hours."
    )

    return history


def save_history(history, model_name) -> None:
    """
    Function to save training history.
    Parameters:
        history (dict) : Dict with logs.
        model_name (str) : Fine-tuned model name (in which subdir save logs).
    """
    os.makedirs(f"{RESULTS_DIR}{model_name}/", exist_ok=True)
    with open(f"{RESULTS_DIR}{model_name}/history.pkl", 'wb') as file:
        pickle.dump(history, file)




# training
set_seed()
print_trainable_parameters(model)
history = train_model(
    model,
    optimizers,
    train_dataloader,
    test_dataloader,
    MODEL_NAME_TO_SAVE,
    "AVG benchmarks acc"
)
save_history(history, MODEL_NAME_TO_SAVE)
model.save_pretrained(f"{MODELS_DIR}{MODEL_NAME_TO_SAVE}_final", from_pt=True)
tokenizer.save_pretrained(f"{MODELS_DIR}{MODEL_NAME_TO_SAVE}_final")
