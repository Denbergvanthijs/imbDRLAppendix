import csv
import os

from imbDRL.agents.ddqn import TrainDDQN
from imbDRL.data import get_train_test_val, load_csv
from imbDRL.metrics import classification_metrics, network_predictions
from tensorflow.keras.layers import Dense, Dropout
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU is faster than GPU on structured data

episodes = 100_000  # Total number of episodes
warmup_steps = 170_000  # Amount of warmup steps to collect data with random policy
memory_length = warmup_steps  # Max length of the Replay Memory
batch_size = 32
collect_steps_per_episode = 2000
collect_every = 500

target_update_period = 800  # Period to overwrite the target Q-network with the default Q-network
target_update_tau = 1  # Soften the target model update
n_step_update = 1

layers = [Dense(256, activation="relu"), Dropout(0.2),
          Dense(256, activation="relu"), Dropout(0.2),
          Dense(2, activation=None)]  # No activation, pure Q-values

learning_rate = 0.00025  # Learning rate
gamma = 0.0  # Discount factor
min_epsilon = 0.5  # Minimal and final chance of choosing random action
decay_episodes = episodes // 10  # Number of episodes to decay from 1.0 to `min_epsilon`

min_class = [1]  # Labels of the minority classes
maj_class = [0]  # Labels of the majority classes
_X_train, _y_train, _X_test, _y_test = load_csv("./data/credit0.csv", "./data/credit1.csv", "Class", ["Time"], normalization=True)

fp_dqn = "./results/creditcardfraud/dqn.csv"
fieldnames = ("Gmean", "F1", "Precision", "Recall", "TP", "TN", "FP", "FN")

# Create empty files
with open(fp_dqn, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

# Run the model ten times
for _ in tqdm(range(10)):
    # New train-test split
    X_train, y_train, X_test, y_test, X_val, y_val = get_train_test_val(_X_train, _y_train, _X_test, _y_test, min_class, maj_class,
                                                                        val_frac=0.2, print_stats=False)

    model = TrainDDQN(episodes, warmup_steps, learning_rate, gamma, min_epsilon, decay_episodes, target_update_period=target_update_period,
                      target_update_tau=target_update_tau, batch_size=batch_size, collect_steps_per_episode=collect_steps_per_episode,
                      memory_length=memory_length, collect_every=collect_every, n_step_update=n_step_update, progressbar=False)

    model.compile_model(X_train, y_train, layers)
    model.train(X_val, y_val, "F1")

    # Predictions of model for `X_test`
    best_network = model.load_model(fp=model.model_path)
    y_pred = network_predictions(best_network, X_test)
    dqn_stats = classification_metrics(y_test, y_pred)

    # Write current DQN run to `fp_dqn`
    with open(fp_dqn, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(dqn_stats)
