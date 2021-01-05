import csv

from imbDRL.agents.ddqn import TrainDDQN
from imbDRL.data import get_train_test_val, load_image
from imbDRL.metrics import classification_metrics, network_predictions
from tqdm import tqdm

episodes = 120_000  # Total episodes
warmup_steps = 50_000  # Amount of warmup steps to collect data with random policy
memory_length = 100_000  # Max length of the Replay Memory
collect_steps_per_episode = 1
target_update_period = 10_000
target_update_tau = 1
batch_size = 32
n_step_update = 4

conv_layers = ((32, (5, 5), 2), (32, (5, 5), 2), )  # Convolutional layers
dense_layers = (256, )  # Dense layers
dropout_layers = None  # Dropout layers

learning_rate = 0.00025  # Learning rate
gamma = 0.1  # Discount factor
min_epsilon = 0.01  # Minimal and final chance of choosing random action
decay_episodes = 100_000  # Number of episodes to decay from 1.0 to `min_epsilon`

min_class = [4, 5, 6]  # Minority classes
maj_class = [7, 8, 9]  # Majority classes
_X_train, _y_train, _X_test, _y_test = load_image("famnist")

fp_dqn = "./results/lin/famnist_2.csv"
fieldnames = ("Gmean", "F1", "Precision", "Recall", "TP", "TN", "FP", "FN", "P")

# Create empty files
with open(fp_dqn, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

for p in (0.04, 0.02, 0.01, 0.005):
    # Run the model ten times
    for _ in tqdm(range(10), desc=f"Running model for imbalance ratio:{p}"):
        # New train-test split
        X_train, y_train, X_test, y_test, X_val, y_val = get_train_test_val(_X_train, _y_train, _X_test, _y_test, min_class, maj_class,
                                                                            imb_rate=p, imb_test=False, val_frac=0.1, print_stats=False)

        model = TrainDDQN(episodes, warmup_steps, learning_rate, gamma, min_epsilon, decay_episodes, target_update_tau=target_update_tau,
                          collect_steps_per_episode=collect_steps_per_episode, target_update_period=target_update_period,
                          n_step_update=n_step_update, batch_size=batch_size, memory_length=memory_length, progressbar=False)

        model.compile_model(X_train, y_train, conv_layers, dense_layers, dropout_layers, imb_rate=p)
        model.train(X_val, y_val, "Gmean")

        # Predictions of model for `X_test`
        best_network = model.load_model(fp=model.model_path)
        y_pred = network_predictions(best_network, X_test)
        dqn_stats = classification_metrics(y_test, y_pred)

        # Write current DQN run to `fp_dqn`
        with open(fp_dqn, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow({**dqn_stats, "P": p})
