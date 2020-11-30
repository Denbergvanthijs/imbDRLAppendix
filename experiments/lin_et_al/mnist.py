import csv

from imbDRL.data import get_train_test_val, load_image
from imbDRL.examples.ddqn.example_classes import TrainCustomDDQN
from imbDRL.metrics import classification_metrics, network_predictions
from tqdm import tqdm

episodes = 120_000  # Total episodes
warmup_episodes = 50_000  # Amount of warmup steps to collect data with random policy
memory_length = 100_000  # Max length of the Replay Memory
collect_steps_per_episode = 1
target_model_update = 10_000
target_update_tau = 1
batch_size = 32
n_step_update = 4

conv_layers = ((32, (5, 5), 2), (32, (5, 5), 2), )  # Convolutional layers
dense_layers = (256, )  # Dense layers
dropout_layers = None  # Dropout layers

lr = 0.00025  # Learning rate
gamma = 0.1  # Discount factor
min_epsilon = 0.01  # Minimal and final chance of choosing random action
decay_episodes = 100_000  # Number of episodes to decay from 1.0 to `min_epsilon`, divided by 4

imb_rate = 0.01  # Imbalance rate
min_class = [2]  # Minority classes, same setup as in original paper
maj_class = [0, 1, 3, 4, 5, 6, 7, 8, 9]  # Majority classes
X_train, y_train, X_test, y_test, = load_image("mnist")

fp_dqn = "./results/lin/mnist.csv"
fieldnames = ("Gmean", "F1", "Precision", "Recall", "TP", "TN", "FP", "FN", "P")

# Create empty files
with open(fp_dqn, "w", newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

for p in (0.01, 0.002, 0.001, 0.0005):
    # Run the model ten times
    for _ in tqdm(range(10), desc=f"Running model for imbalance ratio:{p}"):
        # New train-test split
        X_train, y_train, X_test, y_test, X_val, y_val = get_train_test_val(X_train, y_train, X_test, y_test, min_class, maj_class,
                                                                            imb_test=False, val_frac=0.1, print_stats=False)

        model = TrainCustomDDQN(episodes, warmup_episodes, lr, gamma, min_epsilon, decay_episodes, target_update_tau=target_update_tau,
                                collect_steps_per_episode=collect_steps_per_episode, target_model_update=target_model_update,
                                n_step_update=n_step_update, batch_size=batch_size, memory_length=memory_length, progressbar=False)

        model.compile_model(X_train, y_train, imb_rate, conv_layers, dense_layers, dropout_layers)
        model.train(X_val, y_val, "Gmean")

        # Predictions of model for `X_test`
        y_pred = network_predictions(model.agent._target_q_network, X_test)
        dqn_stats = classification_metrics(y_test, y_pred)

        # Write current DQN run to `fp_dqn`
        with open(fp_dqn, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow({**dqn_stats, "P": p})