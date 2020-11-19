import csv

from imbDRL.data import get_train_test_val, load_creditcard
from imbDRL.environments import ClassifyEnv
from imbDRL.examples.ddqn.example_classes import TrainCustomDDQN
from imbDRL.metrics import classification_metrics, network_predictions
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tqdm import tqdm

episodes = 50_000  # Total number of episodes
warmup_episodes = 60_000  # Amount of warmup steps to collect data with random policy
memory_length = warmup_episodes  # Max length of the Replay Memory

target_model_update = episodes // 100  # Period to overwrite the target Q-network with the default Q-network
target_update_tau = 0.01  # Soften the target model update

conv_layers = None  # Convolutional layers
dense_layers = (256, 256, )  # Dense layers
dropout_layers = (0.2, 0.2, )  # Dropout layers

lr = 0.001  # Learning rate
gamma = 0.0  # Discount factor
min_epsilon = 0.05  # Minimal and final chance of choosing random action
decay_episodes = 25_000  # Number of episodes to decay from 1.0 to `min_epsilon`

imb_rate = 0.001729  # Imbalance rate
min_class = [1]  # Labels of the minority classes
maj_class = [0]  # Labels of the majority classes
X_train, y_train, X_test, y_test = load_creditcard(normalization=True, fp_train="./data/credit0.csv", fp_test="./data/credit1.csv")

df_dqn = "./results/creditcardfraud/dqn.csv"
fieldnames = ("Gmean", "F1", "Precision", "Recall", "TP", "TN", "FP", "FN")

# Create empty files
with open(df_dqn, "w", newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

# Run the model ten times
for _ in tqdm(range(10)):
    # New train-test split
    X_train, y_train, X_test, y_test, X_val, y_val = get_train_test_val(
        X_train, y_train, X_test, y_test, min_class, maj_class, val_frac=0.2, print_stats=False)

    # Change Python environment to TF environment
    train_env = TFPyEnvironment(ClassifyEnv(X_train, y_train, imb_rate))

    model = TrainCustomDDQN(episodes, warmup_episodes, lr, gamma, min_epsilon, decay_episodes,
                            target_model_update=target_model_update, target_update_tau=target_update_tau, progressbar=False)

    model.compile_model(train_env, conv_layers, dense_layers, dropout_layers)
    model.train(X_val, y_val)

    # Predictions of model for `X_test`
    y_pred = network_predictions(model.agent._target_q_network, X_test)
    dqn_stats = classification_metrics(y_test, y_pred)

    # Write current baseline run to `fp_baseline`
    with open(df_dqn, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(dqn_stats)
