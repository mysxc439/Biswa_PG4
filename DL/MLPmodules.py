try:
    import uuid, csv, os
    import time, numpy as np, pandas as pd
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        mean_squared_error, mean_absolute_error, r2_score
    )
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from tensorflow.keras import Sequential, layers, regularizers
    from tensorflow.keras.optimizers import Adam, AdamW
    from tensorflow.keras.utils import to_categorical

except Exception as e:
    print(e)

class MLPConfig:
    def __init__(
        self,
        task,                 # "classification" or "regression"
        hidden_layers,
        activation,           # "relu", "tanh", "leakyrelu"
        output_activation,    # "softmax", "sigmoid", "linear"
        optimizer,            # "Adam", "AdamW"
        learning_rate, dropout, l2, loss, metric,
        batch_size, epochs, test_size, random_state, scaling,
        name =""  # specific notes
    ):
        self.id = str(uuid.uuid4())[:8]
        self.task = task
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.output_activation = output_activation
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.dropout = dropout
        self.l2 = l2
        self.test_size = test_size
        self.random_state = random_state
        self.scaling = scaling
        self.loss = loss
        self.metric = metric
        self.name = name

    def to_dict(self):
        return self.__dict__

def get_activation(name):
    if name.lower() == "leakyrelu":
        return layers.LeakyReLU(alpha=0.01)
    else:
        return name

def get_optimizer(name, lr):
    if name.lower() == "adam":
        return Adam(learning_rate=lr)
    elif name.lower() == "adamw":
        return AdamW(learning_rate=lr)
    else:
        raise ValueError("Unsupported optimizer")

class MLPModel:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.train_time = None

    def build(self, input_dim, output_dim):
        model = Sequential()

        for i, units in enumerate(self.config.hidden_layers):
            act = get_activation(self.config.activation)
            if i == 0:
                model.add(layers.Dense(
                    units,
                    input_dim=input_dim,
                    kernel_regularizer=regularizers.l2(self.config.l2)
                ))
            else:
                model.add(layers.Dense(
                    units,
                    kernel_regularizer=regularizers.l2(self.config.l2)
                ))
            model.add(act)
            if self.config.dropout > 0:
                model.add(layers.Dropout(self.config.dropout))

        model.add(layers.Dense(output_dim, activation=self.config.output_activation))

        opt = get_optimizer(self.config.optimizer, self.config.learning_rate)

        model.compile(
            optimizer=opt,
            loss=self.config.loss,
            metrics=[self.config.metric]
        )

        self.model = model

    def train(self, X_train, y_train):
        start = time.time()
        self.model.fit(
            X_train, y_train,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            verbose=0
        )
        self.train_time = time.time() - start

    def predict(self, X):
        return self.model.predict(X, verbose=0)

    def evaluate(self, X_test, y_test):
        preds = self.predict(X_test)

        if self.config.task == "classification":
            y_pred = np.argmax(preds, axis=1)
            return {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average="macro"),
                "recall": recall_score(y_test, y_pred, average="macro"),
                "f1": f1_score(y_test, y_pred, average="macro"),
                "train_time": self.train_time
            }
        else:
            preds = preds.flatten()
            return {
                "mse": mean_squared_error(y_test, preds),
                "mae": mean_absolute_error(y_test, preds),
                "r2": r2_score(y_test, preds),
                "train_time": self.train_time
            }



def prepare_data(X, y, configs, n_classes=None):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=configs.test_size,
        random_state=configs.random_state,
        stratify=y if configs.task == "classification" else None
    )

    if configs.scaling == "standard":
        scaler = StandardScaler()
    elif configs.scaling == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = None

    if scaler:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    if configs.task == "classification":
        y_train_cat = to_categorical(y_train, num_classes=n_classes)
        return X_train, X_test, y_train_cat, y_test
    else:
        return X_train, X_test, y_train, y_test
    
class Experiment:
    def __init__(self, X, y, configs, n_classes=None, result_file="results.csv"):
        self.X = X
        self.y = y
        self.configs = configs
        self.n_classes = n_classes
        self.logger = ExperimentLogger(result_file)

    def run(self):
        for cfg in self.configs:
            print(f"Running {cfg.task.upper()} | Config {cfg.id}")

            data = prepare_data(self.X, self.y, cfg, self.n_classes)

            if cfg.task == "classification":
                X_tr, X_te, y_tr_cat, y_te = data
                output_dim = self.n_classes
                y_train = y_tr_cat
            else:
                X_tr, X_te, y_train, y_te = data
                output_dim = 1

            model = MLPModel(cfg)
            model.build(input_dim=X_tr.shape[1], output_dim=output_dim)
            model.train(X_tr, y_train)
            metrics = model.evaluate(X_te, y_te)

            self.logger.log(cfg, metrics)
            print("Done →", metrics)

class ExperimentLogger:
    def __init__(self, file_path):
        self.file_path = file_path
        self.exists = os.path.exists(file_path)

    def log(self, config, metrics):
        row = {**config.to_dict(), **metrics}

        with open(self.file_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not self.exists:
                writer.writeheader()
                self.exists = True
            writer.writerow(row)