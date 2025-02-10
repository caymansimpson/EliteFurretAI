# -*- coding: utf-8 -*-
"""This script trains a supervised model for move prediction; if you have collected replays, you can build a model
to try to play like humans.
"""
import os.path
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import wandb
from elitefurretai.model_utils import Embedder, ModelBattleOrder, BattleDataset


class PolicyNetwork(nn.Module):
    def __init__(self, input_size, cfg):
        super(PolicyNetwork, self).__init__()

        self._config = cfg

        # Construct network based on layers in cfg and whether we have batch_nrom
        layers = []
        for i, layer in enumerate(cfg["layers"]):
            if i == 0:
                layers.append(nn.Linear(input_size, layer))
            else:
                layers.append(nn.Linear(cfg["layers"][i - 1], layer))

            if cfg["batch_norm"]:
                layers.append(nn.BatchNorm1d(layer))

            layers.append(nn.ReLU())

        # Output layer
        last_layer_size = ModelBattleOrder.action_space() if cfg["label"] == "order" else 1

        layers.append(nn.Linear(cfg["layers"][-1], last_layer_size))
        self.network = nn.Sequential(*layers)

        if self._config["optimizer"] == "SGD":
            self.optimizer = optim.SGD(self.parameters(), lr=self._config["learning_rate"])
        elif self._config["optimizer"] == "RMSprop":
            self.optimizer = optim.RMSprop(
                self.parameters(), lr=self._config["learning_rate"]
            )
        elif self._config["optimizer"] == "Adam":
            self.optimizer = optim.Adam(
                self.parameters(), lr=self._config["learning_rate"]
            )
        else:
            raise ValueError("Unknown optimizer: " + self._config["optimizer"])

        self.scheduler = None
        if cfg["scheduler"]:
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=0.1
            )

        # Assuming 'model' is your neural network module
        for m in self.network.modules():
            if isinstance(m, nn.Linear) and cfg["initializer"] == "He":
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            elif isinstance(m, nn.Linear) and cfg["initializer"] == "Xavier":
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        return self.network(x)

    def train_batch(
        self,
        X_train,
        y_train,
    ):
        # Convert numpy arrays to PyTorch tensors
        X_tensor = torch.FloatTensor(X_train)
        y_tensor = torch.FloatTensor(y_train)

        y_tensor = y_tensor.view(-1, 1)  # Add this line to reshape to [batch_size, 1]

        # Use Cross Entropy Loss for policy learning
        criterion = nn.HuberLoss()
        if self._config["loss"] == "CrossEntropy":
            criterion = nn.CrossEntropyLoss()  # Should aim for 0.05
        elif self._config["loss"] == "BCE":
            criterion = nn.BCEWithLogitsLoss()

        # Zero gradients
        self.optimizer.zero_grad()

        # Forward pass
        policy_outputs = self(X_tensor)

        # Compute loss
        loss = criterion(policy_outputs, y_tensor)

        # Backward pass
        loss.backward()

        # Optimize
        self.optimizer.step()

        return loss.item()

    def validate(self, data_loader: DataLoader, device="cpu", v=True):
        self.network.eval()  # Set model to evaluation mode
        total_loss = 0
        total_correct = 0
        total_samples = 0

        criterion = nn.HuberLoss()
        if self._config["loss"] == "CrossEntropy":
            criterion = nn.CrossEntropyLoss()  # Should aim for 0.05

        with torch.no_grad():  # Disable gradient computation
            for inputs, targets in data_loader:
                # Move to device
                inputs = inputs.to(device)
                targets = targets.to(device)

                # Forward pass
                outputs = self.network(inputs)
                loss = criterion(outputs, targets)

                # Accumulate loss
                total_loss += loss.item() * inputs.size(0)

                # Calculate accuracy (for classification)
                predictions = outputs.argmax(dim=1)  # or threshold for binary
                total_correct += (predictions == targets).sum().item()
                total_samples += inputs.size(0)

        # Calculate averages
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        if v:
            print(f"\nValidation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

        self.network.train()

        return avg_loss, accuracy


class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.model = nn.Linear(input_dim, 1)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.scheduler = None

    def forward(self, x):
        outputs = torch.sigmoid(self.model(x))
        return outputs

    def train_batch(
        self,
        X_train,
        y_train,
    ):
        self.optimizer.zero_grad()
        outputs = self.model(X_train)
        loss = self.criterion(outputs, y_train.view(-1, 1))
        loss.backward()
        self.optimizer.step()
        return loss

    def validate(self, data_loader: DataLoader, device="cpu", v=True):
        return None, None


def main(cfg):

    desktop_path = "/Users/cayman/Desktop"
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed(cfg["seed"])

    # Initialize everything
    wandb.init(project="elitefurretai-perish", config=cfg)
    embedder = Embedder(format=cfg["format"], simple=False)
    model = PolicyNetwork(input_size=embedder.embedding_size, cfg=cfg)
    if cfg["model"] == "LR":
        model = LogisticRegression(embedder.embedding_size)
    wandb.watch(model, log="all", log_freq=10)
    files = sorted(map(lambda x: os.path.join(sys.argv[1], x), os.listdir(sys.argv[1])))

    # For tracking progress through iterations
    start, last, batch_num, steps = time.time(), 0, 0, 0

    # Print training run
    est_steps_per_battle = 8.8
    total_battles = len(files) * (cfg["train_slice"][1] - cfg["train_slice"][0])
    total_steps = total_battles * 2 * est_steps_per_battle * cfg["epochs"]
    cfg["num_params"] = sum(p.numel() for p in model.parameters())
    cfg["estimated_training_steps"] = int(total_steps)
    print("===== Training Stats =====")
    for key, value in cfg.items():
        print(f"{key:<15} {value}")  # Left-align key with 15 spaces
    print()

    # Generate normalizations
    means, stds = [0.0] * embedder.embedding_size, [1.0] * embedder.embedding_size
    if cfg["normalization"]:
        # means, stds = BattleDataset.generate_normalizations(
        #     files[int(cfg["normalization"][0] * len(files)) : int(cfg["normalization"][1] * len(files))],
        #     cfg["format"],
        #     bd_eligibility_func=BattleData.is_valid_for_supervised_learning,
        #     verbose=True,
        # )

        # torch.save(means, os.path.join(desktop_path, "means.pt"))
        # torch.save(stds, os.path.join(desktop_path, "stds.pt"))
        means = torch.load(os.path.join(desktop_path, "means.pt"), weights_only=False)
        stds = torch.load(os.path.join(desktop_path, "stds.pt"), weights_only=False)

    # Generate dataset
    print("Generating Dataset...")
    dataset = BattleDataset(
        files=files[int(cfg["train_slice"][0] * len(files)) : int(cfg["train_slice"][1] * len(files))],
        format=cfg["format"],
        label_type=cfg["label"],
        bd_eligibility_func=lambda x: x.is_valid_for_supervised_learning,
        means=means,
        stds=stds
    )

    # Generate Data Loader
    data_loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        num_workers=min(os.cpu_count() or 1, 4),
        shuffle=True
    )
    print("Finished!")

    # Iterate through datasets by batches
    print("\rProcessed 0 steps in 0 secs; (0% done)...", end="")
    for epoch in range(cfg["epochs"]):
        for batch_X, batch_Y in data_loader:

            # Update progress
            batch_num += 1
            steps += len(batch_X)

            # Train the model based on the batch
            loss = model.train_batch(batch_X, batch_Y)

            # Log metrics
            wandb.log({"batch_loss": loss, "step": steps, "epoch": epoch})
            for name, param in model.named_parameters():
                if param.grad is not None:
                    wandb.log(
                        {
                            f"mean_gradients/{name}": param.grad.mean(),
                            f"std_gradients/{name}": param.grad.std(),
                        }
                    )

            # After each 1s, print
            if time.time() - start > last:
                perc_done = 100 * steps / total_steps
                time_left = (time.time() - start) / (perc_done / 100)
                hours_left = time_left // 3600
                minutes_left = time_left // 60
                print(
                    f"\rProcessed {steps} steps in {round(time.time() - start, 2)}s; {round(perc_done, 2)}% done "
                    f"with an estimated {hours_left}h {minutes_left}m left...",
                    end="",
                )
                last += 1

            # Checkpoint each hour
            if last % (60 * 60) == 0 and last > 0:

                # Evaluate Model by creating Dataset/Loader
                eval_dataset = BattleDataset(
                    files=files[int(cfg["val_slice"][0] * len(files)) : int(cfg["val_slice"][1] * len(files))],
                    format=cfg["format"],
                    label_type=cfg["label"],
                    bd_eligibility_func=lambda x: x.is_valid_for_supervised_learning,
                    means=means,
                    stds=stds
                )
                eval_data_loader = DataLoader(
                    eval_dataset,
                    batch_size=cfg["batch_size"],
                    num_workers=min(os.cpu_count() or 1, 4),
                )
                avg_loss, accuracy = model.validate(eval_data_loader)

                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "loss": loss,
                    "avg_validation_loss": avg_loss,
                    "validation_accuracy": accuracy,
                }
                torch.save(
                    checkpoint,
                    f"{desktop_path}/checkpoint_hour_{int(last / 60 / 60)}.pth",
                )

                wandb.log(checkpoint)

        # Update scheduler
        if model.scheduler is not None:
            model.scheduler.step()

    # Finish up
    total_time = (time.time() - start)
    hours = total_time // 3600
    mins = total_time // 60
    secs = total_time % 60
    print(f"Finished training in {hours}h {mins}m {secs}s")
    wandb.finish()

    # Save the model
    torch.save(model.state_dict(), f"{desktop_path}/final_model.pth")

    # Evaluate Model by creating Dataset/Loader
    eval_dataset = BattleDataset(
        files=files[int(cfg["val_slice"][0] * len(files)) : int(cfg["val_slice"][1] * len(files))],
        format=cfg["format"],
        label_type=cfg["label"],
        bd_eligibility_func=lambda x: x.is_valid_for_supervised_learning,
        means=means,
        stds=stds
    )
    eval_data_loader = DataLoader(
        eval_dataset,
        batch_size=cfg["batch_size"],
        num_workers=min(os.cpu_count() or 1, 4),
    )
    model.validate(eval_data_loader)


if __name__ == "__main__":
    config = {
        "model": "LR",
        "learning_rate": 1e-3,
        "epochs": 1,
        "loss": "BCE",
        "layers": (256,),
        "batch_size": 512,
        "optimizer": "Adam",
        "scheduler": True,
        "batch_norm": True,
        "initializer": "He",
        "train_slice": (0.01, 0.1),
        "val_slice": (0.98, 1),
        "data": sys.argv[1],
        "seed": 21,
        "format": "gen9vgc2024regc",
        "label": "win",
        "normalization": (0, .01),
    }
    main(config)
