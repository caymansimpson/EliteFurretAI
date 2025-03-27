# -*- coding: utf-8 -*-
"""This script trains a supervised model for move prediction; if you have collected replays, you can build a model
to try to play like humans.
"""
import os.path
import sys
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import wandb
from elitefurretai.model_utils import Embedder, ModelBattleOrder, BattleDataset, BattleData


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

        # Use Cross Entropy Loss for policy learning
        self.criterion = nn.HuberLoss()
        if self._config["loss"] == "CrossEntropy":
            self.criterion = nn.CrossEntropyLoss()  # Should aim for 0.05
        elif self._config["loss"] == "BCE":
            self.criterion = nn.BCEWithLogitsLoss()

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
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)

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
        mask_train,
    ):

        y_train = y_train.view(-1, 1)  # Add this line to reshape to [batch_size, 1]

        # Zero gradients
        self.optimizer.zero_grad()

        # Forward pass
        policy_outputs = self(X_train)

        # Compute loss
        policy_outputs = policy_outputs[mask_train]
        y_train = y_train[mask_train].view(-1, 1).float()

        # Compute loss
        loss = self.criterion(policy_outputs, y_train)

        # Backward pass
        loss.backward()

        # Optimize
        self.optimizer.step()

        return loss.item()

    def validate(self, data_loader: DataLoader, steps: int, device="cpu", v=True):
        self.eval()
        total_loss = 0
        correct = 0
        tp, tn, fp, fn = 0, 0, 0, 0
        with torch.no_grad():
            for inputs, labels, mask in data_loader:
                inputs = inputs.flatten(start_dim=0, end_dim=1).to(device)
                labels = labels.flatten(start_dim=0, end_dim=1).to(device)
                mask = mask.flatten(start_dim=0, end_dim=1).to(device)
                outputs = self.network(inputs)
                loss = self.criterion(outputs[mask], labels[mask].view(-1, 1).float())
                total_loss += loss.item()
                predicted = torch.round(outputs)
                correct += (predicted == labels).sum().item()

                for i in range(len(labels)):
                    if not mask[i]:
                        continue
                    if labels[i] == 1 and predicted[i] == 1:
                        tp += 1
                    elif labels[i] == 1 and predicted[i] == 0:
                        fn += 1
                    elif labels[i] == 0 and predicted[i] == 1:
                        fp += 1
                    elif labels[i] == 0 and predicted[i] == 0:
                        tn += 1

        accuracy = (tp + fn) / (tp + tn + fp + fn)
        average_loss = total_loss / (tp + tn + fp + fn)

        if v:
            print(f"Validation Loss: {average_loss:.4f}")
            print(f"Validation Accuracy: {accuracy:.4f}")
            print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

        checkpoint = {
            "step": steps,
            "avg_validation_loss": average_loss,
            "validation_accuracy": accuracy,
        }
        wandb.log(checkpoint)

        self.train()

        return average_loss, accuracy


def main(cfg):
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])

    # Initialize everything
    wandb.init(project="elitefurretai-perish", config=cfg)
    embedder = Embedder(format=cfg["format"], type="full")
    model = PolicyNetwork(input_size=embedder.embedding_size, cfg=cfg)
    wandb.watch(model, log="all", log_freq=10)
    files = sorted(map(lambda x: os.path.join(sys.argv[1], x), os.listdir(sys.argv[1])))

    # Print training run
    est_steps_per_battle = 8.8
    total_battles = len(files) * (cfg["train_slice"][1] - cfg["train_slice"][0])
    total_steps = total_battles * 2 * est_steps_per_battle * cfg["epochs"]
    cfg["num_params"] = sum(p.numel() for p in model.parameters())
    cfg["est_steps"] = int(total_steps)
    print("===== Training Information =====")
    for key, value in cfg.items():
        print(f"{key:<15} {value}")  # Left-align key with 15 spaces
    print()

    # Generate normalizations
    means, stds = None, None
    if cfg["normalization"]:
        means = torch.load(os.path.join(cfg["desktop_path"], "means.pt"), weights_only=False, map_location=torch.device('cpu'))
        stds = torch.load(os.path.join(cfg["desktop_path"], "stds.pt"), weights_only=False, map_location=torch.device('cpu'))

    # Generate dataset
    print("Generating Dataset...")
    dataset = BattleDataset(
        files=files[int(cfg["train_slice"][0] * len(files)) : int(cfg["train_slice"][1] * len(files))],
        format=cfg["format"],
        label_type=cfg["label"],
        bd_eligibility_func=BattleData.is_valid_for_supervised_learning,
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
    print("Finished generating dataset!")

    # For tracking progress through training iterations
    start, last, batch_num, steps = time.time(), 0, 0, 0

    # Iterate through datasets by batches
    print("\rProcessed 0 steps in 0 secs; (0% done)...", end="")
    for epoch in range(cfg["epochs"]):
        for batch_X, batch_Y, batch_mask in data_loader:

            # Since each X, Y and mask is a battle, we need to flatten them
            batch_X = batch_X.flatten(start_dim=0, end_dim=1)
            batch_Y = batch_Y.flatten(start_dim=0, end_dim=1)
            batch_mask = batch_mask.flatten(start_dim=0, end_dim=1)

            # Update progress
            batch_num += 1
            steps += len(batch_X)

            # Train the model based on the batch
            loss = model.train_batch(batch_X, batch_Y, batch_mask)

            # Log metrics
            wandb.log({"batch_loss": loss, "step": steps, "epoch": epoch})
            for name, param in model.named_parameters():
                if param.grad is not None and param.grad.numel() > 1:
                    wandb.log(
                        {
                            f"mean_gradients/{name}": param.grad.mean(),
                            f"std_gradients/{name}": param.grad.std(),
                        }
                    )

            # After each 1s, print
            if time.time() - start > last:
                perc_done = 100 * steps / total_steps
                time_left = (time.time() - start) / (perc_done / 100) - (time.time() - start)
                hours_left = int(time_left / (60 * 60))
                minutes_left = int((time_left % (60 * 60)) / 60)
                print(
                    f"\rProcessed {steps} steps in {round(time.time() - start, 2)}s; {round(perc_done, 2)}% done "
                    f"with an estimated {hours_left}h {minutes_left}m left...",
                    end="",
                )
                last = time.time() - start + 1

        # Checkpoint each 5 epochs
        if (epoch + 1) % 5 == 0 and epoch != cfg["epochs"] - 1:

            # Evaluate Model by creating Dataset/Loader
            eval_dataset = BattleDataset(
                files=files[int(cfg["val_slice"][0] * len(files)) : int(cfg["val_slice"][1] * len(files))],
                format=cfg["format"],
                label_type=cfg["label"],
                bd_eligibility_func=BattleData.is_valid_for_supervised_learning,
                means=means,
                stds=stds
            )
            eval_data_loader = DataLoader(
                eval_dataset,
                batch_size=cfg["batch_size"],
                num_workers=min(os.cpu_count() or 1, 4),
            )
            avg_loss, accuracy = model.validate(eval_data_loader, steps=steps)

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "avg_validation_loss": avg_loss,
                "validation_accuracy": accuracy,
            }

            torch.save(
                checkpoint,
                f"{cfg['desktop_path']}/checkpoint_epoch_{epoch}.pth",
            )

        # Update scheduler
        if model.scheduler is not None:
            model.scheduler.step()

    # Finish up
    total_time = (time.time() - start)
    hours = int(total_time / (60 * 60))
    mins = int((total_time % (60 * 60)) / 60)
    secs = int(total_time % 60)
    print(f"Finished training in {hours}h {mins}m {secs}s")

    # Save the model
    torch.save(model.state_dict(), f"{cfg['desktop_path']}/final_model.pth")

    # Evaluate Model by creating Dataset/Loader
    eval_dataset = BattleDataset(
        files=files[int(cfg["val_slice"][0] * len(files)) : int(cfg["val_slice"][1] * len(files))],
        format=cfg["format"],
        label_type=cfg["label"],
        bd_eligibility_func=BattleData.is_valid_for_supervised_learning,
        means=means,
        stds=stds
    )
    eval_data_loader = DataLoader(
        eval_dataset,
        batch_size=cfg["batch_size"],
        num_workers=min(os.cpu_count() or 1, 4),
    )
    model.validate(eval_data_loader, steps=steps)
    wandb.finish()


if __name__ == "__main__":
    config = {
        "model": "NN",
        "learning_rate": 1e-3,
        "epochs": 100,
        "loss": "BCE",
        "layers": (256, 256),
        "batch_size": 512,
        "optimizer": "Adam",
        "scheduler": True,
        "batch_norm": True,
        "initializer": None,
        "train_slice": (0.0, 0.25),
        "val_slice": (0.52, 0.525),
        "data": sys.argv[1],
        "seed": 21,
        "format": "gen9vgc2024regc",
        "label": "win",
        "normalization": (0, .01),
        "desktop_path": sys.argv[1],
    }
    main(config)
