# -*- coding: utf-8 -*-
"""This script trains a supervised model for move prediction; if you have collected replays, you can build a model
to try to play like humans.
"""
import os.path
import time

import torch
import torch.nn as nn
import torch.optim as optim

import wandb
from elitefurretai.model_utils import Embedder, ModelBattleOrder
from elitefurretai.model_utils.training_generator import batch_generator


class PolicyNetwork(nn.Module):
    def __init__(self, input_size, cfg):
        super(PolicyNetwork, self).__init__()

        self._config = cfg

        # Construct network based on layers icfg
        layers = []
        for i, layer in enumerate(cfg["layers"]):
            if i == 0:
                layers.append(nn.Linear(input_size, layer))
            else:
                layers.append(nn.Linear(cfg["layers"][i - 1], layer))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(cfg["layers"][-1], ModelBattleOrder.action_space()))
        layers.append(nn.Softmax(dim=-1))
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
        y_tensor = torch.LongTensor(y_train)

        # Use Cross Entropy Loss for policy learning
        criterion = nn.HuberLoss()
        if self._config["loss"] == "CrossEntropy":
            criterion = nn.CrossEntropyLoss()  # Should aim for 0.05

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

    def validate(self, batch_generator, device="cpu", v=True):
        self.network.eval()  # Set model to evaluation mode
        total_loss = 0
        total_correct = 0
        total_samples = 0

        criterion = nn.HuberLoss()
        if self._config["loss"] == "CrossEntropy":
            criterion = nn.CrossEntropyLoss()  # Should aim for 0.05

        with torch.no_grad():  # Disable gradient computation
            for inputs, targets in batch_generator:
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


def main(cfg):

    # TODO: make sure I set seeds beforehand
    # Initialize everything
    wandb.init(project="elitefurretai-forewarn", config=cfg)
    embedder = Embedder(format=cfg["format"], simple=False)
    model = PolicyNetwork(input_size=embedder.embedding_size, cfg=cfg)
    wandb.watch(model, log="all", log_freq=10)

    # For tracking progress through iterations
    start, last, batch_num = time.time(), 0, 0

    # Print training run
    est_steps_per_battle = 8.8
    est_time_per_battle = 0.051
    total_battles = int(
        sum(1 for entry in os.scandir(config["data"]) if entry.is_file())
    ) * (cfg["train_slice"][1] - cfg["train_slice"][0])
    total_steps = total_battles * 2 * est_steps_per_battle * cfg["epochs"]
    hours = total_battles * est_time_per_battle * cfg["epochs"] // 3600
    minutes = (total_battles * est_time_per_battle * cfg["epochs"] % 3600) // 60
    params = sum(p.numel() for p in model.parameters())
    print("===== Training Stats =====")
    print(f"Total Battles:  {total_battles}")
    print(f"Train Slice:    {cfg['train_slice']}")
    print(f"Val Slice:    {cfg['val_slice']}")
    print(f"Estimate Steps: {int(total_steps)}")
    print(f"Estimate Time:  {hours}h {minutes}m")
    print(f"Layers:         {cfg['layers']}")
    print(f"Num Parameters: {params}")
    print(f"Batch Size:     {cfg['batch_size']}")
    print(f"Epochs:         {cfg['epochs']}")
    print(f"Learning Rate:  {cfg['learning_rate']}")
    print(f"Optimizer:      {cfg['optimizer']}")
    print(f"Loss:           {cfg['loss']}")
    print(f"Scheduler:      {cfg['scheduler']}")
    print()

    # First I want to normalize the data by taking a sample of 10000
    # mean = train_data.mean(dim=0)
    # std = train_data.std(dim=0)
    # train_normalized = (train_data - mean) / std
    pass

    # Iterate through datasets by batches
    print("\rProcessed 0 steps in 0 secs; (0% done)...", end="")
    for epoch in range(cfg["epochs"]):
        for X, Y in batch_generator(cfg):

            # Update progress
            batch_num += 1
            steps = batch_num * cfg["batch_size"]

            # Train the model based on the batch
            loss = model.train_batch(X, Y)

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
                time_left = time.time() - start / (perc_done / 100)
                hours_left = time_left // 3600
                minutes_left = time_left // 60
                print(
                    f"\rProcessed {steps} steps in {round(time.time() - start, 2)}s; {round(perc_done, 2)}% done "
                    + f"with an estimated {hours_left}h {minutes_left}m left...",
                    end="",
                )
                last += 2

            # Checkpoint each hour
            if last % (60 * 60) == 0 and last > 0:
                avg_loss, accuracy = model.validate(batch_generator=batch_generator(cfg))

                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "loss": loss,
                    "avg_validation_loss": avg_loss,
                    "validation_accuracy": accuracy,
                }
                torch.save(
                    checkpoint,
                    f"/Users/cayman/Desktop/checkpoint_hour_{last / 60 / 60}.pth",
                )

                wandb.log(checkpoint)

        # Update scheduler
        if model.scheduler is not None:
            model.scheduler.step()

    print(f"Finished training in {round(time.time() - start, 2)} secs")

    wandb.finish()

    # Save the model
    torch.save(model.state_dict(), "/Users/cayman/Desktop/final_model.pth")
    model.validate(batch_generator=batch_generator(cfg))


if __name__ == "__main__":
    config = {
        "learning_rate": 3e-4,
        "epochs": 1,
        "loss": "CrossEntropy",
        "layers": (256,),
        "batch_size": 512,
        "optimizer": "Adam",
        "scheduler": True,
        "initializer": "He",
        "train_slice": (0, 0.98),
        "val_slice": (0.98, 1),
        "data": "/Users/cayman/Repositories/EliteFurretAI/data/battles/gen9vgc2023regulationc_raw",
        "format": "gen9vgc2024regc",
    }
    main(config)
