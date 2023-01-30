import numpy as np
import torch
import os
from .early_stopping import EarlyStopping


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        training_DataLoader: torch.utils.data.Dataset,
        validation_DataLoader: torch.utils.data.Dataset = None,
        lr_scheduler: torch.optim.lr_scheduler = None,
        score_function=None,
        epochs: int = 100,
        epoch: int = 0,
        notebook: bool = False,
    ):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.training_DataLoader = training_DataLoader
        self.validation_DataLoader = validation_DataLoader
        self.device = device
        self.epochs = epochs
        self.epoch = epoch
        self.notebook = notebook
        self.score_function = score_function

        self.training_loss = []
        self.training_score = []
        self.validation_loss = []
        self.validation_score = []
        self.learning_rate = []

    def run_trainer(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        early_stopping = EarlyStopping(patience=7, verbose=True)

        progressbar = trange(self.epochs, desc="Progress")
        for i in progressbar:
            """Epoch counter"""
            self.epoch += 1  # epoch counter

            """Training block"""
            self._train()

            """Validation block"""
            if self.validation_DataLoader is not None:
                self._validate()

            """Learning rate scheduler block"""
            if self.lr_scheduler is not None:
                if (
                    self.validation_DataLoader is not None
                    and self.lr_scheduler.__class__.__name__ == "ReduceLROnPlateau"
                ):
                    self.lr_scheduler.step(
                        self.validation_loss[i]
                    )  # learning rate scheduler step with validation loss
                else:
                    self.lr_scheduler.step()  # learning rate scheduler step

            early_stopping(self.validation_loss[-1], self.model)
            if early_stopping.early_stop:
                print("Early Stopping")
                break

        return self.training_loss, self.validation_loss, self.learning_rate

    def _train(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        self.model.train()  # train mode
        train_losses = []  # accumulate the losses here
        train_scores = []

        batch_iter = tqdm(
            enumerate(self.training_DataLoader),
            "Training",
            total=len(self.training_DataLoader),
            leave=False,
        )

        for i, (x, y) in batch_iter:
            input, target = (
                x.to(self.device),
                y.to(self.device),
            )  # send to device (GPU or CPU)
            self.optimizer.zero_grad()  # zerograd the parameters
            out = self.model(input)  # one forward pass
            loss = self.criterion(out, target)  # calculate loss
            loss_value = loss.item()
            train_losses.append(loss_value)

            if self.score_function is not None:
                score = self.score_function(out, target)
                score_value = score.item()
                train_scores.append(score_value)

            loss.backward()  # one backward pass
            self.optimizer.step()  # update the parameters

            if self.score_function is not None:
                batch_iter.set_description(
                    f"Training: (loss {loss_value:.4f}, score {score_value:.4f})"
                )  # update progressbar
            else:
                batch_iter.set_description(
                    f"Training: (loss {loss_value:.4f})"
                )  # update progressbar

        self.training_loss.append(np.mean(train_losses))
        if self.score_function:
            self.training_score.append(np.mean(train_scores))
        self.learning_rate.append(self.optimizer.param_groups[0]["lr"])

        batch_iter.close()

    def _validate(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        self.model.eval()  # evaluation mode
        valid_losses = []  # accumulate the losses here
        valid_scores = []
        batch_iter = tqdm(
            enumerate(self.validation_DataLoader),
            "Validation",
            total=len(self.validation_DataLoader),
            leave=False,
        )

        for i, (x, y) in batch_iter:
            input, target = (
                x.to(self.device),
                y.to(self.device),
            )  # send to device (GPU or CPU)

            with torch.no_grad():
                out = self.model(input)
                loss = self.criterion(out, target)
                if self.score_function is not None:
                    score = self.score_function(out, target)
                    score_value = score.item()
                    valid_scores.append(score_value)

                loss_value = loss.item()
                valid_losses.append(loss_value)

                if self.score_function is not None:
                    batch_iter.set_description(
                        f"Validation: (loss {loss_value:.4f}, score {score_value:.4f})"
                    )  # update progressbar
                else:
                    batch_iter.set_description(
                        f"Validation: (loss {loss_value:.4f})"
                    )  # update progressbar

        self.validation_loss.append(np.mean(valid_losses))

        if self.score_function:
            self.validation_score.append(np.mean(valid_scores))

        batch_iter.close()
    
    def save_training_metric(self, model_name: str) -> None:
        path = "./training_metrics"
        exists = os.path.exists(path)
        if not exists:
            print(f"created: {path}")
            os.makedirs(path)
        
        np.savez(f"./training_metrics/{model_name}",
            training_loss=self.training_loss,
            training_score=self.training_score,
            validation_loss=self.validation_loss,
            validation_score=self.validation_score)
