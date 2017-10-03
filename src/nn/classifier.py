import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
import nn.tools as tools
from tqdm import tqdm
from collections import OrderedDict

import nn.losses as losses_utils
import helpers


class CarvanaClassifier:
    def __init__(self, net, max_epochs):
        """
        The classifier for carvana used for training and launching predictions
        Args:
            net (nn.Module): The neural net module containing the definition of your model
            max_epochs (int): The maximum number of epochs on which the model will train
        """
        self.net = net
        self.max_epochs = max_epochs
        self.epoch_counter = 0
        self.use_cuda = torch.cuda.is_available()

    def restore_model(self, model_path):
        """
            Restore a model parameters from the one given in argument
        Args:
            model_path (str): The path to the model to restore

        """
        self.net.load_state_dict(torch.load(model_path))

    def _criterion(self, logits, labels):
        return losses_utils.BinaryCrossEntropyLoss2d().forward(logits, labels) + \
            losses_utils.SoftDiceLoss().forward(logits, labels)

    def _validate_epoch(self, valid_loader, threshold):
        losses = tools.AverageMeter()
        dice_coeffs = tools.AverageMeter()

        it_count = len(valid_loader)
        batch_size = valid_loader.batch_size

        images = None  # To save the last images batch
        targets = None  # To save the last target batch
        preds = None  # To save the last prediction batch
        with tqdm(total=it_count, desc="Validating", leave=False) as pbar:
            for ind, (images, targets) in enumerate(valid_loader):
                if self.use_cuda:
                    images = images.cuda()
                    targets = targets.cuda()

                # Volatile because we are in pure inference mode
                # http://pytorch.org/docs/master/notes/autograd.html#volatile
                images = Variable(images, volatile=True)
                targets = Variable(targets, volatile=True)

                # forward
                logits = self.net(images)
                probs = F.sigmoid(logits)
                preds = (probs > threshold).float()

                loss = self._criterion(logits, targets)
                acc = losses_utils.dice_coeff(preds, targets)
                losses.update(loss.data[0], batch_size)
                dice_coeffs.update(acc.data[0], batch_size)
                pbar.update(1)

        return losses.avg, dice_coeffs.avg, images, targets, preds

    def _train_epoch(self, train_loader, optimizer, threshold):
        losses = tools.AverageMeter()
        dice_coeffs = tools.AverageMeter()

        # Total training files count / batch_size
        batch_size = train_loader.batch_size
        it_count = len(train_loader)
        with tqdm(total=it_count,
                  desc="Epochs {}/{}".format(self.epoch_counter + 1, self.max_epochs),
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{remaining}{postfix}]'
                  ) as pbar:
            for ind, (inputs, target) in enumerate(train_loader):

                if self.use_cuda:
                    inputs = inputs.cuda()
                    target = target.cuda()
                inputs, target = Variable(inputs), Variable(target)

                # forward
                logits = self.net.forward(inputs)
                probs = F.sigmoid(logits)
                pred = (probs > threshold).float()

                # backward + optimize
                loss = self._criterion(logits, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # print statistics
                acc = losses_utils.dice_coeff(pred, target)

                losses.update(loss.data[0], batch_size)
                dice_coeffs.update(acc.data[0], batch_size)

                # Update pbar
                pbar.set_postfix(OrderedDict(loss='{0:1.5f}'.format(loss.data[0]),
                                             dice_coeff='{0:1.5f}'.format(acc.data[0])))
                pbar.update(1)
        return losses.avg, dice_coeffs.avg

    @helpers.st_time(show_func_name=False)
    def _run_epoch(self, train_loader: DataLoader, valid_loader: DataLoader,
                   optimizer, threshold=0.5, callbacks=None):
        # switch to train mode
        self.net.train()

        # Run a train pass on the current epoch
        train_loss, train_dice_coeff = self._train_epoch(train_loader, optimizer, threshold)

        # switch to evaluate mode
        self.net.eval()

        # Run the validation pass
        val_loss, val_dice_coeff, last_images, last_targets, last_preds = \
            self._validate_epoch(valid_loader, threshold)

        # If there are callback call their __call__ method and pass in some arguments
        if callbacks:
            for cb in callbacks:
                cb(step_name="epoch",
                   net=self.net,
                   last_val_batch=(last_images, last_targets, last_preds),
                   epoch_id=self.epoch_counter + 1,
                   train_loss=train_loss, train_dice_coeff=train_dice_coeff,
                   val_loss=val_loss, val_dice_coeff=val_dice_coeff
                   )
        print("train_loss = {:03f}, train_dice_coeff = {:03f}\n"
              "val_loss   = {:03f}, val_dice_coeff   = {:03f}"
              .format(train_loss, train_dice_coeff, val_loss, val_dice_coeff))
        self.epoch_counter += 1

    def train(self, train_loader: DataLoader, valid_loader: DataLoader,
              optimizer, epochs, threshold=0.5, callbacks=None):
        """
            Trains the neural net
        Args:
            train_loader (DataLoader): The Dataloader for training
            valid_loader (DataLoader): The Dataloader for validation
            optimizer (Optimizer): The nn optimizer
            epochs (int): number of epochs
            threshold (float): The threshold used to consider the mask present or not
            callbacks (list): List of callbacks functions to call at each epoch
        Returns:
            str, None: The path where the model was saved, or None if it wasn't saved
        """
        if self.use_cuda:
            self.net.cuda()

        for epoch in range(epochs):
            self._run_epoch(train_loader, valid_loader, optimizer, threshold, callbacks)

        # If there are callback call their __call__ method and pass in some arguments
        if callbacks:
            for cb in callbacks:
                cb(step_name="train",
                   net=self.net,
                   epoch_id=self.epoch_counter + 1,
                   )

    def predict(self, test_loader, callbacks=None):
        """
            Launch the prediction on the given loader and pass
            each predictions to the given callbacks.
        Args:
            test_loader (DataLoader): The loader containing the test dataset
            callbacks (list): List of callbacks functions to call at prediction pass
        """
        # Switch to evaluation mode
        self.net.eval()

        it_count = len(test_loader)

        with tqdm(total=it_count, desc="Classifying") as pbar:
            for ind, (images, files_name) in enumerate(test_loader):
                if self.use_cuda:
                    images = images.cuda()

                images = Variable(images, volatile=True)

                # forward
                logits = self.net(images)
                probs = F.sigmoid(logits)
                probs = probs.data.cpu().numpy()

                # If there are callback call their __call__ method and pass in some arguments
                if callbacks:
                    for cb in callbacks:
                        cb(step_name="predict",
                           net=self.net,
                           probs=probs,
                           files_name=files_name
                           )

                pbar.update(1)
