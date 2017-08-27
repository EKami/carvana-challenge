import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
import nn.tools as tools

import nn.losses as losses
import numpy as np


class CarvanaClassifier:
    def __init__(self, net, train_loader: DataLoader, valid_loader: DataLoader):
        self.net = net
        self.valid_loader = valid_loader
        self.train_loader = train_loader
        self.use_cuda = torch.cuda.is_available()

    def _criterion(self, logits, labels):
        # l = BCELoss2d()(logits, labels)
        l = losses.BCELoss2d().forward(logits, labels) + losses.SoftDiceLoss().forward(logits, labels)
        return l

    def _evaluate_predictions(self, threshold):
        valid_dataset = self.valid_loader.dataset

        num = len(valid_dataset)
        height, width = self.train_loader.dataset.img_resize
        predictions = np.zeros((num, 2 * height, 2 * width), np.float32)

        test_acc = 0
        test_loss = 0
        test_num = 0
        #for ind, (inputs, masks) in enumerate(self.train_loader, 0):
        # for ind, (images, masks, indices) in enumerate(self.valid_loader, 0):
        #     if self.use_cuda:
        #         images = images.cuda()
        #         masks = masks.cuda()
        #
        #     images = Variable(images, volatile=True)
        #     masks = Variable(masks, volatile=True)
        #
        #     # forward
        #     logits = self.net(images)
        #     probs = F.sigmoid(logits)
        #     pred = (probs > threshold).float()
        #
        #     loss = self._criterion(logits, masks)
        #     acc = losses.dice_loss(pred, masks)
        #
        #     batch_size = len(indices)
        #     test_num += batch_size
        #     test_loss += batch_size * loss.data[0]
        #     test_acc += batch_size * acc.data[0]
        #     start = test_num - batch_size
        #     end = test_num
        #     predictions[start:end] = probs.data.cpu().numpy().reshape(-1, 2 * height, 2 * width)
        #
        # assert (test_num == len(self.valid_loader.sampler))
        #
        # test_loss = test_loss / test_num
        # test_acc = test_acc / test_num

        return predictions, test_loss, test_acc

    def train(self, epochs, threshold=0.5):
        self.net.train()

        if self.use_cuda:
            self.net.cuda()
        optimizer = optim.SGD(self.net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)

        smooth_loss = 0.0
        smooth_acc = 0.0
        train_loss = np.nan
        train_acc = np.nan
        it_print = 1
        it_smooth = 20

        for i_epoch, epoch in enumerate(range(epochs)):

            sum_smooth_loss = 0.0
            sum_smooth_acc = 0.0
            sum = 0
            num_its = len(self.train_loader)
            for ind, (inputs, masks) in enumerate(self.train_loader, 0):

                if self.use_cuda:
                    inputs = inputs.cuda()
                    masks = masks.cuda()
                inputs, masks = Variable(inputs), Variable(masks)

                # forward
                logits = self.net.forward(inputs)
                probs = F.sigmoid(logits)
                pred = (probs > threshold).float()

                # backward + optimize
                loss = self._criterion(logits, masks)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # print statistics
                acc = losses.dice_loss(pred, masks)
                lr = tools.get_learning_rate(optimizer)[0]

                sum_smooth_loss += loss.data[0]
                sum_smooth_acc += acc.data[0]
                sum += 1

                if ind % it_smooth == 0:
                    smooth_loss = sum_smooth_loss / sum
                    smooth_acc = sum_smooth_acc / sum
                    sum_smooth_loss = 0.0
                    sum_smooth_acc = 0.0
                    sum = 0

                valid_predictions, valid_loss, valid_acc = self._evaluate_predictions(threshold)

                if ind % it_print == 0 or ind == num_its - 1:
                    train_acc = acc.data[0]
                    train_loss = loss.data[0]
                    print("Epochs {}, batch = {}, train_loss= {}, train_acc = {}, "
                          "valid_loss = {} valid_acc = {}".
                          format(i_epoch, ind, train_loss, train_acc, valid_loss, valid_acc))
                    # print('\r%5.1f   %5d    %0.4f   |  %0.4f  %0.4f | %0.4f  %6.4f | ... ' %
                    #       (epoch + (ind + 1) / num_its, ind + 1, lr, smooth_loss, smooth_acc, train_loss, train_acc),
                    #       end='', flush=True)

    def predict(self, test_loader):
        self.net.eval()
        # probs = predict_in_blocks(net, test_loader, block_size=32000)