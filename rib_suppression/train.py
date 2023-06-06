import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional
from loguru import logger
from sklearn.model_selection import train_test_split
from torch import optim, nn
from torch.utils.data import DataLoader

from models.u_net import UNet
from datasets.rib_shadows import RibShadowsDataset
from utils.average_meter import AverageMeter, dice_coeff


def validate_trained_net_path(path):
    if not os.path.isfile(path) or not path.endswith('.pth'):
        logger.info(f'Invalid or non-existent path to trained net - {path}')
        return False
    return True


def check_training_config(training_config):
    training_config = training_config or {}

    dataset_path = training_config.get('dataset_path')
    if dataset_path is None:
        raise ValueError('dataset path is required')

    if not os.path.exists(dataset_path):
        raise ValueError('dataset path does not exist')

    training_config['lr'] = training_config.get('lr', 1e-4)
    training_config['weight_decay'] = training_config.get('weight_decay', 1e-4)
    training_config['momentum'] = training_config.get('momentum', 0.9)
    training_config['lr-patience'] = training_config.get('lr-patience', 100)
    training_config['epoch_num'] = training_config.get('epoch_num', 5)

    path_to_trained_net = training_config.get('path_to_trained_net')
    if not path_to_trained_net or not validate_trained_net_path(path_to_trained_net):
        path_to_trained_net = 'trained_models/u_net_rib_shadows.pth'
        logger.info(
            f"path to trained net is not valid, so net will be created"
            f"and path to trained net set to default: {path_to_trained_net}")

    training_config['path_to_trained_net'] = path_to_trained_net

    return training_config


class TrainingNet:
    def __init__(self, net: nn.Module = None, training_config=None):
        self._test_loader = None
        self._test_dataset = None
        self._train_loader = None
        self._train_dataset = None
        self._training_config = check_training_config(training_config)
        self._net = self.load_net(net)
        self._dataset = RibShadowsDataset(self._training_config['dataset_path'])
        self._criterion = torch.nn.BCELoss(reduction='sum')
        self._optimizer = optim.Adam(self._net.parameters(), lr=2 * self._training_config['lr'],
                                     weight_decay=self._training_config['weight_decay'],
                                     betas=(self._training_config['momentum'], 0.999))
        self._scheduler = optim.lr_scheduler.ReduceLROnPlateau(self._optimizer, 'min',
                                                               patience=self._training_config['lr-patience'],
                                                               min_lr=1e-10,
                                                               verbose=True)

    def __call__(self, is_visualize: bool = False) -> nn.Module:
        return self.training(is_visualize)

    def load_net(self, net):
        if net is None:
            net = UNet()
            if os.path.isfile(self._training_config.get('path_to_trained_net')):
                logger.info(
                    'trained net with path %s successfully loaded' % self._training_config.get('path_to_trained_net'))
                net.load_state_dict(torch.load(self._training_config.get('path_to_trained_net')))
        else:
            logger.info('net successfully loaded')
        return net

    def training(self, is_visualize: bool = False) -> nn.Module:
        x_train, x_test, y_train, y_test = (
            train_test_split(self._dataset[:][0], self._dataset[:][1], test_size=0.25, random_state=42))
        x_train = torch.stack(x_train)
        x_test = torch.stack(x_test)
        y_train = torch.stack(y_train)
        y_test = torch.stack(y_test)

        self._train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
        self._train_loader = DataLoader(self._train_dataset, batch_size=1, shuffle=True)
        self._test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
        self._test_loader = DataLoader(self._test_dataset, batch_size=1, shuffle=True)

        logger.info('net starts training')
        self._net.train()

        train_loss_history = []
        test_loss_history = []

        for epoch in range(1, self._training_config['epoch_num'] + 1):
            train_dice_score = []
            test_dice_score = []

            self._train(epoch, train_dice_score, train_loss_history)
            test_loss = self._test(epoch, test_dice_score, test_loss_history, is_visualize)
            self._scheduler.step(test_loss)
        plt.plot(train_loss_history)
        plt.xlabel('iter')
        plt.ylabel('train loss')
        plt.show()

        plt.plot(test_loss_history)
        plt.xlabel('epoch')
        plt.ylabel('test loss')
        plt.show()
        torch.save(self._net.state_dict(), '%s.pth' % self._training_config.get('path_to_trained_net').split('.')[0])
        return self._net

    def _train(self, epoch, dice_score, loss_history):
        train_loss = AverageMeter()
        curr_iter = (epoch - 1) * len(self._train_loader)
        for i, data in enumerate(self._train_loader):
            inp, label = data

            self._optimizer.zero_grad()
            out = self._net(inp)

            label = label.expand_as(out)

            loss = self._criterion(out, label)
            loss.backward()
            self._optimizer.step()
            train_loss.update(loss.item())

            dice_score.append(dice_coeff(out, label))
            loss_history.append(loss.item())
            curr_iter += 1
            print('[epoch %d], [iter %d / %d], [train loss %.5f], [train dice score %.5f]' % (
                epoch, curr_iter, self._training_config['epoch_num'] * len(self._train_loader), train_loss.avg,
                np.mean(dice_score)
            ))

    def _test(self, epoch, dice_score, loss_history, is_visualize: bool = False):
        self._net.eval()
        test_loss = AverageMeter()
        preds_all = []
        for i, data in enumerate(self._test_loader):
            inp, label = data

            out = self._net(inp)
            label = label.expand_as(out)
            preds = out.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()

            loss = self._criterion(out, label)
            test_loss.update(loss.item())
            loss_history.append(loss.item())
            preds_all.append(preds)
            dice_score.append(dice_coeff(out, label))
            if i == len(self._test_loader) - 1 and is_visualize:
                image_target_mask = (label.detach().numpy().squeeze().squeeze() * 255)
                image_target_mask = np.where(image_target_mask > 127., 255., 0.)
                cv2.imshow('Image target mask', image_target_mask)
                image_pred_mask = (out.detach().numpy().squeeze().squeeze() * 255)
                image_pred_mask = np.where(image_pred_mask > 127., 255., 0.)
                cv2.imshow('Image pred mask', image_pred_mask)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        print('[epoch %d], [test loss %.5f], [test dice score %.5f]' % (
            epoch, test_loss.avg, np.mean(dice_score)
        ))

        self._net.train()
        return test_loss.avg
