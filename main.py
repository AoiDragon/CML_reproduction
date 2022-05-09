import datetime
import torch
from torch.utils.data import dataloader
import pickle
import numpy as np
from params import args
from preprocess import csr_to_tensor
from dataset import RecDataset_beh, RecDataset
from train_and_test import Trainer
from torch.utils.tensorboard import SummaryWriter


class Model:

    def __init__(self):
        self.train_data_path = args.dataset_path + 'trn_'
        self.test_data_path = args.dataset_path + 'tst_int'
        self.target_behavior = args.target_behavior
        self.behaviors_data_csr = {}
        self.behavior_mats = {}
        self.behavior_mats_t = {}
        self.behaviors = ['click', 'fav', 'cart', 'buy']
        self.user_num = -1
        self.item_num = -1

        now_time = datetime.datetime.now()
        self.time = datetime.datetime.strftime(now_time, '%Y_%m_%d__%H_%M_%S')

        self.epoch = 0
        self.train_loss = []
        self.hr_history = []
        self.ndcg_history = []
        self.best_HR = 0
        self.best_NDCG = 0
        self.best_epoch = 0
        self.cnt = 0

        # Load data
        for i in range(len(self.behaviors)):
            behavior = self.behaviors[i]
            with open(self.train_data_path + behavior, 'rb') as fs:
                data = pickle.load(fs)
                self.behaviors_data_csr[i] = data

                if data.get_shape()[0] > self.user_num:
                    self.user_num = data.get_shape()[0]  # 17435
                if data.get_shape()[1] > self.item_num:
                    self.item_num = data.get_shape()[1]  # 35920

                if behavior == args.target_behavior:
                    self.train_mat_csr = data
                    self.train_label_csr = 1 * (self.train_mat_csr != 0)  # Change timestamp to 1
                    self.labelP = np.squeeze(np.array(
                        np.sum(self.train_label_csr, axis=0)))  # [17435], total interacted user_num for each item

        # Change data to tensor
        for i in range(len(self.behaviors)):
            csr_data = (self.behaviors_data_csr[i] != 0) * 1
            self.behavior_mats[i] = csr_to_tensor(csr_data)
            self.behavior_mats_t[i] = csr_to_tensor(csr_data.T)

        # Build train and test dataloader
        target_users, target_items = self.train_label_csr.nonzero()
        target_user_item = np.hstack((target_users.reshape(-1, 1), target_items.reshape(-1, 1)))
        self.train_dataset = RecDataset_beh(self.behaviors, self.behaviors_data_csr, target_user_item, self.item_num,
                                            self.target_behavior, mode='train')
        self.train_loader = dataloader.DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=0, pin_memory=True)

        with open(self.test_data_path, 'rb') as fs:
            data = pickle.load(fs)
        test_user = np.array([idx for idx, i in enumerate(data) if i is not None])
        test_item = np.array([i for idx, i in enumerate(data) if i is not None])
        test_target_user_item = np.hstack((test_user.reshape(-1, 1), test_item.reshape(-1, 1)))
        test_dataset = RecDataset(test_target_user_item)
        self.test_loader = dataloader.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                                 pin_memory=True)

    def save_history(self):
        history = dict()
        history['loss'] = self.train_loss
        history['HR'] = self.hr_history
        history['NDCG'] = self.ndcg_history
        model_name = self.time

        with open(r'./History/' + model_name + '.his', 'wb') as fs:
            pickle.dump(history, fs)

    def save_model(self):
        history = dict()
        history['loss'] = self.train_loss
        history['HR'] = self.hr_history
        history['NDCG'] = self.ndcg_history
        model_name = self.time

        savePath = r'./Model/' + model_name + r'.pth'
        params = {
            'epoch': self.epoch,
            'model': self.best_model,
            'history': history,
            'user_embed': self.user_embed,
            'user_embeds': self.user_embeds,
            'item_embed': self.item_embed,
            'item_embeds': self.item_embeds,
        }
        torch.save(params, savePath)

    def run(self):

        # torch.autograd.set_detect_anomaly(True)
        trainer = Trainer(self.train_loader, self.behaviors, self.user_num, self.item_num, self.behavior_mats,
                          self.behavior_mats_t, self.behaviors_data_csr, self.train_mat_csr)
        # trainer.test_epoch(self.test_loader)
        # trainer.train_epoch()
        for i in range(args.epoch_num+1):
            self.epoch = i+1
            gnn, epoch_loss, user_embed, item_embed, user_embeds, item_embeds = trainer.train_epoch()
            self.train_loss.append(epoch_loss)
            print(f"epoch {self.epoch},  epoch loss {epoch_loss}")

            HR, NDCG = trainer.test_epoch(self.test_loader)
            self.hr_history.append(HR)
            self.ndcg_history.append(NDCG)

            self.save_metric()

            if HR > self.best_HR:
                self.cnt = 0
                self.best_HR = HR
                self.best_epoch = self.epoch
                self.user_embed = user_embed
                self.item_embed = item_embed
                self.user_embeds = user_embeds
                self.item_embeds = item_embeds
                self.best_model = gnn
                self.save_history()
                self.save_model()

            if NDCG > self.best_NDCG:
                self.cnt = 0
                self.best_NDCG = NDCG
                self.best_epoch = self.epoch
                self.user_embed = user_embed
                self.item_embed = item_embed
                self.user_embeds = user_embeds
                self.item_embeds = item_embeds
                self.best_model = gnn
                self.save_history()
                self.save_model()

            if HR < self.best_HR and NDCG < self.best_NDCG:
                self.cnt += 1

            if self.cnt == args.patience:
                print(f"Early stop at {self.best_epoch} :  best HR: {self.best_HR}, best_NDCG: {self.best_NDCG} \n")
                self.save_history()
                self.save_model()
                break

    def save_metric(self):
        path = './runs/' + self.time + '/' + str(self.epoch) + '/'
        writer = SummaryWriter(path)
        for i in range(self.epoch):
            writer.add_scalar('Loss', self.train_loss[i], i)
            writer.add_scalar('NDCG', self.ndcg_history[i], i)
            writer.add_scalar('HR', self.hr_history[i], i)


if __name__ == '__main__':
    model = Model()
    model.run()
    # model.save_metric()
