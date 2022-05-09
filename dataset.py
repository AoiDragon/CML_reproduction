import torch.utils.data as data
import numpy as np


class RecDataset(data.Dataset):
    """
    Test dataset
    """
    def __init__(self, test_target_user_item):
        super(RecDataset, self).__init__()
        self.data = test_target_user_item

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user = self.data[idx][0]
        item_i = self.data[idx][1]
        return user, item_i


class RecDataset_beh(data.Dataset):
    """
    Build a dataset for each behavior. Length is equal to interact_num under tgt_beh.
    Neg_items and pos_items under non_tgt_beh need to be randomly sampled.
    """
    def __init__(self, behaviors_list, behaviors_data, target_user_item, item_num, target_behavior, mode):
        super(RecDataset_beh, self).__init__()

        self.behaviors_list = behaviors_list
        self.behaviors_num = len(behaviors_list)
        self.behaviors_data = behaviors_data
        self.target_user_item = target_user_item
        self.item_num = item_num
        self.target_behavior = target_behavior
        self.mode = mode

        self.interact_num = self.target_user_item.shape[0]  # 131685
        self.pos_item = [None] * self.interact_num
        self.neg_item = [None] * self.interact_num
        for i in range(self.interact_num):
            self.pos_item[i] = [None] * self.behaviors_num
            self.neg_item[i] = [None] * self.behaviors_num

        self.interact_his = [None] * self.behaviors_num
        for i in range(self.behaviors_num):
            self.interact_his[i] = self.behaviors_data[i].todok()

    def neg_sample(self):

        for index in range(self.behaviors_num):

            user, item = self.behaviors_data[index].nonzero()

            pos_item_set = np.array(list(set(item)))

            # Random initialize
            self.init_pos_item_index = np.random.choice(pos_item_set, size=self.interact_num, replace=True)
            self.init_neg_item_index = np.random.randint(low=0, high=self.item_num, size=self.interact_num)

            for i in range(self.interact_num):

                uid = self.target_user_item[i][0]
                pos_iid = self.pos_item[i][index] = self.init_pos_item_index[i]
                neg_iid = self.neg_item[i][index] = self.init_neg_item_index[i]

                # Sample negative items under every behavior
                while(uid, neg_iid) in self.interact_his[index]:
                    neg_iid = np.random.randint(low=0, high=self.item_num)
                    self.neg_item[i][index] = neg_iid

                if self.behaviors_list[index] == self.target_behavior:  # No need to random sample under tgt_beh
                    self.pos_item[i][index] = item[i]
                else:
                    if (uid, pos_iid) not in self.interact_his[index]:
                        if len(self.behaviors_data[index][uid].data) == 0:  # No interact history under this behavior
                            self.pos_item[i][index] = -1
                        else:
                            user_interact_his = self.behaviors_data[index][uid].toarray()
                            interacted_items_index = np.where(user_interact_his != 0)[1]
                            self.pos_item[i][index] = np.random.choice(interacted_items_index, size=1)[0]

    def __len__(self):
        return len(self.target_user_item)

    def __getitem__(self, idx):
        user = self.target_user_item[idx][0]
        pos_item = self.pos_item[idx]

        if self.mode == 'train':
            neg_item = self.neg_item[idx]
            return user, pos_item, neg_item
        else:
            return user, pos_item











