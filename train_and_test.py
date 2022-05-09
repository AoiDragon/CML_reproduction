import torch
import torch.nn as nn
from tqdm import tqdm
import pickle
import numpy as np
from BGNN import BGNN
from MW_Net import MetaWeightNet
from contrastive_learning import compute_infoNCE_loss
from params import args


class Trainer:

    def __init__(self, dataloader, behaviors, user_num, item_num, behavior_mats, behavior_mats_t, behaviors_data,
                 train_mat):

        self.dataloader = dataloader
        self.behaviors = behaviors
        self.user_num = user_num
        self.item_num = item_num
        self.behavior_mats = behavior_mats
        self.behavior_mats_t = behavior_mats_t
        self.behaviors_data = behaviors_data
        self.train_mat = train_mat

        self.behavior_loss_list = [None] * len(self.behaviors)
        self.user_id_list = [None] * len(self.behaviors)
        self.item_id_pos_list = [None] * len(self.behaviors)
        self.item_id_neg_list = [None] * len(self.behaviors)

        self.meta_multi_single_file = args.dataset_path + '/meta_multi_single_beh_user_index_shuffle'
        self.meta_multi_single = pickle.load(open(self.meta_multi_single_file, 'rb'))
        self.meta_start_index = 0
        self.meta_end_index = self.meta_start_index + args.meta_batch

        self.prepare_model()

    def prepare_model(self):
        self.gnn = BGNN(self.user_num, self.item_num, self.behaviors, self.behavior_mats, self.behavior_mats_t).cuda()
        self.meta_weight_net = MetaWeightNet(len(self.behaviors)).cuda()

        self.gnn_opt = torch.optim.AdamW(self.gnn.parameters(), lr=args.lr, weight_decay=args.gnn_opt_weight_decay)
        self.meta_opt = torch.optim.AdamW(self.meta_weight_net.parameters(), lr=args.meta_lr,
                                          weight_decay=args.meta_opt_weight_decay)

        self.gnn_scheduler = torch.optim.lr_scheduler.CyclicLR(self.gnn_opt, args.gnn_opt_base_lr, args.gnn_opt_max_lr,
                                                               step_size_up=5, step_size_down=10, mode='triangular',
                                                               gamma=0.99, scale_fn=None, scale_mode='cycle',
                                                               cycle_momentum=False, base_momentum=0.8,
                                                               max_momentum=0.9, last_epoch=-1)
        self.meta_scheduler = torch.optim.lr_scheduler.CyclicLR(self.meta_opt, args.meta_opt_base_lr,
                                                                args.meta_opt_max_lr,
                                                                step_size_up=3, step_size_down=7, mode='triangular',
                                                                gamma=0.98, scale_fn=None, scale_mode='cycle',
                                                                cycle_momentum=False, base_momentum=0.9,
                                                                max_momentum=0.99, last_epoch=-1)

    def sample_batch_data(self, user, pos_item, neg_item, index):
        """
        Sample non-zero data from batch
        """
        not_zero_index = np.where(pos_item[index].cpu().numpy() != -1)[0]

        self.user_id_list[index] = user[not_zero_index].long().cuda()
        self.item_id_pos_list[index] = pos_item[index][not_zero_index].long().cuda()
        self.item_id_neg_list[index] = neg_item[index][not_zero_index].long().cuda()

    def sample_meta_data(self, batIds, labelMat):
        """
        Sample non-zero data for recommendation from meta data
        """

        temLabel = labelMat[batIds.cpu()].toarray()
        batch = len(batIds)
        user_id = []
        item_id_pos = []
        item_id_neg = []

        cur = 0
        for i in range(batch):
            # 实际交互历史
            posset = np.reshape(np.argwhere(temLabel[i] != 0), [-1])
            sampNum = min(args.samp_num, len(posset))
            if sampNum == 0:
                poslocs = [np.random.choice(labelMat.shape[1])]
                neglocs = [poslocs[0]]
            else:
                poslocs = np.random.choice(posset, sampNum)
                neglocs = self.neg_samp(temLabel[i], sampNum, labelMat.shape[1])

            for j in range(sampNum):
                user_id.append(batIds[i].item())
                item_id_pos.append(poslocs[j].item())
                item_id_neg.append(neglocs[j])
                cur += 1

        return torch.as_tensor(np.array(user_id)).long().cuda(), torch.as_tensor(
            np.array(item_id_pos)).long().cuda(), torch.as_tensor(
            np.array(item_id_neg)).long().cuda()

    def neg_samp(self, temLabel, sampSize, nodeNum):
        negset = [None] * sampSize
        cur = 0
        while cur < sampSize:
            rdmItm = np.random.choice(nodeNum)
            if temLabel[rdmItm] == 0:
                negset[cur] = rdmItm
                cur += 1
        return negset

    def inner_product(self, user_embedding, pos_item_embedding, neg_item_embedding):
        """
        Make recommendation
        """
        pos_pred = torch.sum(torch.mul(user_embedding, pos_item_embedding), dim=1)  # [user_num, 1]
        neg_pred = torch.sum(torch.mul(user_embedding, neg_item_embedding), dim=1)
        return pos_pred, neg_pred

    def train_epoch(self):
        self.dataloader.dataset.neg_sample()
        epoch_loss = 0
        for user, pos_item, neg_item in tqdm(self.dataloader):
            batch_user = user.long().cuda()
            meta_user = torch.as_tensor(self.meta_multi_single[self.meta_start_index:self.meta_end_index]).cuda()

            if self.meta_end_index == self.meta_multi_single.shape[0]:
                self.meta_start_index = 0
            else:
                self.meta_start_index = (self.meta_start_index + args.meta_batch) % (
                        self.meta_multi_single.shape[0] - 1)
            self.meta_end_index = min(self.meta_start_index + args.meta_batch, self.meta_multi_single.shape[0])

            self.round1(batch_user, pos_item, neg_item)
            self.round2(meta_user)
            loss, user_embed, item_embed, user_embeds, item_embeds = self.round3(batch_user)
            epoch_loss += loss
        self.gnn_scheduler.step()
        self.meta_scheduler.step()

        return self.gnn, epoch_loss, user_embed, item_embed, user_embeds, item_embeds

    def round1(self, user, pos_item, neg_item):
        """
        Use batch data to update copied-GNN and MW-net
        """
        # Tmp data
        behavior_loss_list = [None] * len(self.behaviors)

        # Copy GNN
        self.tmp_GNN = BGNN(self.user_num, self.item_num, self.behaviors, self.behavior_mats,
                            self.behavior_mats_t).cuda()
        self.tmp_GNN.load_state_dict(self.gnn.state_dict())
        tmp_opt = torch.optim.AdamW(self.tmp_GNN.parameters(), lr=args.lr, weight_decay=args.gnn_opt_weight_decay)

        all_user_embedding, all_item_embedding, all_user_embeddings, all_item_embeddings = self.tmp_GNN()

        # Recommend
        for index in range(len(self.behaviors)):
            self.sample_batch_data(user, pos_item, neg_item, index)

            meta_user_embedding = all_user_embedding[self.user_id_list[index]]
            meta_pos_item_embedding = all_item_embedding[self.item_id_pos_list[index]]
            meta_neg_item_embedding = all_item_embedding[self.item_id_neg_list[index]]

            pos_pred, neg_pred = self.inner_product(meta_user_embedding, meta_pos_item_embedding,
                                                    meta_neg_item_embedding)
            behavior_loss_list[index] = -torch.log((pos_pred.view(-1) - neg_pred.view(-1)).sigmoid() + 1e-8)

        # Compute infoNCE_loss
        infoNCE_loss_list, step_user_index = compute_infoNCE_loss(all_user_embeddings, user, self.behaviors)

        # Compute loss weights
        infoNCE_loss_list_weights, behavior_loss_list_weights = \
            self.meta_weight_net(infoNCE_loss_list, behavior_loss_list, step_user_index, self.user_id_list,
                                 all_user_embeddings, all_user_embedding)

        # Compute model loss
        for i in range(len(self.behaviors)):
            behavior_loss_list[i] = (behavior_loss_list[i] * behavior_loss_list_weights[i]).sum()
            infoNCE_loss_list[i] = (infoNCE_loss_list[i] * infoNCE_loss_list_weights[i]).sum()

        bpr_loss = sum(behavior_loss_list) / len(behavior_loss_list)
        infoNCE_loss = sum(infoNCE_loss_list) / len(infoNCE_loss_list)
        reg_loss = torch.norm(meta_user_embedding, 2) ** 2 + torch.norm(meta_pos_item_embedding, 2) ** 2 + \
                   torch.norm(meta_neg_item_embedding, 2) ** 2

        model_loss = (bpr_loss + args.beta * infoNCE_loss + args.reg * reg_loss) / args.batch_size

        tmp_opt.zero_grad(set_to_none=True)
        self.meta_opt.zero_grad(set_to_none=True)
        # with torch.autograd.detect_anomaly():
        model_loss.backward()

        nn.utils.clip_grad_norm_(self.meta_weight_net.parameters(), max_norm=20, norm_type=2)
        nn.utils.clip_grad_norm_(self.tmp_GNN.parameters(), max_norm=20, norm_type=2)

        # Optimize tmp_GNN and MW_net
        tmp_opt.step()
        self.meta_opt.step()

    def round2(self, user):
        """
        Use meta data to update MW-net
        """

        behavior_loss_list = [None] * len(self.behaviors)
        user_index_list = [None] * len(self.behaviors)

        # Use updated tmp_GNN to compute embeddings
        all_user_embedding, all_item_embedding, all_user_embeddings, all_item_embeddings = self.tmp_GNN()

        # Recommend
        for index in range(len(self.behaviors)):
            user_id, pos_item_id, neg_item_id = self.sample_meta_data(user, self.behaviors_data[index])
            user_index_list[index] = user_id

            rec_user_embedding = all_user_embedding[user_id]
            rec_pos_item_embedding = all_item_embedding[pos_item_id]
            rec_neg_item_embedding = all_item_embedding[neg_item_id]

            pos_pred, neg_pred = self.inner_product(rec_user_embedding, rec_pos_item_embedding, rec_neg_item_embedding)

            behavior_loss_list[index] = -torch.log((pos_pred.view(-1) - neg_pred.view(-1)).sigmoid() + 1e-8)

        # Compute infoNCE_loss of this epoch
        self.infoNCE_loss_list, step_user_index = compute_infoNCE_loss(all_user_embeddings, user, self.behaviors)

        # Compute loss weights
        infoNCE_loss_list_weights, behavior_loss_list_weights = \
            self.meta_weight_net(self.infoNCE_loss_list, behavior_loss_list, step_user_index, user_index_list,
                                 all_user_embeddings, all_user_embedding)

        # Compute model loss
        for i in range(len(self.behaviors)):
            behavior_loss_list[i] = (behavior_loss_list[i] * behavior_loss_list_weights[i]).sum()
            self.infoNCE_loss_list[i] = (self.infoNCE_loss_list[i] * infoNCE_loss_list_weights[i]).sum()

        bpr_loss = sum(behavior_loss_list) / len(behavior_loss_list)
        infoNCE_loss = sum(self.infoNCE_loss_list) / len(self.infoNCE_loss_list)
        reg_loss = torch.norm(rec_user_embedding, 2) ** 2 + torch.norm(rec_pos_item_embedding, 2) ** 2 + \
                   torch.norm(rec_neg_item_embedding, 2) ** 2

        model_loss = 0.5 * (bpr_loss + args.beta * infoNCE_loss + args.reg * reg_loss) / args.batch_size

        self.meta_opt.zero_grad(set_to_none=True)
        # with torch.autograd.detect_anomaly():
        model_loss.backward()
        nn.utils.clip_grad_norm_(self.meta_weight_net.parameters(), max_norm=20, norm_type=2)

        # Optimize MW_net
        self.meta_opt.step()

    def round3(self, user):
        """
        Use batch data to update GNN
        """

        # Use original GNN model to compute embeddings
        all_user_embedding, all_item_embedding, all_user_embeddings, all_item_embeddings = self.gnn()

        # Recommend
        for index in range(len(self.behaviors)):
            rec_user_embedding = all_user_embedding[self.user_id_list[index]]
            rec_pos_item_embedding = all_item_embedding[self.item_id_pos_list[index]]
            rec_neg_item_embedding = all_item_embedding[self.item_id_neg_list[index]]

            pos_pred, neg_pred = self.inner_product(rec_user_embedding, rec_pos_item_embedding, rec_neg_item_embedding)

            self.behavior_loss_list[index] = -torch.log((pos_pred.view(-1) - neg_pred.view(-1)).sigmoid() + 1e-8)

        # Compute infoNCE_loss
        infoNCE_loss_list, step_user_index = compute_infoNCE_loss(all_user_embeddings, user, self.behaviors)

        # Compute loss weights without gradients
        with torch.no_grad():
            infoNCE_loss_list_weights, behavior_loss_list_weights = \
                self.meta_weight_net(infoNCE_loss_list, self.behavior_loss_list, step_user_index, self.user_id_list,
                                     all_user_embeddings, all_user_embedding)

        for i in range(len(self.behaviors)):
            infoNCE_loss_list[i] = (infoNCE_loss_list[i] * infoNCE_loss_list_weights[i]).sum()
            self.behavior_loss_list[i] = (self.behavior_loss_list[i] * behavior_loss_list_weights[i]).sum()

        bpr_loss = sum(self.behavior_loss_list) / len(self.behavior_loss_list)
        infoNCE_loss = sum(infoNCE_loss_list) / len(infoNCE_loss_list)
        reg_loss = torch.norm(rec_user_embedding, 2) ** 2 + torch.norm(rec_pos_item_embedding, 2) ** 2 + \
                   torch.norm(rec_neg_item_embedding, 2) ** 2

        model_loss = (bpr_loss + args.beta * infoNCE_loss + args.reg * reg_loss) / args.batch_size
        loss = model_loss

        self.gnn_opt.zero_grad(set_to_none=True)
        # with torch.autograd.detect_anomaly():
        model_loss.backward()
        nn.utils.clip_grad_norm_(self.gnn.parameters(), max_norm=20, norm_type=2)

        # Optimize GNN
        self.gnn_opt.step()

        return loss, all_user_embedding, all_item_embedding, all_user_embeddings, all_item_embeddings

    def sample_test_batch(self, user_id, item_id):
        """"""

        batch_size = len(user_id)
        tmp_len = batch_size * 100

        sub_train_mat = self.train_mat[user_id].toarray()

        user_item1 = item_id
        user_compute = [None] * tmp_len
        item_compute = [None] * tmp_len
        user_item100 = [None] * batch_size

        cnt = 0
        for i in range(batch_size):
            pos_item = user_item1[i]
            neg_set = np.reshape(np.argwhere(sub_train_mat[i] == 0), [-1])

            # Sample 99 neg_items
            neg_items = np.random.permutation(neg_set)[:99]

            # Concatenate 99 neg_items with 1 pos_item
            user_item100[i] = np.concatenate((neg_items, np.array([pos_item])))

            # Build user-item pairs, 1 user corresponds to 1 pos_item and 99 neg_items
            for j in range(100):
                user_compute[cnt] = user_id[i]
                item_compute[cnt] = user_item100[i][j]
                cnt += 1

        return user_compute, item_compute, user_item1, user_item100

    def calc_metric(self, pred, user_item1, user_item100):

        hit = 0
        ndcg = 0

        for i in range(pred.shape[0]):

            _, shoot_index = torch.topk(pred[i], args.shoot)
            shoot_index = shoot_index.cpu()
            shoot_item = user_item100[i][shoot_index].tolist()

            if type(shoot_item) != int and (user_item1[i] in shoot_item):
                hit += 1
                pos = shoot_item.index(user_item1[i])
                ndcg += np.reciprocal(np.log2(pos + 2))
            elif type(shoot_item) == int and (user_item1[i] == shoot_item):
                hit += 1
                ndcg += np.reciprocal(np.log2(2))

        return hit, ndcg

    def test_epoch(self, data_loader):

        epochHR, epochNDCG = 0, 0

        with torch.no_grad():
            all_user_embedding, all_item_embedding, all_user_embeddings, all_item_embeddings = self.gnn()

        cnt = 0
        tot = 0
        for user, pos_item in data_loader:
            user_compute, item_compute, user_item1, user_item100 = self.sample_test_batch(user, pos_item)

            user_embedding = all_user_embedding[user_compute]
            item_embedding = all_item_embedding[item_compute]

            pred = torch.sum(torch.mul(user_embedding, item_embedding), dim=1)
            hit, ndcg = self.calc_metric(torch.reshape(pred, [user.shape[0], 100]), user_item1, user_item100)

            epochHR += hit
            epochNDCG += ndcg

            cnt += 1
            tot += user.shape[0]

        result_HR = epochHR / tot
        result_NDCG = epochNDCG / tot
        print(f"Step {cnt}:  hit:{result_HR}, ndcg:{result_NDCG}")

        return result_HR, result_NDCG
