import torch
import torch.nn as nn
from params import args


class MetaWeightNet(nn.Module):
    def __init__(self, behavior_num):
        super(MetaWeightNet, self).__init__()

        self.behavior_num = behavior_num
        self.batch_norm = nn.BatchNorm1d(1)
        self.sigmoid = nn.Sigmoid()
        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(p=args.drop_rate)

        self.SSL_layer1 = nn.Linear(3 * args.hidden_dim, 1)
        self.SSL_layer2 = nn.Linear(2 * args.hidden_dim, 1)
        self.RS_layer1 = nn.Linear(3 * args.hidden_dim, 1)
        self.RS_layer2 = nn.Linear(args.hidden_dim, 1)

    def forward(self, infoNCELoss_list, bprLoss_list, step_user_index, batch_user_index, user_embeddings,
                user_embedding):
        """
        step_user_index: user used in SSL
        batch_user_index: user used in recommend
        """
        infoNCELoss_weight_list = [None] * self.behavior_num
        bprLoss_weight_list = [None] * self.behavior_num

        for i in range(self.behavior_num):
            infoNCELoss_mat = infoNCELoss_list[i].unsqueeze(1).repeat(1, args.hidden_dim)
            SSL_Z1 = torch.cat((torch.cat((infoNCELoss_mat, user_embeddings[i][step_user_index]), dim=1),
                                user_embedding[step_user_index]), dim=1)
            SSL_Z2 = (infoNCELoss_list[i].unsqueeze(1).repeat(1, args.hidden_dim * 2)) * \
                     torch.cat((user_embeddings[i][step_user_index], user_embedding[step_user_index]), dim=1)

            # SSL_Z2 =  torch.cat((user_embeddings[i][step_user_index], user_embedding[step_user_index]), dim=1)

            SSL_weight_1 = self.sigmoid(self.batch_norm(self.prelu(self.SSL_layer1(SSL_Z1))).squeeze())
            SSL_weight_2 = self.sigmoid(self.batch_norm(self.prelu(self.SSL_layer2(SSL_Z2))).squeeze())
            infoNCELoss_weight_list[i] = (SSL_weight_1 + SSL_weight_2) / 2

            bprLoss_mat = bprLoss_list[i].unsqueeze(1).repeat(1, args.hidden_dim)
            RS_Z1 = torch.cat((torch.cat((bprLoss_mat, user_embeddings[i][batch_user_index[i]]), dim=1),
                               user_embedding[batch_user_index[i]]), dim=1)
            RS_Z2 = bprLoss_mat * user_embedding[batch_user_index[i]]
            # RS_Z2 = torch.cat((user_embeddings[i][batch_user_index[i]], user_embedding[batch_user_index[i]]), dim=1)

            RS_weight_1 = self.prelu(self.RS_layer1(RS_Z1))
            RS_weight_2 = self.prelu(self.RS_layer2(RS_Z2))

            if RS_weight_1.shape[0] > 1:
                RS_weight_1 = self.sigmoid(self.batch_norm(RS_weight_1)).squeeze()
            else:
                # print(1)
                # print(RS_weight_1)
                # print(RS_weight_1.shape)
                # RS_weight_1[0][0] = 0.5
                # RS_weight_1 = self.sigmoid(RS_weight_1).squeeze()
                RS_weight_1 = torch.tensor([[1]], dtype=torch.float).cuda()

            if RS_weight_2.shape[0] > 1:
                RS_weight_2 = self.sigmoid(self.batch_norm(RS_weight_2)).squeeze()
            else:
                # print(2)
                # print(RS_weight_2)
                # print(RS_weight_2.shape)
                # RS_weight_2[0][0] = 0.5
                # RS_weight_2 = self.sigmoid(RS_weight_2).squeeze()
                RS_weight_2 = torch.tensor([[1]], dtype=torch.float).cuda()

            # RS_weight_1 = self.sigmoid(self.batch_norm(self.prelu(self.RS_layer1(RS_Z1))).squeeze())
            # RS_weight_2 = self.sigmoid(self.batch_norm(self.prelu(self.RS_layer2(RS_Z2))).squeeze())
            bprLoss_weight_list[i] = (RS_weight_1 + RS_weight_2) / 2

        return infoNCELoss_weight_list, bprLoss_weight_list
