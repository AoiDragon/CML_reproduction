import torch
import torch.nn as nn
from params import args


class BGNN(nn.Module):
    def __init__(self, user_num, item_num, behavior_list, behavior_mats, behavior_mats_t):
        super(BGNN, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.behavior_list = behavior_list
        self.behavior_mats = behavior_mats
        self.behavior_mats_t = behavior_mats_t

        self.layer_num = args.layer_num
        self.hidden_dim = args.hidden_dim

        self.user_embedding, self.item_embedding = self.init_embedding()
        self.u_concat_w, self.i_concat_w = self.init_weight()

        self.layers = nn.ModuleList()
        for i in range(self.layer_num):
            layer = GNNLayer(args.hidden_dim, args.hidden_dim, self.user_num, self.item_num,
                             self.behavior_list, self.behavior_mats, self.behavior_mats_t)
            self.layers.append(layer)

    def init_embedding(self):
        user_embedding = torch.nn.Embedding(self.user_num, self.hidden_dim)
        item_embedding = torch.nn.Embedding(self.item_num, self.hidden_dim)
        nn.init.xavier_uniform_(user_embedding.weight)
        nn.init.xavier_uniform_(item_embedding.weight)

        return user_embedding, item_embedding

    def init_weight(self):
        u_concat_w = nn.Parameter(torch.Tensor(self.layer_num * self.hidden_dim, self.hidden_dim))
        i_concat_w = nn.Parameter(torch.Tensor(self.layer_num * self.hidden_dim, self.hidden_dim))
        nn.init.xavier_uniform_(u_concat_w)
        nn.init.xavier_uniform_(i_concat_w)

        return u_concat_w, i_concat_w

    def forward(self):
        """
        Concatenate and project every layer's result
        """
        user_embedding_list = []
        item_embedding_list = []
        user_embeddings_list = []
        item_embeddings_list = []

        user_embedding = self.user_embedding.weight
        item_embedding = self.item_embedding.weight

        for i, layer in enumerate(self.layers):
            user_embedding, item_embedding, user_embeddings, item_embeddings = layer(user_embedding, item_embedding)

            user_embedding_list.append(user_embedding)
            item_embedding_list.append(item_embedding)
            user_embeddings_list.append(user_embeddings)
            item_embeddings_list.append(item_embeddings)

        # Concatenation
        user_embedding = torch.cat(user_embedding_list, dim=1)
        item_embedding = torch.cat(item_embedding_list, dim=1)
        user_embeddings = torch.cat(user_embeddings_list, dim=2)
        item_embeddings = torch.cat(item_embeddings_list, dim=2)

        # Projection
        user_embedding = torch.matmul(user_embedding, self.u_concat_w)
        item_embedding = torch.matmul(item_embedding, self.i_concat_w)
        user_embeddings = torch.matmul(user_embeddings, self.u_concat_w)
        item_embeddings = torch.matmul(item_embeddings, self.i_concat_w)

        return user_embedding, item_embedding, user_embeddings, item_embeddings


class GNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, user_num, item_num, behavior_list, behavior_mats, behavior_mats_t):
        super(GNNLayer, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.behavior_list = behavior_list
        self.behavior_mats = behavior_mats
        self.behavior_mats_t = behavior_mats_t
        self.behavior_num = len(self.behavior_list)

        self.u_w = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.i_w = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.act = nn.PReLU()
        nn.init.xavier_uniform_(self.i_w)
        nn.init.xavier_uniform_(self.u_w)

    def forward(self, init_user_embedding, init_item_embedding):
        """
        Return multi/single behavior aggregation result
        """

        user_embedding_list = [None] * self.behavior_num
        item_embedding_list = [None] * self.behavior_num

        # Message passing
        for i in range(self.behavior_num):
            user_embedding_list[i] = torch.spmm(self.behavior_mats[i], init_item_embedding)
            item_embedding_list[i] = torch.spmm(self.behavior_mats_t[i], init_user_embedding)

        user_embeddings = torch.stack(user_embedding_list, dim=0)  # 4*17435*16
        item_embeddings = torch.stack(item_embedding_list, dim=0)

        # Multi-behaviors aggregation
        multi_aggr_user_embedding = self.act(torch.matmul(torch.mean(user_embeddings, dim=0), self.u_w))
        multi_aggr_item_embedding = self.act(torch.matmul(torch.mean(item_embeddings, dim=0), self.i_w))

        # Single-behavior aggregation
        single_user_embeddings = self.act(torch.matmul(user_embeddings, self.u_w))
        single_item_embeddings = self.act(torch.matmul(item_embeddings, self.i_w))

        return multi_aggr_user_embedding, multi_aggr_item_embedding, single_user_embeddings, single_item_embeddings

