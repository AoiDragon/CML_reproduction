import torch
import numpy as np
from params import args


def compute(x1, x2, neg1_index=None, neg2_index=None):  # [1024, 16], [1024, 16]

    if neg1_index != None:
        x1 = x1[neg1_index]
        x2 = x2[neg2_index]

    N = x1.shape[0]
    D = x1.shape[1]

    x1 = x1
    x2 = x2
    #  bmm是batch乘法；x1,x2进行内积
    scores = torch.exp(
        torch.div(torch.bmm(x1.view(N, 1, D), x2.view(N, D, 1)).view(N, 1), np.power(D, 1) + 1e-8))  # [1024, 1]
    # scores = e^((x1 dot x2) / 16)
    return scores


def compute_pos_score(tgt_behavior, aux_behavior, step_user_index):
    pos_score = compute(tgt_behavior[step_user_index], aux_behavior[step_user_index]).squeeze()

    return pos_score


def compute_neg_score(tgt_behavior_embedding, aux_behavior_embedding, step_user_index):
    step_user_num = step_user_index.shape[0]
    mini_batch_size = args.CL_mini_batch_size
    mini_batch_num = int(np.ceil(step_user_num / mini_batch_size))

    neg_score_list = []

    for i in range(mini_batch_num):
        # Sample mini-batch user
        start = i * mini_batch_size
        end = min((i + 1) * mini_batch_size, step_user_num)
        mini_batch_user_index = step_user_index[start:end]
        neg1_index, neg2_index = sample_neg_pair_index(mini_batch_user_index, step_user_index)

        now_neg_score = compute(tgt_behavior_embedding, aux_behavior_embedding, neg1_index, neg2_index)
        now_neg_score = torch.sum(now_neg_score.squeeze().view(len(mini_batch_user_index), -1), -1)
        neg_score_list.append(now_neg_score)

    neg_score = torch.cat(neg_score_list, 0)

    return neg_score


def sample_neg_pair_index(mini_batch_user_index, step_user_index):
    # Remove duplicated user in step_user
    mini_batch_user_index_set = set(np.array(mini_batch_user_index.cpu()))
    step_user_index_set = set(np.array(step_user_index.cpu()))
    step_user_index_set = step_user_index_set - mini_batch_user_index_set

    # Reshape step_user
    neg2_index = torch.as_tensor(np.array(list(step_user_index_set))).long().cuda()  # [187]
    neg2_index = torch.unsqueeze(neg2_index, dim=0)  # [1, 187]
    neg2_index = neg2_index.repeat(len(mini_batch_user_index), 1)  # [15, 187]
    neg2_index = neg2_index.view(-1)  # [187*15]

    # Reshape mini-batch user
    neg1_index = mini_batch_user_index.long().cuda()  # [15]
    neg1_index = neg1_index.unsqueeze(dim=1)  # [15, 1]
    neg1_index = neg1_index.repeat(1, len(step_user_index_set))  # [15, 187]
    neg1_index = neg1_index.view(-1)  # [15*187]

    return neg1_index, neg2_index


def compute_infoNCE_loss(user_embeddings, batch_user_index, behaviors):
    infoNCE_loss_list = []

    # Sample step users
    CL_len = int(batch_user_index.shape[0] / 10)
    step_user_index = np.random.choice(batch_user_index.cpu().numpy(), size=CL_len, replace=False)
    step_user_index = torch.as_tensor(step_user_index).cuda()

    for index in range(len(behaviors)):
        # Compute pos_score in mini-batch users
        pos_score = compute_pos_score(user_embeddings[-1], user_embeddings[index], step_user_index)
        # Compute neg_score between mini-batch users and step users
        neg_score = compute_neg_score(user_embeddings[-1], user_embeddings[index], step_user_index)

        # Compute loss
        infoNCE_loss = -torch.log(torch.div(pos_score, neg_score + 1e-8) + 1e-8)
        infoNCE_loss = torch.where(torch.isnan(infoNCE_loss), torch.full_like(infoNCE_loss, 1e-8), infoNCE_loss)
        infoNCE_loss_list.append(infoNCE_loss)

    return infoNCE_loss_list, step_user_index
