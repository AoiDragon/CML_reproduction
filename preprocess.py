import torch
import numpy as np


def csr_to_tensor(behavior_data_csr):
    """
    Change behavior data from csr_matrix to torch.tensor matrix
    :param behavior_data_csr: behavior data organized in csr_matrix
    :return: behavior data matrix (and its transpose) organized in tensor
    """
    behavior_data_coo = behavior_data_csr.tocoo()
    user_index = behavior_data_coo.row
    item_index = behavior_data_coo.col

    indices = torch.from_numpy(np.vstack((user_index, item_index)).astype(np.int64))
    values = torch.from_numpy(behavior_data_coo.data)
    shape = torch.Size(behavior_data_coo.shape)

    behavior_mat = torch.sparse.FloatTensor(indices, values, shape).float().cuda()

    return behavior_mat




