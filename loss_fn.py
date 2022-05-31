import torch
import torch.nn as nn
def Similar_sum(w_similar, c_similar, different_1, different_2, tau = 1):
    exp_w = torch.exp(w_similar/tau)
    exp_c = torch.exp(c_similar/tau)
    exp_d1 = torch.exp(different_1/tau)
    exp_d2 = torch.exp(different_2/tau)

    return -torch.log(exp_w*exp_c/(exp_w + exp_c + exp_d1 + exp_d2))
    