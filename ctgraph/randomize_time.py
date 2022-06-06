import torch

from ctgraph.models.recommendation.dgsr_utils import relative_order


def randomize_time(batch):
    # randomize timesteps
    real_t = batch[('u', 'b', 'i')].t
    fake_t = torch.rand(real_t.shape,device=real_t.device) * (real_t.max()-real_t.min()) + real_t.min() #fake t is in same range
    batch[('u', 'b', 'i')].t = fake_t

    # randomize timesteps target
    real_t_target = batch['target'].t
    fake_t_target = torch.rand(real_t_target.shape,device=real_t.device) * (real_t.max()-real_t.min()) + real_t.min()
    batch[('target')].t = fake_t_target

    # randomize oui
    u_code = batch['u'].code
    i_code = batch['i'].code
    edge_index = batch[('u', 'b', 'i')].edge_index
    user_per_trans, item_per_trans = edge_index[0], edge_index[1]

    oui = batch[('u', 'b', 'i')].oui
    oiu = batch[('u', 'b', 'i')].oiu
    rui = relative_order(oui, user_per_trans, n=50)
    riu = relative_order(oiu, item_per_trans, n=50)

    n=50
    batch[("u", "b", "i")].oui = rui[torch.randperm(rui.shape[0], device=real_t.device)] # torch.randint(low=39, high=50, size=batch[("u", "b", "i")].oui.shape, device=real_t.device)
    batch[("u", "b", "i")].oiu = riu[torch.randperm(riu.shape[0], device=real_t.device)] # torch.randint(low=39, high=50, size=batch[("u", "b", "i")].oiu.shape,device=real_t.device)


    return batch
