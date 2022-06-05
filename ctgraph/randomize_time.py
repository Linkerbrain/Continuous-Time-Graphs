import torch


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
    n=50
    batch[("u", "b", "i")].oui = torch.randint(low=0, high=n, size=batch[("u", "b", "i")].oui.shape, device=real_t.device)
    batch[("u", "b", "i")].oiu = torch.randint(low=0, high=n, size=batch[("u", "b", "i")].oiu.shape,device=real_t.device)

    return batch
