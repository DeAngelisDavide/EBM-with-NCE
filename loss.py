import torch 
import torch.nn.functional as F


def nce_loss(energy, noise, x, gen):
    #for first term
    f_x = energy.f(x)  #.f in case the forward is changed by introducing z
    logp_x = noise.log_prob(x)
    logz = energy.z

    first_term = f_x - torch.logsumexp(torch.stack([f_x, logz+logp_x[:, None]], dim=1), dim=1)

    #for second term
    logp_n = noise.log_prob(gen)
    #p_n = torch.exp(logp_n)
    f_n = energy.f(gen)

    second_term =  logz + logp_n[:, None] - torch.logsumexp(torch.stack([f_n, logz+logp_n[:, None]], dim=1), dim=1)

    #Monitoring
    print("z * q(x):", (logz + logp_x[:5]).detach())
    print("f_x:", f_x.min().item(), f_x.max().item())
    print("logp_x:", logp_x.min().item(), logp_x.max().item())
    print("f_n:", f_n.min().item(), f_n.max().item())
    print("logp_n:", logp_n.min().item(), logp_n.max().item())
    print("Z:", energy.z)
    print("first term:", first_term[:10])
    print("second term:", second_term[:10])
    return -torch.mean(first_term + second_term)





def nce_loss2(energy, noise, x, x_neg):
    f_x = energy.f(x)                  # [B]
    logp_x = noise.log_prob(x)       # [B]

    f_n = energy.f(x_neg)              # [B]
    logp_n = noise.log_prob(x_neg)   # [B]

    logZ = energy.z

    pos = F.logsigmoid(f_x - logZ - logp_x)
    neg = F.logsigmoid(-(f_n - logZ - logp_n))

    ##Monitoring
    print("z * q(x):", (logZ + logp_x[:5]).detach())
    print("f_x:", f_x.min().item(), f_x.max().item())
    print("logp_x:", logp_x.min().item(), logp_x.max().item())
    print("f_n:", f_n.min().item(), f_n.max().item())
    print("logp_n:", logp_n.min().item(), logp_n.max().item())
    print("Z:", energy.z)
    print("first term:", pos[:1])
    print("second term:", neg[:1])

    return -(pos.mean() + neg.mean())