import torch 

import torch 

def langevin_sampling(energy_model, noise, batch_size=1, steps=50, epsilon=0.01, device = "cpu"):
    """
    Langevin dynamics sampling directly in embedding space.
    Sampling z ~ p(z) ∝ exp(f(z))
    """
    energy_model.to(device).eval()

    z = noise.sample(batch_size)
    z_noise = z.clone()
    score_norms = []
    energy_values = []

    for step in range(steps):
        z.requires_grad_(True)

        f_z = energy_model.f(z).sum()
        energy_values.append(f_z.item())

        grad = torch.autograd.grad(f_z, z, create_graph=False, retain_graph=False)[0]

        # Norm tracking
        score_norm = torch.norm(grad.view(grad.shape[0], -1), dim=1).mean().item()
        score_norms.append(score_norm)

        # Langevin update
        noise = torch.randn_like(z)
        z = z + (epsilon / 2) * grad + torch.sqrt(torch.tensor(epsilon, device=z.device)) * noise
        z = z.detach()

    return z, score_norms, energy_values, z_noise