import matplotlib.pyplot as plt
import torch 
import os
from sklearn.manifold import TSNE

def eval_model(energy, noise, data_tensor, epoch, save_dir="eval_plots", device='cpu', sample_size=500):
    os.makedirs(save_dir, exist_ok=True)

    real_data = data_tensor[:sample_size].to(device)
    noise_data = noise.sample(sample_size).to(device)

    with torch.no_grad():
        logp_real = energy(real_data).cpu().numpy()
        logp_noise = energy(noise_data).cpu().numpy()

    ##Plot comparison noise-real
    plt.figure(figsize=(7, 5))
    plt.hist(logp_real, bins=50, alpha=0.6, label="Real")
    plt.hist(logp_noise, bins=50, alpha=0.6, label="Noise")
    plt.xlabel("Model output")
    plt.ylabel("Frequency")
    plt.title(f"Distribution at Epoch {epoch}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/distribution_epoch{epoch}.png")
    plt.close()
    
    #tsne
    try:
        tsne_input = torch.cat([real_data, noise_data], dim=0).cpu().numpy()
        projected = TSNE(n_components=2).fit_transform(tsne_input)

        plt.figure(figsize=(7, 5))
        plt.scatter(projected[:sample_size, 0], projected[:sample_size, 1], label="Real", alpha=0.6)
        plt.scatter(projected[sample_size:, 0], projected[sample_size:, 1], label="Noise", alpha=0.6)
        plt.legend()
        plt.title(f"t-SNE Projection at Epoch {epoch}")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/tsne_epoch_{epoch}.png")
        plt.close()
    except Exception as e:
        print(f"[WARNING] t-SNE failed: {e}")
