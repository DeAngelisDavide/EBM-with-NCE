import torch
import os

def train_ebm(
    energy,
    noise,
    dataloader,
    optimizer,
    nce_loss_fn,
    device="cpu",
    epochs=100,
    patience=5,
    min_delta=1e-4,
    resume=False,
    checkpoint_path='ckpts/nce.pth.tar',
    save_path='ckpts/nce.pth.tar'
):


    start_epoch = 0
    best_loss = float('inf')
    patience_counter = 0

    # Resume training if requested
    if resume and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint at {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        energy.load_state_dict(checkpoint['energy'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['value']

    # Main training loop
    for epoch in range(start_epoch, start_epoch + epochs):
        epoch_loss = 0.0

        for i, x in enumerate(dataloader):
            x = x[0].to(device)

            # Sample from noise distribution
            gen = noise.sample((x.shape[0],)).to(device)

            optimizer.zero_grad()
            loss = nce_loss_fn(energy, noise, x, gen)
            loss.backward()

            # Debug: gradient of latent variable (optional)
            print("grad z:", energy.z.grad if hasattr(energy, 'z') else "no z attr")

            optimizer.step()
            epoch_loss += loss.item()

            print(f"[Epoch {epoch}/{start_epoch + epochs - 1}] [Batch {i}/{len(dataloader)}]")

        # Compute average loss
        epoch_loss /= len(dataloader)
        print(f"Epoch {epoch} average loss: {epoch_loss:.4f}")

        # Early stopping & checkpointing
        if best_loss - epoch_loss > min_delta:
            best_loss = epoch_loss
            patience_counter = 0
            print("Loss improved, saving checkpoint...")

            os.makedirs('ckpts', exist_ok=True)
            torch.save({
                'energy': energy.state_dict(),
                'value': best_loss,
                'epoch': epoch,
            }, save_path)

        else:
            patience_counter += 1
            print(f"No improvement. Patience counter: {patience_counter}/{patience}")
