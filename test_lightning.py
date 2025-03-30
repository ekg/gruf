import os
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.strategies import DDPStrategy

# Set environment variables
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_SOCKET_IFNAME"] = "eno1"  # Use the specific interface shown in your logs

class SimpleModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 1)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        print(f"Rank {self.global_rank} | Batch {batch_idx} | Loss: {loss.item():.4f}")
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)

def main():
    pl.seed_everything(42)
    torch.set_float32_matmul_precision('medium')

    # Minimal dataset
    x = torch.randn(20, 10)
    y = torch.randn(20, 1)
    dataset = TensorDataset(x, y)

    dataloader = DataLoader(dataset, batch_size=5, num_workers=0)

    model = SimpleModel()

    # Create a DDPStrategy with the gloo backend
    ddp_strategy = DDPStrategy(process_group_backend="gloo")

    trainer = pl.Trainer(
        max_steps=5,
        accelerator="gpu",
        devices=2,
        strategy=ddp_strategy,  # Pass the strategy object here
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False
    )

    print("Starting training...")
    trainer.fit(model, dataloader)
    print("Training completed.")

if __name__ == "__main__":
    main()
