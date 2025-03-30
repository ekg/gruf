import torch
import pytorch_lightning as pl
import time
import sys

print("Starting test script...")

class TestModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 1)
        
    def forward(self, x):
        return self.layer(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        print(f"TRAINING STEP {batch_idx}")
        time.sleep(0.5)  # Sleep to make sure we can see output
        return {"loss": torch.sum(self.layer(x))}
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.1)

# Create dummy data
class DummyDataLoader:
    def __iter__(self):
        return self
    
    def __next__(self):
        return [torch.randn(2, 10), torch.randn(2, 1)]

print("Creating model and trainer...")
model = TestModel()
trainer = pl.Trainer(max_steps=5, accelerator="auto")

print("Starting training...")
trainer.fit(model, train_dataloaders=DummyDataLoader())
print("Training completed!")
