import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam

from minGRU_pytorch.minLM import minLM

class LightningMinLM(pl.LightningModule):
    def __init__(
        self,
        num_tokens=256,
        dim=512,
        depth=6,
        ff_mult=4,
        expansion=1.5,
        conv_kernel_size=3,
        learning_rate=1e-4,
        use_lstm=False
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = minLM(
            num_tokens=num_tokens,
            dim=dim,
            depth=depth,
            ff_mult=ff_mult,
            expansion=expansion,
            conv_kernel_size=conv_kernel_size,
            use_lstm=use_lstm
        )
        self.learning_rate = learning_rate

    def forward(self, x, prev_hiddens=None):
        """Regular forward for inference or evaluation"""
        return self.model(x, prev_hiddens=prev_hiddens, return_loss=False)

    def training_step(self, batch, batch_idx, hiddens=None):
        """
        The training_step function for TBPTT
        
        Args:
            batch: Shape [batch_size, seq_len+1] tokens
            batch_idx: The batch index
            hiddens: Hidden states from previous chunks (for TBPTT)
        """
        # Run the model with the hidden states from the previous chunk
        logits, new_hiddens = self.model(
            batch,
            prev_hiddens=hiddens,
            return_loss=False,
            return_prev_hiddens=True
        )

        # Standard language modeling loss (predict next token)
        inp_tokens = batch[:, :-1]
        labels = batch[:, 1:]

        # We drop the last logit since there's no next token to predict
        logits_for_loss = logits[:, :-1]  # => [batch_size, seq_len-1, num_tokens]
        loss = F.cross_entropy(
            logits_for_loss.reshape(-1, logits_for_loss.size(-1)),
            labels.reshape(-1)
        )

        self.log("train_loss", loss, prog_bar=True)
        
        # Return loss and the new hidden states for the next chunk
        return {"loss": loss, "hiddens": new_hiddens}

    def validation_step(self, batch, batch_idx):
        """Full sequence validation"""
        loss = self.model(batch, return_loss=True)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)
