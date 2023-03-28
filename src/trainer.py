import os
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from relevance_model import QABert, QADataset
from torchvision.transforms import ToTensor
from pytorch_lightning import Trainer, seed_everything

seed_everything(42, workers=True)

train_data = QADataset('data/WikiQA-train.tsv')
test_data = QADataset('data/WikiQA-test.tsv')
dev_data = QADataset('data/WikiQA-dev.tsv')


train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=True)
dev_dataloader = DataLoader(dev_data, batch_size=32, shuffle=True)

model = QABert()

trainer = Trainer(
    num_nodes=int(os.environ.get("GROUP_WORLD_SIZE", 1)),
    accelerator="cpu",
    devices=1,
    max_epochs=30,
    strategy="ddp"
)

trainer.fit(model, train_dataloader, test_dataloader)
trainer.test(model, test_dataloader, ckpt_path='ckpt_model/latest.ckpt')