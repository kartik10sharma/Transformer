import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from tqdm import tqdm

from config import config
from models.gpt_model import MiniGPT
from tokenizer import get_tokenizer
from dataset import TextDataset

# Load and tokenize data
tokenizer = get_tokenizer()
with open("data/appliance.txt", "r", encoding="utf-8") as f:
    text = f.read()
tokens = tokenizer.encode(text)
dataset = TextDataset(tokens, block_size=64)
dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

# Model and training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MiniGPT(config["vocab_size"]).to(device)
optimizer = AdamW(model.parameters(), lr=config["lr"])
criterion = CrossEntropyLoss()

# Training loop
for epoch in range(config["epochs"]):
    model.train()
    total_loss = 0
    for x, y in tqdm(dataloader):
        x, y = x.to(device), y.to(device)
        mask = torch.tril(torch.ones((x.size(1), x.size(1)))).to(device).unsqueeze(0).unsqueeze(0)
        logits = model(x, mask=mask)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}: Loss = {total_loss:.4f}")
    torch.save(model.state_dict(), f"checkpoints/model_epoch{epoch+1}.pt")
