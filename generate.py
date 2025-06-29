from models.gpt_model import MiniGPT
from tokenizer import get_tokenizer
import torch

def generate_text(prompt, model, tokenizer, max_length=100):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    for _ in range(max_length):
        mask = torch.tril(torch.ones((input_ids.size(1), input_ids.size(1)))).to(device).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            logits = model(input_ids, mask=mask)
        next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
        input_ids = torch.cat((input_ids, next_token), dim=1)
    return tokenizer.decode(input_ids[0])

# Usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = get_tokenizer()
model = MiniGPT(vocab_size=tokenizer.vocab_size).to(device)
model.load_state_dict(torch.load("checkpoints/model_epoch10.pt"))
prompt = "My microwave is"
print(generate_text(prompt, model, tokenizer))
