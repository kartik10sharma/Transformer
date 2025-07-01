
# 🧠 Transformer From Scratch in PyTorch

A clean and educational implementation of the original **Transformer model** introduced in the paper [*Attention is All You Need*](https://arxiv.org/abs/1706.03762), written in PyTorch.

This repo provides an end-to-end encoder-decoder Transformer architecture for sequence-to-sequence tasks like machine translation, text generation, summarization, and more.

---

## 🔍 Features

- ✅ Custom `MultiHeadSelfAttention` layer
- ✅ Complete `TransformerBlock` with normalization, residuals, and feedforward layers
- ✅ Encoder and Decoder with positional embeddings
- ✅ Support for padding masks and look-ahead masks
- ✅ End-to-end training-ready Transformer model
- ✅ Modular, readable, and extensible codebase

---

## 📁 Project Structure

```

Transformer/
├── models/
│   └── modelimp.py   # Core Transformer model implementation
├── README.md          # Project documentation

````

---

## 📌 Model Architecture

The model is composed of the following layers:

### 🔸 Encoder
- Embedding (word + position)
- N × Transformer blocks

### 🔸 Decoder
- Embedding (word + position)
- Masked self-attention
- Encoder-Decoder cross-attention
- Feed-forward layers
- Final output projection

### 🔸 Attention Mechanism
- Multi-head scaled dot-product attention with masking
- Efficient matrix multiplications using `einsum`

---

## ⚙️ How to Use

### 1. Clone the Repository

```bash
git clone https://github.com/kartik10sharma/transformer-pytorch.git
cd transformer-pytorch
````

### 2. Install Requirements

```bash
pip install torch
```

### 3. Run the Model

```bash
python models/modelimp.py
```

This runs a forward pass on dummy data:

```python
x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]])
trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 8], [1, 5, 6, 2, 4, 7, 6, 2]])
```

Expected output shape:

```
torch.Size([2, 7, 10])
```

---

## 🧪 Example Use Cases

This model can be adapted for:

* 🔁 Machine Translation
* ✂️ Text Summarization
* 💬 Chatbots
* 🤖 Code Generation
* 📜 Autocomplete Models
* 🧠 NLP Educational Projects

---

## 📦 Customize the Model

In `modelimp.py`, you can easily customize model on following parameters :

```python
embed_size
num_layers
heads
forward_expansion
dropout
max_length
```

---

## 🧠 Learning Objectives

This repository is perfect for:

* Understanding the mechanics of attention and self-attention
* Learning how to build transformer models from scratch
* Preparing for advanced deep learning or NLP research

---

## 🧰 To-Do & Future Work

* [ ] Add positional encoding visualization
* [ ] Add training loop and evaluation metrics
* [ ] Add dataset support (e.g., IWSLT, WMT)
* [ ] Add support for inference using greedy / beam search
* [ ] Export model to ONNX for deployment

---

## 🙌 Contributing

Feel free to fork this project and submit a pull request for any improvements or bug fixes.

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Kartik Sharma**


>  For education, experimentation, and future production-level ideas.

```


