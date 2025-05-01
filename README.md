# Vision Transformer (ViT) 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1sn2k2a6M3hw6X8_wzrWEWijXNzowbEzl?usp=sharing)

> An implementation of the Vision Transformer (ViT) architecture, originally proposed in *[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)* by Dosovitskiy et al.

## 🚀 Overview

The Vision Transformer (ViT) applies a Transformer architecture directly to image patches and achieves state-of-the-art performance on image classification tasks, rivalling convolutional neural networks (CNNs). This repository contains a clean and modular implementation of ViT in PyTorch.

## 🧠 Architecture Highlights

- Images are split into fixed-size patches.
- Patches are linearly embedded and fed into a standard Transformer encoder.
- A learnable classification token is used for the final prediction.
- Positional embeddings maintain the spatial structure of patches.

## 📦 Features

- Modular and readable PyTorch implementation.
- Training and evaluation pipelines.
- Pretrained model support (if available).
- Configurable patch size, embedding dimension, and number of heads/layers.

## 📝 Table of Contents

- [Installation](#installation)
- [Model Architecture](#model-architecture)
- [Train Dataset Explanation](#train-dataset-explanation)
- [Results](#results)
- [Colab Demo](#colab-demo)
- [Citation](#citation)

## 🛠️ Installation

```bash
git clone https://github.com/datasciritwik/vit.git
cd vit
```

## 🧩 Model Architecture

```
VisionT(
  (embeddings_block): PatchEmbedding(
    (patcher): Sequential(
      (0): Conv2d(1, 16, kernel_size=(4, 4), stride=(4, 4))
      (1): Flatten(start_dim=2, end_dim=-1)
    )
    (dropout): Dropout(p=0.001, inplace=False)
  )
  (encoder_blocks): TransformerEncoder(
    (layers): ModuleList(
      (0-3): 4 x TransformerEncoderLayer(
        (self_attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=16, out_features=16, bias=True)
        )
        (linear1): Linear(in_features=16, out_features=768, bias=True)
        (dropout): Dropout(p=0.001, inplace=False)
        (linear2): Linear(in_features=768, out_features=16, bias=True)
        (norm1): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
        (norm2): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
        (dropout1): Dropout(p=0.001, inplace=False)
        (dropout2): Dropout(p=0.001, inplace=False)
      )
    )
  )
  (mlp_head): Sequential(
    (0): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
    (1): Linear(in_features=16, out_features=10, bias=True)
  )
)
```

## 🧾 Train Dataset Explanation

The `MNISTTrainDataset` class is a custom PyTorch Dataset used to train the Vision Transformer model on the MNIST dataset.

```python
class MNISTTrainDataset(Dataset):
  def __init__(self, images, labels, indicies):
    self.images = images
    self.labels = labels
    self.indicies = indicies
    self.transform = transforms.Compose([
      transforms.ToPILImage(),
      transforms.RandomRotation(15),
      transforms.ToTensor(),
      transforms.Normalize([0.5], [8.5])
    ])

  def __len__(self):
    return len(self.images)

  def __getitem__(self, idx):
    image = self.images[idx].reshape((28, 28)).astype(np.uint8)
    label = self.labels[idx]
    index = self.indicies[idx]
    image = self.transform(image)

    return {"image": image, "label": label, "index": index}
```

### 📌 Explanation:
- `images`, `labels`, and `indices` are arrays of MNIST data.
- Applies data augmentation (rotation) and normalization for robustness.
- Reshapes and transforms each image.
- Returns a dictionary for each item: image tensor, label, and original index.

## 🔗 Colab Demo

Try out the Vision Transformer in your browser using Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1sn2k2a6M3hw6X8_wzrWEWijXNzowbEzl?usp=sharing)

## 📚 Citation

If you use this code or model, please cite the original paper:

```bibtex
@article{dosovitskiy2020image,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and et al.},
  journal={arXiv preprint arXiv:2010.11929},
  year={2020}
}
```

## 🙌 Contributing

Contributions are welcome! Please open an issue or pull request if you find a bug or want to add a feature.

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for more details.

---

Let me know if you'd like a version tailored to TensorFlow or Hugging Face Transformers.
