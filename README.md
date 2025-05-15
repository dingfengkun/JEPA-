
# JEPA: Joint Embedding Predictive Architecture (PyTorch)

This repository implements a JEPA‑style self‑supervised learner: it predicts **latent representations** of masked portions of an input (image or video) from the **context embeddings**, avoiding costly pixel reconstruction and learning high‑level semantics.

---

## 🚀 Key Features

- **Latent‑space prediction**: Predict only embedding vectors of masked patches, not raw pixels.  
- **Asymmetric dual encoders**:  
  - **Context encoder** (trainable)  
  - **Target encoder** (EMA of context) for stable “teacher” targets.  
- **Block‑wise masking**: Randomly mask multiple patches; predictor reconstructs them from context.  
- **Lightweight predictor head**: Small MLP or transformer head focuses training on encoder quality.  
- **Flexible backbone**: Swap between CNNs, ResNets, ViTs, or even energy transformers.  
- **Optional extra losses**: Huber/MSE on embeddings, cycle‑consistency, VICReg invariance/covariance.

---

## 📁 Repository Structure

```

DL25SP/
├── configs/                 # YAML files for experiments
├── dataset.py               # data loading & block‑masking utilities
├── jepa\_model.py            # context encoder, target encoder (EMA), predictor
├── losses.py                # embedding prediction losses (MSE, Huber…)
├── train\_jepa.py            # training loop, EMA updates, logging
├── evaluator.py             # linear probe / KNN evaluation on frozen encoder
├── utils.py                 # misc helpers (EMA, metrics, checkpoints)
└── README.md                # this documentation

````

---

## ⚙️ Installation

```bash
git clone https://github.com/dingfengkun/JEPA.git
cd JEPA
pip install -r requirements.txt
````

Dependencies include: `torch`, `torchvision`, `pyyaml`, `einops`, `tqdm`.

---

## 🏋️‍♂️ Training

Edit or select a config in `configs/`. A minimal example:

```yaml
# configs/imagenet_vit.yaml
backbone: vit_base_patch16_224
mask_ratio: 0.5
batch_size: 256
learning_rate: 1e-4
epochs: 100
ema_decay: 0.99
loss_type: huber
```

Launch training:

```bash
python train_jepa.py --config configs/imagenet_vit.yaml
```

**What happens inside**:

1. Sample a batch, randomly mask patches per image.
2. Encode **context** patches with the trainable encoder.
3. Encode **target** (small patches) with the EMA encoder.
4. Predictor takes context embeddings (+ positional info) → predicts target embeddings.
5. Compute Huber/MSE loss between predicted vs. EMA‑encoded target.
6. Backprop on context encoder + predictor; update EMA encoder.

---

## 📊 Evaluation

After pretraining, freeze **context encoder** and run a **linear probe** or **KNN** on downstream classification:

```bash
python evaluator.py --model_path checkpoints/best.pth \
                    --dataset cifar10 \
                    --eval_mode linear_probe
```

---

## 🛠️ Tips for lowering loss further

* **Backbone**: upgrade to a larger ViT or ResNet gives richer embeddings.
* **Mask scheduling**: gradually increase the mask ratio over epochs.
* **Loss**: try Huber over MSE if you see large outliers.
* **Additional regularizers**: add cycle‑consistency or VICReg terms (see community forks).
* **Optimizer**: AdamW with cosine scheduler often yields smoother convergence.

---

## 🔖 Citation

If you use this code, please cite:

```bibtex
@article{assran2023self,
  title={Self‑Supervised Learning from Images with a Joint‑Embedding Predictive Architecture},
  author={Assran, Mahmoud and Duval, Quentin and Misra, Ishan and Bojanowski, Piotr and Vincent, Pascal and Rabbat, Michael and LeCun, Yann and Ballas, Nicolas},
  journal={arXiv preprint arXiv:2301.08243},
  year={2023}
}
```

---

That README captures all the **core mechanisms** (latent prediction, EMA, masking) that drive down loss, and gives users concrete guidance on structure, training, evaluation, and further tweaks. Let me know if you’d like to include example training curves or code snippets inside!
::contentReference[oaicite:5]{index=5}
```

[1]: https://github.com/facebookresearch/ijepa?utm_source=chatgpt.com "facebookresearch/ijepa: Official codebase for I-JEPA, the ... - GitHub"
[2]: https://github.com/gaasher/I-JEPA?utm_source=chatgpt.com "gaasher/I-JEPA - GitHub"
[3]: https://github.com/LumenPallidium/jepa?utm_source=chatgpt.com "Experiments in Joint Embedding Predictive Architectures (JEPAs)."
