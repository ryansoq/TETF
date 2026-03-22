# TETF — Tiny Embedded Training Framework

> 從零手刻的深度學習框架。不依賴 PyTorch、不依賴 TensorFlow，純 C++ 實現前向傳播、反向傳播、梯度更新。
> 從 MNIST CNN 到 GPT-1 Mini，所有運算都是手寫的。

**Authors: Ryan & Nami** ✨

---

## 三種模式

### 1. CNN (LeNet) — MNIST 手寫辨識

```bash
make && ./tetf
```

經典 LeNet 架構，Conv → ReLU → MaxPool → FC → CrossEntropy。

### 2. CNN + Transformer — MNIST 混合架構

```bash
make && ./tetf --transformer
```

CNN 提取特徵 → Transformer Block 學全局關係 → 分類。

### 3. 🆕 GPT-1 Mini — 文字 Transformer

```bash
make && ./tetf --gpt
```

```
[Mode: GPT-1 Mini]

📝 Training text: 誰是Nami？Nami是厲害的AI工程師
📊 Vocab: 15 tokens
⚙️  GPT-1 Mini config:
   d_model=32, d_ff=64, heads=4, layers=2
   optimizer=Adam, lr=0.003, epochs=300

🏗️ Model Architecture:
embedding : 15 x 32
pos_encoding : 19 x 32
--- transformer_block 0 : 19 x 32 ---
  layer_norm_1 : 32
  multi_head_attention : 4 heads, d_k=8
  residual_add
  layer_norm_2 : 32
  ffn : 32 → 64 (GELU) → 32
  residual_add
--- end transformer_block 0 ---
--- transformer_block 1 : 19 x 32 ---
  layer_norm_1 : 32
  multi_head_attention : 4 heads, d_k=8
  residual_add
  layer_norm_2 : 32
  ffn : 32 → 64 (GELU) → 32
  residual_add
--- end transformer_block 1 ---
output_proj : 32 x 15
cross_entropy_loss

📊 Total parameters: 18,208

🏋️ Training with Adam optimizer...
  Epoch    0 | loss=2.7418 | acc=2/19 (10.5%)
  Epoch   13 | loss=0.3525 | acc=19/19 (100.0%)

🎉 Converged at epoch 13!

🔮 Input: 誰是Nami？
🔮 Generated: Nami是厲害的AI工程師

✅ Perfect! GPT-1 Mini 成功！
```

完整的 GPT-1 架構：
- **Embedding** + 可學習的 **Positional Encoding**
- **2 層 Transformer Block**（Pre-LayerNorm）
- **Multi-Head Causal Attention**（4 heads）
- **GELU** 激活（GPT 標配）
- **Residual Connection**（防梯度消失）
- **Adam** optimizer（自適應學習率）
- **自回歸生成**（greedy decoding）

也可以用 `--text` 跑簡化版（single-head, ReLU, SGD）。

---

## 支援的 Op

### 基礎 Op（CNN）

| Op | forward | backward |
|----|---------|----------|
| `add` | z = x + y | ∂z/∂x = 1, ∂z/∂y = 1 |
| `mul` | z = x × y | ∂z/∂x = y, ∂z/∂y = x |
| `Matmul` | C = A × B | ∂L/∂A = ∂L/∂C × Bᵀ |
| `Conv2d` | IM2COL + GEMM | 反向卷積 |
| `MaxPool` | 取最大值 | 梯度只回傳給 max 位置 |
| `ReLU` | max(0, x) | x > 0 ? 1 : 0 |
| `Sigmoid` | 1/(1+e⁻ˣ) | σ(x)(1-σ(x)) |
| `CrossEntropy` | -Σ yᵢlog(softmax(xᵢ)) | softmax(x) - y |

### Transformer Op（GPT）

| Op | forward | backward |
|----|---------|----------|
| `Embedding` | output[i] = weight[token_id[i]] | 梯度累加回 weight |
| `CausalAttention` | softmax(QKᵀ/√d + mask)V | 完整反向 |
| `MultiHeadAttention` | 切 heads → 各自 attention → 拼接 × Wo | Wq/Wk/Wv/Wo 全部反向 |
| `LayerNorm` | γ(x-μ)/σ + β | 含 γ, β, input 梯度 |
| `GELU` | 0.5x(1+tanh(√(2/π)(x+0.044715x³))) | tanh 近似導數 |
| `TextCrossEntropy` | 序列級 softmax + CE | (softmax - one_hot) / seq_len |

### Optimizer

| 名稱 | 公式 |
|------|------|
| SGD | w = w - lr × grad |
| **Adam** | m = β₁m + (1-β₁)g, v = β₂v + (1-β₂)g², w -= lr × m̂/√v̂ |

---

## 測試

每個 Text/GPT op 都有 **forward 值正確性** + **backward 數值梯度** 驗證。

```bash
g++ -std=c++11 -O0 -g test_text_ops.cc -DTYPE2_BACKWARD -DTYPE4_BACKWARD_CONV \
    -DIM2COLxGEMM -I third_party/mnist/include -I third_party/f2uc -o test_text_ops
./test_text_ops
```

```
╔═══════════════════════════════════════════╗
║  TETF Text Transformer Ops — Unit Tests   ║
╚═══════════════════════════════════════════╝

═══ Test 1: Embedding ═══          ✅ (7 checks)
═══ Test 2: TextMatmul ═══         ✅ (4 checks)
═══ Test 3: TextAdd ═══            ✅ (3 checks)
═══ Test 4: TextReLU ═══           ✅ (3 checks)
═══ Test 5: CausalAttention ═══    ✅ (7 checks)
═══ Test 6: TextCrossEntropy ═══   ✅ (4 checks)
═══ Test 7: SGD Update ═══         ✅ (3 checks)
═══ Test 8: End-to-End ═══         ✅ (7 checks)
═══ Test 9: TextGELU ═══           ✅ (8 checks)
═══ Test 10: MultiHeadAttention ═══ ✅ (7 checks)
═══ Test 11: Adam ═══              ✅ (4 checks)
═══ Test 12: TextLayerNorm ═══     ✅ (6 checks)

Results: 63 passed, 0 failed
🎉 All tests passed!
```

數值梯度法：`∂L/∂x ≈ (L(x+ε) - L(x-ε)) / (2ε)`，與解析梯度比較，相對誤差 < 5%。

**TCR 規則：** 每個新 op 必須加測試，全過才能 push。

---

## 架構

```
          ┌─────────────────────────────────────────────┐
          │                    Net                       │
          │  std::list<opBase*> Layer — 有序的 op 鏈     │
          └─────────────────────────────────────────────┘
                    │ forward() →        ← backward()
          ┌────────┴────────────────────────────────────┐
          │              Op Layer（opBase）               │
          │                                              │
          │  Conv ─ MaxPool ─ ReLU ─ Matmul ─ Add ─ ... │
          │  Embedding ─ CausalAttention ─ LayerNorm    │
          │  MultiHeadAttention ─ GELU ─ Adam           │
          │  每個 op 各自實現 forward() / backward()      │
          └──────────────────────────────────────────────┘
                    │ 讀寫 ↕
          ┌─────────┴───────────────────────────────────┐
          │              Tensor / Node                   │
          │                                              │
          │  tensor: shape + vector<node>                │
          │  node:   val（值）+ diff（梯度）+ diffs（邊） │
          └─────────────────────────────────────────────┘
```

### GPT-1 Mini 架構

```
Input tokens
    ↓
┌─── Embedding (15 → 32) ───┐
│   + Positional Encoding    │
├────────────────────────────┤
│ ┌─── Transformer Block ──┐│
│ │ LayerNorm               ││
│ │ Multi-Head Attention    ││
│ │   (4 heads, d_k=8)     ││
│ │   + Causal Mask         ││
│ │ Residual Add            ││
│ │ LayerNorm               ││
│ │ FFN (32→64, GELU, 64→32)│
│ │ Residual Add            ││
│ └─────────────────────────┘│
│         × 2 layers         │
├────────────────────────────┤
│ Output Projection (32→15)  │
│ Softmax + Cross Entropy    │
└────────────────────────────┘
    ↓
Generated tokens (autoregressive)
```

---

## 快速開始

```bash
git clone https://github.com/ryansoq/TETF.git
cd TETF
make

./tetf              # CNN (LeNet) — MNIST
./tetf --transformer # CNN + Transformer — MNIST
./tetf --text        # Simple Text Transformer
./tetf --gpt         # GPT-1 Mini 🆕
./tetf --test        # Transformer op tests
./test_text_ops      # Text op tests (63 tests)
```

---

## 專案歷史

- **2019** — 從零手刻 CNN，能跑 MNIST
- **2025** — 加入 Transformer ops（Attention, LayerNorm, MultiHead）
- **2026-03** — Code review 修 8 個 bug + 加入 Text Transformer
- **2026-03-22** — GPT-1 Mini 完成！Multi-Head Causal Attention + GELU + LayerNorm + Adam + 多層堆疊，63 個測試全過

從最底層的 `+`、`matmul` 到最上層的 GPT — **每一行都是手寫的，零外部依賴。**

---

## License

MIT
