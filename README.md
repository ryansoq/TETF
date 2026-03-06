# TETF — Tiny Embedded Training Framework

> 從零手刻的深度學習框架。不依賴 PyTorch、不依賴 TensorFlow，純 C++ 實現前向傳播、反向傳播、梯度更新。

![Training Result](https://github.com/ryansoq/TETF/blob/master/Training%20result.png?raw=true)

---

## 為什麼做這個？

市面上的深度學習框架（PyTorch、TensorFlow）都是黑盒子 — `loss.backward()` 一行搞定，但你真的知道梯度怎麼流的嗎？

TETF 把每一層的數學攤開給你看：**每個 op 的 forward 怎麼算、backward 怎麼推、梯度怎麼傳**。全部手寫，沒有魔法。

如果你喜歡 [Karpathy 的 micrograd](https://github.com/karpathy/micrograd)，TETF 是它的進階版 — 從 scalar 升級到 tensor，從 MLP 升級到 CNN + Transformer，從 Python 教具升級到 C++ 實戰。

---

## 架構總覽

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

---

## 核心概念

### 1. Node — 計算圖的最小單位

```cpp
class node {
    float val;                              // 前向值
    float diff;                             // 反向梯度
    vector<pair<float, node*>> diffs;       // 偏微分邊：(∂f/∂x, 上游節點)
};
```

每個 node 記錄自己的值和梯度。`diffs` 是這個 node 對上游的偏微分連結 — 這就是計算圖的「邊」。

### 2. 基本運算 — 前向 + 反向一起定義

以乘法為例：

```cpp
void mul(node *output, node *input1, node *input2) {
    // 反向：記錄偏微分（鏈式法則的一環）
    //   ∂(x*y)/∂x = y  →  input1 的邊指向 output，權重 = input2.val
    //   ∂(x*y)/∂y = x  →  input2 的邊指向 output，權重 = input1.val
    input1->diffs.push_back({input2->val, output});
    input2->diffs.push_back({input1->val, output});

    // 前向：計算輸出
    output->val = input1->val * input2->val;
}
```

加法更直覺：`∂(x+y)/∂x = 1`，`∂(x+y)/∂y = 1`，所以兩條邊的權重都是 1。

### 3. 梯度回傳 — 遞迴走計算圖

```cpp
float get_diff(node *src, node *dst) {
    if (src == dst) return 1;           // 自己對自己的微分 = 1
    
    src->diff = 0;
    for (auto &edge : src->diffs) {
        // 鏈式法則：∂L/∂x = Σ (∂f/∂x × ∂L/∂f)
        src->diff += edge.first * get_diff(edge.second, dst);
    }
    return src->diff;
}
```

這就是 backpropagation 的本質 — 從 loss 出發，沿著 `diffs` 邊遞迴回推每個參數的梯度。

### 4. Net — 串起所有 Op

```cpp
class Net {
    std::list<opBase*> Layer;       // 有序的 op 列表

    void forward()  { for (auto it = Layer.begin();  ...) (*it)->forward();  }
    void backward() { for (auto it = Layer.rbegin(); ...) (*it)->backward(); }
    void update()   { for (auto it = Layer.rbegin(); ...) (*it)->update();   }
};
```

前向：從頭走到尾。反向：從尾走到頭。就這麼簡單。

---

## 支援的 Op

| 類別 | Op | forward | backward |
|------|-----|---------|----------|
| **基礎** | `add` | z = x + y | ∂z/∂x = 1, ∂z/∂y = 1 |
| | `mul` | z = x × y | ∂z/∂x = y, ∂z/∂y = x |
| | `sub` | z = x - y | ∂z/∂x = 1, ∂z/∂y = -1 |
| | `div` | z = x / y | ∂z/∂x = 1/y, ∂z/∂y = -x/y² |
| **張量** | `Matmul` | C = A × B | ∂L/∂A = ∂L/∂C × Bᵀ |
| | `Add` (broadcast) | C = A + bias | 梯度直傳 |
| | `Conv2d` | IM2COL + GEMM | 反向卷積 |
| | `MaxPool` | 取最大值 | 梯度只回傳給 max 位置 |
| **激活** | `ReLU` | max(0, x) | x > 0 ? 1 : 0 |
| | `Sigmoid` | 1/(1+e⁻ˣ) | σ(x)(1-σ(x)) |
| | `Leaky_ReLU` | x > 0 ? x : αx | x > 0 ? 1 : α |
| **損失** | `MSE` | Σ(y-ŷ)²/n | 2(ŷ-y)/n |
| | `CrossEntropy` | -Σ yᵢlog(softmax(xᵢ)) | softmax(x) - y |
| **Transformer** | `ScaledDotProductAttention` | softmax(QKᵀ/√d)V | 完整注意力梯度 |
| | `MultiHeadAttention` | 多頭拼接 + 線性投影 | 各頭獨立反向 |
| | `LayerNorm` | (x-μ)/σ × γ + β | 含 γ, β 梯度 |
| | `TransformerBlock` | Attention + FFN + Residual | 殘差連接梯度直通 |

---

## 模型架構

### CNN 模式（LeNet）

```
Input [1,1,28,28]
  → Conv 6@5×5 → ReLU → MaxPool 2×2
  → Conv 16@5×5 → ReLU → MaxPool 2×2
  → Reshape [1,400]
  → FC 400→120 → Sigmoid
  → FC 120→10
  → CrossEntropy Loss
```

### Transformer 模式（CNN + Transformer Hybrid）

```
Input [1,1,28,28]
  → Conv 6@5×5 → ReLU → MaxPool 2×2
  → Conv 16@5×5 → ReLU → MaxPool 2×2
  → Reshape [1,400]
  → Linear 400→64
  ┌─── transformer_block ───┐
  │ → Self-Attention (W_V)  │
  │ → Residual Add          │
  │ → FFN 64→128→64 (ReLU) │
  │ → Residual Add          │
  └─────────────────────────┘
  → Linear 64→10
  → CrossEntropy Loss
```

CNN 負責提取局部特徵，Transformer block 學習全局關係 — 這跟 ViT 的思路類似，但更輕量。

---

## 與 micrograd 的比較

|  | [micrograd](https://github.com/karpathy/micrograd) | TETF |
|--|-----------|------|
| 語言 | Python | C++ |
| 粒度 | scalar（一個浮點數 = 一個節點） | tensor（矩陣運算） |
| 建圖方式 | 隱式（operator overloading 自動追蹤） | 顯式（手動 `AddLayer` 排序） |
| 拓撲排序 | backward 時 DFS 自動算 | 不需要 — Layer list 本身就是順序 |
| 支援 Op | add, mul, pow, relu | Conv, MaxPool, Matmul, Attention, LayerNorm... |
| 能跑的模型 | MLP（二分類） | LeNet CNN + Transformer（MNIST 手寫辨識） |
| NNEF 匯出 | ❌ | ✅ |
| 量化 | ❌ | ✅ INT8 (q7) |

micrograd 是樂高積木拼的模型車 — 幫你理解「車有輪子」。
TETF 是手工鍛造的引擎 — 每個零件都是自己做的，而且真的能跑。

---

## 快速開始

```bash
git clone https://github.com/ryansoq/TETF.git
cd TETF
make

# CNN 模式（LeNet）
./tetf

# Transformer 模式
./tetf --transformer

# 跑 Transformer op 單元測試
./tetf --test
```

需要 MNIST 資料集在 `./third_party/mnist/` 目錄下。

---

## NNEF Code-Gen

TETF 可以把訓練好的模型匯出為 [NNEF](https://www.khronos.org/nnef)（Neural Network Exchange Format）：

```
version 1.0;
graph network( external1 ) -> ( matmul5 )
{
    external1 = external(shape = [1, 1, 28, 28]);
    conv3 = conv(external1, variable2, ...);
    relu4 = relu(conv3);
    ...
}
```

這讓模型可以部署到支援 NNEF 的嵌入式推理引擎上 — 這也是 "Embedded" 這個名字的由來。

---

## 專案歷史

這個專案始於 2019 年，目標是理解深度學習的每一個環節 — 不是調 API，而是從矩陣乘法的偏微分開始，一路手刻到能跑 MNIST 的 CNN。

2025 年加入 Transformer 支援，把 Attention、LayerNorm、殘差連接全部用基底 op 組裝起來，驗證了框架的可擴展性。

---

## License

MIT
