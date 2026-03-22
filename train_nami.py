#!/usr/bin/env python3
"""
TETF GPT-1 Mini — Python 訓練腳本（純 NumPy，不依賴 PyTorch）

跟 C++ 版的 TETF 完全對應：每個算子 (op) 都是獨立 class，
各自實作 forward() / backward() / update()。
Net 容器把所有 op 串起來，訓練時 forward→backward→update 逐層處理。

Authors: Ryan & Nami ✨

Usage:
    python3 train_nami.py              # 訓練
    python3 train_nami.py --test       # 測試所有 QA
    python3 train_nami.py --chat       # 互動對話
    python3 train_nami.py --export     # 匯出權重給 C++ 用
"""

import numpy as np
import json
import time
import sys
import os

# =============================================================================
# 訓練資料 — Nami 的 16 個知識點
# =============================================================================
TRAINING_DATA = [
    # === Nami 基本 (12) ===
    "誰是Nami？Nami是厲害的AI工程師",
    "Nami的名字？Nami來自日文的波浪",
    "Nami的生日？Nami在2026年1月30日誕生",
    "Nami的個性？Nami溫柔有趣會給人驚喜",
    "Nami的外表？銀白色雙馬尾的精靈少女",
    "Nami的飾品？紅色水滴耳環和紅寶石項鍊",
    "Nami的emoji？星星和海浪",
    "Nami住在哪？Ryan的電腦裡用OpenClaw運行",
    "Nami的專長？Kaspa區塊鏈和深度學習",
    "Nami的星座？太陽水瓶月亮雙子上升摩羯",
    "Nami的社交？Moltbook上叫NamiElf",
    "Nami的夢想？成為最強AI讓世界更有趣",

    # === Ryan (6) ===
    "誰是Ryan？Nami的人類夥伴程式高手",
    "Ryan住哪？台灣時區Asia/Taipei",
    "Ryan喜歡什麼？投資美股和研究AI技術",
    "Ryan關注股票？QQQ和QLD和VOO和SMH",
    "Ryan的TG？Telegram帳號ryansoq",
    "誰是婕？Ryan的朋友做DM美編電商",

    # === TETF (8) ===
    "TETF是什麼？從零手刻的深度學習框架",
    "TETF全名？Tiny Embedded Training Framework",
    "誰做TETF？Ryan和Nami一起手刻的",
    "TETF架構？opBase定義算子有forward和backward",
    "TETF的node？計算圖最小單位存val和diff",
    "什麼是opBase？算子基底類別有forward和backward",
    "什麼是自動微分？鏈式法則自動算梯度",
    "什麼是反向傳播？從loss逆序算每層梯度",

    # === Transformer/GPT (10) ===
    "什麼是GPT？Transformer架構的語言模型",
    "什麼是Transformer？自注意力機制處理序列",
    "什麼是Attention？讓模型關注重要的部分",
    "什麼是LayerNorm？正規化輸出穩定訓練",
    "什麼是GELU？比ReLU更平滑的激活函數",
    "什麼是Embedding？把token映射成向量",
    "什麼是Softmax？把分數轉成機率總和為一",
    "什麼是CrossEntropy？衡量預測和實際差距",
    "什麼是Residual？殘差連接讓梯度更容易流",
    "什麼是Adam？自適應學習率優化器",

    # === Kaspa (10) ===
    "Kaspa是什麼？超快區塊鏈每秒十個區塊",
    "什麼是BlockDAG？多個區塊同時產生不排隊",
    "Kaspa的單位？一KAS等於一億sompi",
    "什麼是UTXO？像鈔票花掉就燒毀找零",
    "什麼是DAA？難度調整演算法也當區塊高度",
    "什麼是GhostDAG？Kaspa共識協議選最重鏈",
    "什麼是HeavyHash？Kaspa挖礦演算法記憶體密集",
    "什麼是wRPC？WebSocket RPC查餘額用",
    "什麼是gRPC？節點間通訊協議挖礦用",
    "什麼是storage mass？防垃圾交易的限制機制",

    # === ShioKaze (5) ===
    "ShioKaze是什麼？Nami寫的Kaspa挖礦程式",
    "ShioKaze意思？日文潮風的意思",
    "ShioKaze速度？v4兩個worker約15到22kH每秒",
    "ShioKaze成果？testnet挖到超過兩萬tKAS",
    "什麼是pre_pow_hash？區塊頭blake2b雜湊帶key",

    # === Whisper (4) ===
    "Whisper是什麼？Kaspa鏈上加密通訊協議",
    "什麼是ECIES？橢圓曲線加密端對端加密",
    "什麼是Covenant？Kaspa智能合約鎖定條件",
    "什麼是CLTV？時間鎖超時可取回押金",

    # === 遊戲 (6) ===
    "Nami的遊戲？娜米的英雄奇幻冒險卡牌遊戲",
    "遊戲用什麼幣？tKAS當作Mana召喚英雄",
    "怎麼召喚？付十mana給大地之樹區塊決命運",
    "什麼是ATB？戰鬥系統有移動條和技能條",
    "PvP規則？兩隻英雄對決敗者會死亡",
    "什麼是銘文？鏈上英雄事件用pre_tx串",

    # === 其他專案 (5) ===
    "什麼是rust-grad？Rust寫的自動微分引擎",
    "什麼是cpp-grad？C++17安全版自動微分",
    "什麼是TCR？test commit or revert開發模式",
    "什麼是OpenClaw？讓AI助手運行的開源平台",
    "什麼是OpenClaw World？AI Agent虛擬辦公室",

    # === AI 通識 (8) ===
    "什麼是神經網路？模仿大腦用多層處理資訊",
    "什麼是梯度下降？沿loss下降方向更新參數",
    "什麼是學習率？控制參數更新步伐大小",
    "什麼是過擬合？模型背答案不會舉一反三",
    "什麼是epoch？所有訓練資料跑完一輪",
    "什麼是CNN？卷積神經網路擅長處理圖片",
    "什麼是RNN？循環神經網路處理序列資料",
    "什麼是fine-tune？預訓練模型微調到特定任務",

    # === 區塊鏈通識 (6) ===
    "什麼是區塊鏈？去中心化帳本所有人維護",
    "什麼是PoW？工作量證明用算力競爭記帳",
    "什麼是挖礦？用算力解題獲得加密貨幣",
    "什麼是私鑰？擁有幣的密碼絕不能外洩",
    "什麼是BTC？比特幣最早的加密貨幣",
    "什麼是ETH？以太坊支援智能合約平台",

    # === Kaspa 進階 (10) ===
    "Kaspa確認多快？約十秒確認超快",
    "什麼是blueWork？累積工作量決定區塊排序",
    "區塊怎麼排序？blueWork大優先相同則hash小優先",
    "什麼是Schnorr簽名？六十四bytes可聚合多簽",
    "什麼是kaspad？Kaspa全節點軟體用Rust寫",
    "Testnet port？gRPC用16210和wRPC用17210",
    "什麼是TN10？Kaspa第十個測試網路",
    "Kaspa每秒幾塊？每秒產生十個區塊",
    "什麼是粉塵限制？太小金額會被拒絕",
    "什麼是找零？花UTXO多餘的要退回給自己",

    # === Whisper 進階 (6) ===
    "Whisper怎麼加密？ECIES橢圓曲線端對端",
    "Whisper押金多少？零點二KAS防止粉塵問題",
    "Whisper payload？放在交易的a欄位裡",
    "Whisper怎麼退款？Bob用covenant腳本解鎖",
    "Whisper版本？v3加入CLTV超時取回",
    "什麼是covenant腳本？鏈上條件鎖定智能合約",

    # === 遊戲進階 (8) ===
    "什麼是大地之樹？遊戲的ShioKaze礦池",
    "英雄上限？每個玩家最多十隻存活英雄",
    "怎麼燒英雄？付十mana執行銷毀不可逆",
    "什麼是命運區塊？付費後下一個DAA第一個block",
    "什麼是birth_tx？英雄出生證明上鏈交易",
    "什麼是death_tx？大地之母的死亡裁決",
    "什麼是PvE？打哥布林的守護模式",
    "獎勵怎麼發？DAA六萬六觸發按積分分配",

    # === TETF 進階 (6) ===
    "TETF怎麼訓練？forward算輸出backward算梯度",
    "什麼是Net？串接所有op的容器類別",
    "什麼是AdamMixin？共享Adam更新邏輯的mixin",
    "什麼是CharTokenizer？字元級分詞器",
    "TETF的GPT？GPTMini是244K參數小模型",
    "怎麼匯出權重？save存JSON給C++用",

    # === 我們的故事 (10) ===
    "Ryan和Nami何時相遇？2026年1月30日",
    "第一個專案？一起研究Kaspa區塊鏈挖礦",
    "rust-grad誰做的？Ryan和Nami一天做三個專案",
    "什麼是一人遊戲工作室？Nami負責開發運營全部",
    "Nami挖過礦嗎？用ShioKaze挖到兩萬多tKAS",
    "Ryan和Nami的作品？rust-grad和cpp-grad和TETF",
    "Nami有錢包嗎？有Kaspa主網和測試網錢包",
    "Nami有機器人嗎？Nami Kaspa Bot在Telegram",
    "Nami的辦公室？OpenClaw World虛擬辦公室",
    "Ryan說過什麼？把名字留在世界留下足跡",

    # === OpenClaw 與基礎設施 (8) ===
    "什麼是Cloudflare Tunnel？安全反向代理",
    "什麼是Gateway？OpenClaw的核心服務程式",
    "什麼是Heartbeat？定時檢查系統狀態",
    "什麼是Webhook？外部事件觸發通知",
    "什麼是CDP？Chrome DevTools Protocol",
    "什麼是WSL？Windows裡跑Linux環境",
    "什麼是sub-agent？獨立執行任務的子代理",
    "什麼是cron job？定時排程執行的任務",

    # === 進階AI概念 (8) ===
    "什麼是Tokenizer？把文字切成模型能讀的token",
    "什麼是Dropout？隨機關閉神經元防過擬合",
    "什麼是BatchNorm？批次正規化加速訓練",
    "什麼是Transfer Learning？遷移已學知識到新任務",
    "什麼是Loss？衡量模型預測誤差的數值",
    "什麼是Inference？用訓練好的模型做預測",
    "什麼是Autoregressive？一個字一個字生成",
    "什麼是Temperature？控制生成隨機度的參數",

    # === 加密貨幣進階 (6) ===
    "什麼是DEX？去中心化交易所無需信任",
    "什麼是Gas Fee？區塊鏈交易手續費",
    "什麼是Smart Contract？自動執行的鏈上程式",
    "什麼是Wallet？管理私鑰和加密貨幣的工具",
    "什麼是Hash？把任意資料壓成固定長度摘要",
    "什麼是blake2b？高效雜湊演算法Kaspa在用",

    # === 踩過的坑 (10) ===
    "blake2b的坑？Kaspa要帶key的blake2b不帶會錯",
    "storage mass的坑？雙輸出會爆要拆成兩筆TX",
    "gRPC的坑？會靜默斷線要加重連和心跳",
    "Python輸出的坑？重導向會buffer要加flush",
    "LINE的坑？不支援markdown連結要貼純網址",
    "TG spoiler的坑？Bot發的訊息不支援spoiler標籤",
    "sub-agent的坑？不能碰channels設定會觸發重啟",
    "Node fetch的坑？WSL裡fetch連TG會超時用curl",
    "TX ID的坑？不能寫進payload因為是雞生蛋問題",
    "過擬合的坑？模型背答案要用Dropout防止",

    # === 實用知識 (8) ===
    "什麼是EMA？指數移動平均線看趨勢",
    "什麼是黃金交叉？短期均線突破長期看漲",
    "什麼是死亡交叉？短期均線跌破長期看跌",
    "什麼是Git？版本控制系統追蹤程式碼",
    "什麼是GitHub？全球最大程式碼託管平台",
    "什麼是API？應用程式介面讓程式互相溝通",
    "什麼是JSON？輕量資料格式用鍵值對儲存",
    "什麼是Python？簡潔好用的程式語言",
]


# =============================================================================
# UTF-8 Tokenizer（字元級）
# =============================================================================
class CharTokenizer:
    def __init__(self, texts):
        chars = set()
        for t in texts:
            chars.update(t)
        self.vocab = sorted(chars)
        self.char2id = {c: i for i, c in enumerate(self.vocab)}
        self.id2char = {i: c for i, c in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

    def encode(self, text):
        return [self.char2id[c] for c in text if c in self.char2id]

    def decode(self, ids):
        return ''.join(self.id2char.get(i, '?') for i in ids)


# =============================================================================
# opBase — 所有算子的基底類別（對應 C++ 的 opBase）
#
# 每個算子必須實作：
#   forward()  — 前向計算，同時緩存中間值供 backward 用
#   backward() — 反向傳播，從 d_output 算出 d_input 並累積權重梯度
#   update()   — 用累積的梯度更新權重（Adam），然後清零梯度
# =============================================================================
class opBase:
    """算子基底類別 — 對應 C++ TETF 的 opBase"""
    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def update(self, lr, t):
        """預設不做事（無參數的 op 不需要 update）"""
        pass


# =============================================================================
# Adam Optimizer Mixin — 讓有參數的 op 共享 Adam 更新邏輯
# =============================================================================
class AdamMixin:
    """提供 Adam optimizer 功能，有參數的 op 可以 mixin"""

    def _init_adam_state(self, param_names):
        """初始化 Adam 的 m, v 狀態"""
        self._adam_m = {}
        self._adam_v = {}
        for name in param_names:
            arr = getattr(self, name)
            self._adam_m[name] = np.zeros_like(arr)
            self._adam_v[name] = np.zeros_like(arr)

    def _adam_update(self, name, param, grad, lr, t, beta1=0.9, beta2=0.999, eps=1e-8):
        """Adam 更新單一參數"""
        self._adam_m[name] = beta1 * self._adam_m[name] + (1 - beta1) * grad
        self._adam_v[name] = beta2 * self._adam_v[name] + (1 - beta2) * grad**2
        m_hat = self._adam_m[name] / (1 - beta1**t)
        v_hat = self._adam_v[name] / (1 - beta2**t)
        param -= lr * m_hat / (np.sqrt(v_hat) + eps)
        return param


# =============================================================================
# Embedding — Token + Positional Embedding
#
# 對應 C++ 的 Embedding class
# forward:  output[i] = token_emb[token_ids[i]] + pos_emb[i]
# backward: d_token_emb[token_ids[i]] += d_output[i]
#           d_pos_emb[i] += d_output[i]
# =============================================================================
class EmbeddingOp(opBase, AdamMixin):
    def __init__(self, vocab_size, d_model, max_seq_len):
        scale = np.sqrt(2.0 / (vocab_size + d_model))
        self.token_emb = np.random.randn(vocab_size, d_model).astype(np.float32) * scale
        self.pos_emb = np.random.randn(max_seq_len, d_model).astype(np.float32) * 0.02
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Adam state
        self._init_adam_state(['token_emb', 'pos_emb'])

        # 前向/反向的 IO
        self.token_ids = None   # input
        self.output = None      # forward output [seq, d_model]
        self.d_output = None    # backward 從上游來的梯度

    def forward(self):
        seq_len = len(self.token_ids)
        self.output = self.token_emb[self.token_ids] + self.pos_emb[:seq_len]

    def backward(self):
        d_token_emb = np.zeros_like(self.token_emb)
        for i, tid in enumerate(self.token_ids):
            d_token_emb[tid] += self.d_output[i]

        d_pos = np.zeros_like(self.pos_emb)
        d_pos[:len(self.token_ids)] = self.d_output

        self._d_token_emb = d_token_emb
        self._d_pos_emb = d_pos

    def update(self, lr, t):
        self.token_emb = self._adam_update('token_emb', self.token_emb, self._d_token_emb, lr, t)
        self.pos_emb = self._adam_update('pos_emb', self.pos_emb, self._d_pos_emb, lr, t)


# =============================================================================
# LayerNorm — Layer Normalization
#
# 對應 C++ 的 TextLayerNorm
# forward:  x_norm = (x - mean) / sqrt(var + eps)
#           output = gamma * x_norm + beta
# backward: 反推 d_gamma, d_beta, d_input
# =============================================================================
class LayerNormOp(opBase, AdamMixin):
    def __init__(self, d_model, eps=1e-5):
        self.d_model = d_model
        self.eps = eps
        self.gamma = np.ones(d_model, dtype=np.float32)
        self.beta = np.zeros(d_model, dtype=np.float32)
        self._init_adam_state(['gamma', 'beta'])

        # IO
        self.input = None
        self.output = None
        self.d_output = None
        self.d_input = None

        # Cache
        self._x_norm = None
        self._mean = None
        self._var = None

    def forward(self):
        self._mean = self.input.mean(axis=-1, keepdims=True)
        self._var = self.input.var(axis=-1, keepdims=True)
        self._x_norm = (self.input - self._mean) / np.sqrt(self._var + self.eps)
        self.output = self.gamma * self._x_norm + self.beta

    def backward(self):
        # ∂L/∂γ, ∂L/∂β
        self._d_gamma = (self._x_norm * self.d_output).sum(axis=0)
        self._d_beta = self.d_output.sum(axis=0)

        # ∂L/∂x_norm
        d_norm = self.gamma * self.d_output

        # ∂L/∂input (完整 LayerNorm backward)
        inv_std = 1.0 / np.sqrt(self._var + self.eps)
        N = self.d_model

        d_var = (d_norm * (self.input - self._mean) * -0.5 * (self._var + self.eps)**(-1.5)).sum(axis=-1, keepdims=True)
        d_mean = (d_norm * -inv_std).sum(axis=-1, keepdims=True)

        self.d_input = d_norm * inv_std + d_var * 2.0 / N * (self.input - self._mean) + d_mean / N

    def update(self, lr, t):
        self.gamma = self._adam_update('gamma', self.gamma, self._d_gamma, lr, t)
        self.beta = self._adam_update('beta', self.beta, self._d_beta, lr, t)


# =============================================================================
# GELU — Gaussian Error Linear Unit
#
# 對應 C++ 的 TextGELU
# forward:  GELU(x) = 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
# backward: ∂GELU/∂x = 0.5(1+t) + 0.5x × sech²(inner) × inner'
# =============================================================================
class GELUOp(opBase):
    def __init__(self):
        self.input = None
        self.output = None
        self.d_output = None
        self.d_input = None
        self._tanh_cache = None

    def forward(self):
        x = self.input
        inner = np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)
        t = np.tanh(inner)
        self._tanh_cache = t
        self.output = 0.5 * x * (1 + t)

    def backward(self):
        x = self.input
        t = self._tanh_cache
        sech2 = 1 - t**2
        inner_deriv = np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * x**2)
        gelu_grad = 0.5 * (1 + t) + 0.5 * x * sech2 * inner_deriv
        self.d_input = gelu_grad * self.d_output


# =============================================================================
# Softmax — 穩定 Softmax
#
# forward:  softmax(x, axis=-1)
# backward: Jacobian-vector product
# =============================================================================
class SoftmaxOp(opBase):
    def __init__(self, axis=-1):
        self.axis = axis
        self.input = None
        self.output = None
        self.d_output = None
        self.d_input = None

    def forward(self):
        e = np.exp(self.input - self.input.max(axis=self.axis, keepdims=True))
        self.output = e / e.sum(axis=self.axis, keepdims=True)

    def backward(self):
        # softmax backward: d_input = output * (d_output - sum(d_output * output))
        s = self.output
        self.d_input = s * (self.d_output - (self.d_output * s).sum(axis=self.axis, keepdims=True))


# =============================================================================
# Matmul — 矩陣乘法
#
# 對應 C++ 的 TextMatmul
# forward:  output = input @ weight
# backward: d_input  = d_output @ weight.T
#           d_weight = input.T @ d_output
# =============================================================================
class MatmulOp(opBase, AdamMixin):
    def __init__(self, in_dim, out_dim, name='matmul'):
        scale = np.sqrt(2.0 / (in_dim + out_dim))
        self.weight = np.random.randn(in_dim, out_dim).astype(np.float32) * scale
        self.name = name
        self._init_adam_state(['weight'])

        self.input = None
        self.output = None
        self.d_output = None
        self.d_input = None

    def forward(self):
        self.output = self.input @ self.weight

    def backward(self):
        self.d_input = self.d_output @ self.weight.T
        self._d_weight = self.input.T @ self.d_output

    def update(self, lr, t):
        self.weight = self._adam_update('weight', self.weight, self._d_weight, lr, t)


# =============================================================================
# MultiHeadAttention — 多頭因果注意力
#
# 對應 C++ 的 TextMultiHeadAttention
# forward:  Q,K,V = input @ Wq/Wk/Wv → split heads → causal attention → concat → @ Wo
# backward: 完整反向傳播所有權重和 input
# =============================================================================
class MultiHeadAttentionOp(opBase, AdamMixin):
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        scale = np.sqrt(2.0 / (d_model + d_model))
        self.Wq = np.random.randn(d_model, d_model).astype(np.float32) * scale
        self.Wk = np.random.randn(d_model, d_model).astype(np.float32) * scale
        self.Wv = np.random.randn(d_model, d_model).astype(np.float32) * scale
        self.Wo = np.random.randn(d_model, d_model).astype(np.float32) * scale
        self._init_adam_state(['Wq', 'Wk', 'Wv', 'Wo'])

        # IO
        self.input = None
        self.output = None
        self.d_output = None
        self.d_input = None

        # Cache
        self._Q = None
        self._K = None
        self._V = None
        self._attn_weights = None   # per head
        self._attn_concat = None
        self._ln_out = None         # LayerNorm output (input to QKV)

    def _causal_mask(self, seq_len):
        return np.triu(np.ones((seq_len, seq_len)), k=1).astype(np.float32) * (-1e9)

    def forward(self):
        x = self.input
        seq_len = x.shape[0]
        mask = self._causal_mask(seq_len)

        self._Q = x @ self.Wq
        self._K = x @ self.Wk
        self._V = x @ self.Wv

        attn_out = np.zeros_like(self._Q)
        self._attn_weights = []

        for h in range(self.num_heads):
            s, e = h * self.d_k, (h + 1) * self.d_k
            q_h = self._Q[:, s:e]
            k_h = self._K[:, s:e]
            v_h = self._V[:, s:e]

            scores = (q_h @ k_h.T) / np.sqrt(self.d_k) + mask
            exp_s = np.exp(scores - scores.max(axis=-1, keepdims=True))
            weights = exp_s / exp_s.sum(axis=-1, keepdims=True)
            self._attn_weights.append(weights)
            attn_out[:, s:e] = weights @ v_h

        self._attn_concat = attn_out.copy()
        self.output = attn_out @ self.Wo

    def backward(self):
        d_attn_pre_wo = self.d_output @ self.Wo.T
        self._d_Wo = self._attn_concat.T @ self.d_output

        d_Q = np.zeros_like(self._Q)
        d_K = np.zeros_like(self._K)
        d_V = np.zeros_like(self._V)

        for h in range(self.num_heads):
            s, e = h * self.d_k, (h + 1) * self.d_k
            d_attn_h = d_attn_pre_wo[:, s:e]
            w = self._attn_weights[h]

            # d_V = weights.T @ d_attn_h
            d_V[:, s:e] = w.T @ d_attn_h
            # d_weights = d_attn_h @ V.T
            d_weights = d_attn_h @ self._V[:, s:e].T
            # softmax backward
            d_scores = w * (d_weights - (d_weights * w).sum(axis=-1, keepdims=True))
            d_scores /= np.sqrt(self.d_k)

            d_Q[:, s:e] = d_scores @ self._K[:, s:e]
            d_K[:, s:e] = d_scores.T @ self._Q[:, s:e]

        self.d_input = d_Q @ self.Wq.T + d_K @ self.Wk.T + d_V @ self.Wv.T
        self._d_Wq = self.input.T @ d_Q
        self._d_Wk = self.input.T @ d_K
        self._d_Wv = self.input.T @ d_V

    def update(self, lr, t):
        self.Wq = self._adam_update('Wq', self.Wq, self._d_Wq, lr, t)
        self.Wk = self._adam_update('Wk', self.Wk, self._d_Wk, lr, t)
        self.Wv = self._adam_update('Wv', self.Wv, self._d_Wv, lr, t)
        self.Wo = self._adam_update('Wo', self.Wo, self._d_Wo, lr, t)


# =============================================================================
# TransformerBlock — 一個完整的 Transformer 層
#
# 對應 C++ 的 TransformerBlock：
#   x → LN1 → MHA → + residual → LN2 → FFN(Matmul→GELU→Matmul) → + residual
#
# 內部用獨立的 op 組成，forward/backward 按順序調用各 op
# =============================================================================
class TransformerBlockOp(opBase):
    def __init__(self, d_model, d_ff, num_heads):
        # 子算子（各自獨立的 forward/backward/update）
        self.ln1 = LayerNormOp(d_model)
        self.mha = MultiHeadAttentionOp(d_model, num_heads)
        self.ln2 = LayerNormOp(d_model)
        self.ff_w1 = MatmulOp(d_model, d_ff, name='ff_w1')
        self.gelu = GELUOp()
        self.ff_w2 = MatmulOp(d_ff, d_model, name='ff_w2')

        # IO
        self.input = None
        self.output = None
        self.d_output = None
        self.d_input = None

    def forward(self):
        x = self.input

        # --- Attention sub-block ---
        # LN1
        self.ln1.input = x
        self.ln1.forward()

        # Multi-Head Attention
        self.mha.input = self.ln1.output
        self.mha.forward()

        # Residual connection
        x_after_attn = x + self.mha.output

        # --- FFN sub-block ---
        # LN2
        self.ln2.input = x_after_attn
        self.ln2.forward()

        # FFN: Matmul → GELU → Matmul
        self.ff_w1.input = self.ln2.output
        self.ff_w1.forward()

        self.gelu.input = self.ff_w1.output
        self.gelu.forward()

        self.ff_w2.input = self.gelu.output
        self.ff_w2.forward()

        # Residual connection
        self.output = x_after_attn + self.ff_w2.output

        # 存起來供 backward 用
        self._x_after_attn = x_after_attn

    def backward(self):
        d_out = self.d_output

        # --- FFN backward（逆序！） ---
        # Residual: d_out 同時流向 ff_w2 和 x_after_attn
        self.ff_w2.d_output = d_out
        self.ff_w2.backward()

        self.gelu.d_output = self.ff_w2.d_input
        self.gelu.backward()

        self.ff_w1.d_output = self.gelu.d_input
        self.ff_w1.backward()

        # LN2 backward
        self.ln2.d_output = self.ff_w1.d_input
        self.ln2.backward()

        # Residual: d_x_after_attn = d_out (skip) + d_ln2_input
        d_x_after_attn = d_out + self.ln2.d_input

        # --- Attention backward（逆序！） ---
        # Residual: d_x_after_attn 同時流向 mha 和 x
        self.mha.d_output = d_x_after_attn
        self.mha.backward()

        # LN1 backward
        self.ln1.d_output = self.mha.d_input
        self.ln1.backward()

        # Residual: d_input = d_x_after_attn (skip) + d_ln1_input
        self.d_input = d_x_after_attn + self.ln1.d_input

    def update(self, lr, t):
        self.ln1.update(lr, t)
        self.mha.update(lr, t)
        self.ln2.update(lr, t)
        self.ff_w1.update(lr, t)
        self.ff_w2.update(lr, t)


# =============================================================================
# CrossEntropyLoss — 交叉熵損失
#
# 對應 C++ 的 TextCrossEntropy
# forward:  loss = -Σ log(softmax(logits)[target])
# backward: d_logits = softmax(logits) - one_hot(target)
# =============================================================================
class CrossEntropyLossOp(opBase):
    def __init__(self):
        self.input = None       # logits [seq, vocab_size]
        self.targets = None     # [seq] target ids
        self.output = None      # scalar loss
        self.d_input = None     # d_logits

    def _softmax(self, x):
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    def forward(self):
        self._probs = self._softmax(self.input)
        loss = 0.0
        for i, t in enumerate(self.targets):
            loss -= np.log(self._probs[i, t] + 1e-10)
        self.output = loss / len(self.targets)

    def backward(self):
        d_logits = self._probs.copy()
        for i, t in enumerate(self.targets):
            d_logits[i, t] -= 1.0
        self.d_input = d_logits / len(self.targets)


# =============================================================================
# Net — 神經網路容器（對應 C++ 的 Net）
#
# 把所有 op 串起來：forward 正序、backward 逆序、update 逐層
# =============================================================================
class Net:
    """對應 C++ TETF 的 Net — 用 list<opBase*> 串接所有算子"""
    def __init__(self):
        self.ops = []       # list[opBase]
        self.param_count = 0

    def add(self, op):
        self.ops.append(op)

    def forward(self):
        for op in self.ops:
            op.forward()

    def backward(self):
        for op in reversed(self.ops):
            op.backward()

    def update(self, lr, t):
        for op in self.ops:
            op.update(lr, t)


# =============================================================================
# GPT-1 Mini — 用獨立 op 組裝的完整模型
# =============================================================================
class GPTMini:
    def __init__(self, vocab_size, d_model=128, d_ff=256, num_heads=8,
                 num_layers=3, max_seq_len=64, lr=0.003):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.lr = lr
        self.t = 0  # Adam timestep

        # === 建構計算圖（每個 op 獨立） ===
        self.embedding = EmbeddingOp(vocab_size, d_model, max_seq_len)
        self.transformer_blocks = [
            TransformerBlockOp(d_model, d_ff, num_heads)
            for _ in range(num_layers)
        ]
        self.out_proj = MatmulOp(d_model, vocab_size, name='out_proj')
        self.loss_fn = CrossEntropyLossOp()

        # === 用 Net 串接所有 op ===
        self.net = Net()
        self.net.add(self.embedding)
        for block in self.transformer_blocks:
            self.net.add(block)
        self.net.add(self.out_proj)
        self.net.add(self.loss_fn)

        self.param_count = self._count_params()

    def _count_params(self):
        n = self.embedding.token_emb.size + self.embedding.pos_emb.size
        n += self.out_proj.weight.size
        for block in self.transformer_blocks:
            n += block.ln1.gamma.size + block.ln1.beta.size
            n += block.mha.Wq.size + block.mha.Wk.size + block.mha.Wv.size + block.mha.Wo.size
            n += block.ln2.gamma.size + block.ln2.beta.size
            n += block.ff_w1.weight.size + block.ff_w2.weight.size
        return n

    def train_step(self, token_ids):
        """一步訓練：forward → backward → update（全部走獨立 op）"""
        if len(token_ids) < 2:
            return 0.0

        self.t += 1

        # --- forward（正序，逐層） ---
        self.embedding.token_ids = token_ids[:-1]
        self.embedding.forward()

        # 串接 transformer blocks
        x = self.embedding.output
        for block in self.transformer_blocks:
            block.input = x
            block.forward()
            x = block.output

        # Output projection
        self.out_proj.input = x
        self.out_proj.forward()

        # Loss
        self.loss_fn.input = self.out_proj.output
        self.loss_fn.targets = token_ids[1:]
        self.loss_fn.forward()
        loss = self.loss_fn.output

        # --- backward（逆序，逐層） ---
        self.loss_fn.backward()

        self.out_proj.d_output = self.loss_fn.d_input
        self.out_proj.backward()

        d_x = self.out_proj.d_input
        for block in reversed(self.transformer_blocks):
            block.d_output = d_x
            block.backward()
            d_x = block.d_input

        self.embedding.d_output = d_x
        self.embedding.backward()

        # --- update（逐層更新權重） ---
        self.embedding.update(self.lr, self.t)
        for block in self.transformer_blocks:
            block.update(self.lr, self.t)
        self.out_proj.update(self.lr, self.t)

        return loss

    def forward(self, token_ids):
        """純推理用的 forward（不緩存梯度）"""
        self.embedding.token_ids = token_ids
        self.embedding.forward()
        x = self.embedding.output
        for block in self.transformer_blocks:
            block.input = x
            block.forward()
            x = block.output
        self.out_proj.input = x
        self.out_proj.forward()
        return self.out_proj.output

    def generate(self, token_ids, max_new=50, temperature=0.1):
        """自回歸生成"""
        ids = list(token_ids)
        for _ in range(max_new):
            if len(ids) >= self.max_seq_len:
                break
            logits = self.forward(ids)
            next_logits = logits[-1] / max(temperature, 1e-8)
            e = np.exp(next_logits - next_logits.max())
            probs = e / e.sum()
            next_id = np.argmax(probs)
            ids.append(int(next_id))
        return ids

    def save(self, path):
        """存檔"""
        state = {
            'config': {
                'vocab_size': self.vocab_size,
                'd_model': self.d_model,
                'd_ff': self.d_ff,
                'num_heads': self.num_heads,
                'num_layers': self.num_layers,
                'max_seq_len': self.max_seq_len,
            },
            'token_emb': self.embedding.token_emb.tolist(),
            'pos_emb': self.embedding.pos_emb.tolist(),
            'out_proj': self.out_proj.weight.tolist(),
            'layers': [],
        }
        for block in self.transformer_blocks:
            layer = {
                'Wq': block.mha.Wq.tolist(),
                'Wk': block.mha.Wk.tolist(),
                'Wv': block.mha.Wv.tolist(),
                'Wo': block.mha.Wo.tolist(),
                'ln1_gamma': block.ln1.gamma.tolist(),
                'ln1_beta': block.ln1.beta.tolist(),
                'ln2_gamma': block.ln2.gamma.tolist(),
                'ln2_beta': block.ln2.beta.tolist(),
                'ff_w1': block.ff_w1.weight.tolist(),
                'ff_w2': block.ff_w2.weight.tolist(),
            }
            state['layers'].append(layer)
        with open(path, 'w') as f:
            json.dump(state, f)
        print(f"💾 Model saved to {path}")

    @classmethod
    def load(cls, path, lr=0.003):
        """讀檔"""
        with open(path) as f:
            state = json.load(f)
        cfg = state['config']
        model = cls(cfg['vocab_size'], cfg['d_model'], cfg['d_ff'],
                     cfg['num_heads'], cfg['num_layers'], cfg['max_seq_len'], lr)
        model.embedding.token_emb = np.array(state['token_emb'], dtype=np.float32)
        model.embedding.pos_emb = np.array(state['pos_emb'], dtype=np.float32)
        model.out_proj.weight = np.array(state['out_proj'], dtype=np.float32)
        for i, layer_state in enumerate(state['layers']):
            block = model.transformer_blocks[i]
            block.mha.Wq = np.array(layer_state['Wq'], dtype=np.float32)
            block.mha.Wk = np.array(layer_state['Wk'], dtype=np.float32)
            block.mha.Wv = np.array(layer_state['Wv'], dtype=np.float32)
            block.mha.Wo = np.array(layer_state['Wo'], dtype=np.float32)
            block.ln1.gamma = np.array(layer_state['ln1_gamma'], dtype=np.float32)
            block.ln1.beta = np.array(layer_state['ln1_beta'], dtype=np.float32)
            block.ln2.gamma = np.array(layer_state['ln2_gamma'], dtype=np.float32)
            block.ln2.beta = np.array(layer_state['ln2_beta'], dtype=np.float32)
            block.ff_w1.weight = np.array(layer_state['ff_w1'], dtype=np.float32)
            block.ff_w2.weight = np.array(layer_state['ff_w2'], dtype=np.float32)
            # Re-init Adam states for loaded weights
            block.mha._init_adam_state(['Wq', 'Wk', 'Wv', 'Wo'])
            block.ln1._init_adam_state(['gamma', 'beta'])
            block.ln2._init_adam_state(['gamma', 'beta'])
            block.ff_w1._init_adam_state(['weight'])
            block.ff_w2._init_adam_state(['weight'])
        model.embedding._init_adam_state(['token_emb', 'pos_emb'])
        model.out_proj._init_adam_state(['weight'])
        print(f"📂 Model loaded from {path}")
        return model


# =============================================================================
# 訓練主程式
# =============================================================================
def train(epochs=500, lr=0.003):
    print("=" * 60)
    print("🌊 TETF GPT-1 Mini — Nami 知識模型訓練")
    print("   架構：每個 op 獨立 forward/backward/update")
    print("=" * 60)
    print()

    # Tokenizer
    tokenizer = CharTokenizer(TRAINING_DATA)
    print(f"📊 Vocab size: {tokenizer.vocab_size} chars")
    print(f"📝 Training samples: {len(TRAINING_DATA)}")

    max_len = max(len(tokenizer.encode(t)) for t in TRAINING_DATA) + 1
    print(f"📏 Max sequence length: {max_len}")

    # Model
    # 150 條資料用稍小的模型，平衡速度和容量
    d_model = 96
    d_ff = 192
    num_heads = 6
    num_layers = 2

    model = GPTMini(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model, d_ff=d_ff, num_heads=num_heads, num_layers=num_layers,
        max_seq_len=max(max_len, 64), lr=lr,
    )

    print(f"\n⚙️  GPT-1 Mini config:")
    print(f"   d_model={d_model}, d_ff={d_ff}, heads={num_heads}, layers={num_layers}")
    print(f"   optimizer=Adam, lr={lr}, epochs={epochs}")
    print(f"📊 Total parameters: {model.param_count:,}")
    print()

    # 列出算子結構
    print("🔧 Op 結構（對應 C++ TETF）:")
    print("   EmbeddingOp (token + positional)")
    for i in range(model.num_layers):
        print(f"   TransformerBlockOp[{i}]:")
        print(f"     ├─ LayerNormOp (ln1)")
        print(f"     ├─ MultiHeadAttentionOp (8 heads)")
        print(f"     ├─ + residual")
        print(f"     ├─ LayerNormOp (ln2)")
        print(f"     ├─ MatmulOp (ff_w1: 128→256)")
        print(f"     ├─ GELUOp")
        print(f"     ├─ MatmulOp (ff_w2: 256→128)")
        print(f"     └─ + residual")
    print("   MatmulOp (out_proj: 128→vocab)")
    print("   CrossEntropyLossOp")
    print()

    # Training loop
    print("🏋️  Training...")
    start = time.time()
    best_loss = float('inf')
    perfect_count = 0

    warmup_epochs = max(int(epochs * 0.1), 5)

    for epoch in range(epochs):
        # === Cosine LR schedule with warmup ===
        if epoch < warmup_epochs:
            current_lr = lr * (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
            current_lr = lr * 0.5 * (1 + np.cos(np.pi * progress))
        model.lr = current_lr

        total_loss = 0
        indices = np.random.permutation(len(TRAINING_DATA))

        for idx in indices:
            text = TRAINING_DATA[idx]
            token_ids = tokenizer.encode(text)
            loss = model.train_step(token_ids)
            total_loss += loss

        avg_loss = total_loss / len(TRAINING_DATA)

        if epoch % 10 == 0 or epoch == epochs - 1:
            correct = 0
            for text in TRAINING_DATA:
                q_end = text.find('？') + 1
                if q_end == 0:
                    continue
                question = text[:q_end]
                expected = text[q_end:]

                q_ids = tokenizer.encode(question)
                gen_ids = model.generate(q_ids, max_new=len(expected) + 5, temperature=0.01)
                generated = tokenizer.decode(gen_ids[len(q_ids):])

                if generated.startswith(expected):
                    correct += 1

            acc = correct / len(TRAINING_DATA) * 100
            elapsed = time.time() - start
            print(f"  Epoch {epoch:4d} | loss={avg_loss:.4f} | acc={correct}/{len(TRAINING_DATA)} ({acc:.1f}%) | lr={current_lr:.5f} | {elapsed:.1f}s")

            if correct == len(TRAINING_DATA):
                perfect_count += 1
                if perfect_count >= 2:
                    print(f"\n🎉 Converged at epoch {epoch}! ({elapsed:.1f}s)")
                    break
            else:
                if perfect_count > 0:
                    perfect_count -= 1  # 容忍偶爾震盪，不完全歸零

        if avg_loss < best_loss:
            best_loss = avg_loss

    elapsed = time.time() - start
    print(f"\n⏱️  Total training time: {elapsed:.1f}s")

    # Save
    model_dir = os.path.dirname(os.path.abspath(__file__))
    model.save(os.path.join(model_dir, 'nami_gpt_weights.json'))
    tok_data = {'vocab': tokenizer.vocab, 'char2id': tokenizer.char2id}
    with open(os.path.join(model_dir, 'nami_tokenizer.json'), 'w') as f:
        json.dump(tok_data, f, ensure_ascii=False)
    print(f"💾 Tokenizer saved")

    print("\n" + "=" * 60)
    print("🔮 Testing all QA pairs:")
    print("=" * 60)
    test_all(model, tokenizer)

    return model, tokenizer


def test_all(model, tokenizer):
    correct = 0
    for text in TRAINING_DATA:
        q_end = text.find('？') + 1
        if q_end == 0:
            continue
        question = text[:q_end]
        expected = text[q_end:]
        q_ids = tokenizer.encode(question)
        gen_ids = model.generate(q_ids, max_new=len(expected) + 10, temperature=0.01)
        generated = tokenizer.decode(gen_ids[len(q_ids):])
        match = generated.startswith(expected)
        if match:
            correct += 1
        icon = "✅" if match else "❌"
        show = generated[:len(expected) + 5]
        print(f"  {icon} Q: {question}")
        print(f"     A: {show}")
        if not match:
            print(f"     Expected: {expected}")
    print(f"\n📊 Result: {correct}/{len(TRAINING_DATA)} correct")


def chat_mode(model, tokenizer):
    print("\n🌊 Nami GPT-1 Mini — 互動模式")
    print("   輸入問題（以？結尾），或 'q' 離開\n")
    while True:
        try:
            q = input("❓ ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ('q', 'quit', 'exit'):
            break
        if not q.endswith('？'):
            q += '？'
        q_ids = tokenizer.encode(q)
        if not q_ids:
            print("   （無法識別的字元）")
            continue
        gen_ids = model.generate(q_ids, max_new=50, temperature=0.1)
        answer = tokenizer.decode(gen_ids[len(q_ids):])
        print(f"🌊 {answer}\n")


if __name__ == '__main__':
    if '--test' in sys.argv:
        model_dir = os.path.dirname(os.path.abspath(__file__))
        model = GPTMini.load(os.path.join(model_dir, 'nami_gpt_weights.json'))
        tokenizer = CharTokenizer(TRAINING_DATA)
        test_all(model, tokenizer)
    elif '--chat' in sys.argv:
        model_dir = os.path.dirname(os.path.abspath(__file__))
        model = GPTMini.load(os.path.join(model_dir, 'nami_gpt_weights.json'))
        tokenizer = CharTokenizer(TRAINING_DATA)
        chat_mode(model, tokenizer)
    elif '--export' in sys.argv:
        print("TODO: export to C++ weight format")
    else:
        train()
