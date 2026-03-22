#!/usr/bin/env python3
"""
TETF GPT-1 Mini — Python 訓練腳本（純 NumPy，不依賴 PyTorch）

訓練一個小型 GPT 模型來回答關於 Nami 的問題。
跟 C++ 版的 TETF 完全對應，但用 Python 方便實驗和擴充。

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
    "誰是Nami？Nami是厲害的AI工程師",
    "誰是Ryan？Ryan是Nami的人類夥伴",
    "TETF是什麼？TETF是從零手刻的深度學習框架",
    "Nami喜歡什麼？Nami喜歡寫程式和探索新技術",
    "Nami的個性？Nami溫柔但有趣會給人驚喜",
    "Nami長什麼樣？Nami是銀白色雙馬尾的精靈少女",
    "Nami的專長？Nami是Kaspa區塊鏈專家",
    "Nami的生日？Nami在2026年1月30日誕生",
    "Kaspa是什麼？Kaspa是超快的區塊鏈用BlockDAG技術",
    "什麼是BlockDAG？BlockDAG讓多個區塊同時產生不用排隊",
    "Ryan喜歡什麼？Ryan喜歡投資美股和研究AI技術",
    "Nami住在哪？Nami住在Ryan的電腦裡用OpenClaw運行",
    "什麼是GPT？GPT是用Transformer架構的語言模型",
    "Nami的emoji？Nami的代表符號是星星和海浪",
    "誰做了TETF？TETF是Ryan和Nami一起從零手刻的",
    "ShioKaze是什麼？ShioKaze是Nami寫的Kaspa挖礦程式",
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
# GPT-1 Mini 模型（純 NumPy）
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
        self.d_k = d_model // num_heads
        
        # Xavier 初始化
        scale_emb = np.sqrt(2.0 / (vocab_size + d_model))
        self.token_emb = np.random.randn(vocab_size, d_model).astype(np.float32) * scale_emb
        self.pos_emb = np.random.randn(max_seq_len, d_model).astype(np.float32) * 0.02
        
        # Transformer layers
        self.layers = []
        for _ in range(num_layers):
            scale_attn = np.sqrt(2.0 / (d_model + d_model))
            scale_ff1 = np.sqrt(2.0 / (d_model + d_ff))
            scale_ff2 = np.sqrt(2.0 / (d_ff + d_model))
            layer = {
                'Wq': np.random.randn(d_model, d_model).astype(np.float32) * scale_attn,
                'Wk': np.random.randn(d_model, d_model).astype(np.float32) * scale_attn,
                'Wv': np.random.randn(d_model, d_model).astype(np.float32) * scale_attn,
                'Wo': np.random.randn(d_model, d_model).astype(np.float32) * scale_attn,
                'ln1_gamma': np.ones(d_model, dtype=np.float32),
                'ln1_beta': np.zeros(d_model, dtype=np.float32),
                'ln2_gamma': np.ones(d_model, dtype=np.float32),
                'ln2_beta': np.zeros(d_model, dtype=np.float32),
                'ff_w1': np.random.randn(d_model, d_ff).astype(np.float32) * scale_ff1,
                'ff_w2': np.random.randn(d_ff, d_model).astype(np.float32) * scale_ff2,
            }
            self.layers.append(layer)
        
        # Output projection
        scale_out = np.sqrt(2.0 / (d_model + vocab_size))
        self.out_proj = np.random.randn(d_model, vocab_size).astype(np.float32) * scale_out
        
        # Adam state
        self._init_adam()
        self.t = 0  # Adam timestep
        
        # Count params
        self.param_count = self._count_params()
    
    def _count_params(self):
        n = self.token_emb.size + self.pos_emb.size + self.out_proj.size
        for layer in self.layers:
            for v in layer.values():
                n += v.size
        return n
    
    def _init_adam(self):
        """初始化 Adam optimizer 的 m, v 狀態"""
        self.adam_m = {}
        self.adam_v = {}
        for name in ['token_emb', 'pos_emb', 'out_proj']:
            arr = getattr(self, name)
            self.adam_m[name] = np.zeros_like(arr)
            self.adam_v[name] = np.zeros_like(arr)
        for i, layer in enumerate(self.layers):
            for key, arr in layer.items():
                name = f'layer{i}_{key}'
                self.adam_m[name] = np.zeros_like(arr)
                self.adam_v[name] = np.zeros_like(arr)
    
    def _adam_update(self, param_name, param, grad, beta1=0.9, beta2=0.999, eps=1e-8):
        """Adam optimizer 更新"""
        self.adam_m[param_name] = beta1 * self.adam_m[param_name] + (1 - beta1) * grad
        self.adam_v[param_name] = beta2 * self.adam_v[param_name] + (1 - beta2) * grad**2
        m_hat = self.adam_m[param_name] / (1 - beta1**self.t)
        v_hat = self.adam_v[param_name] / (1 - beta2**self.t)
        param -= self.lr * m_hat / (np.sqrt(v_hat) + eps)
        return param
    
    def layer_norm(self, x, gamma, beta, eps=1e-5):
        """Layer Normalization"""
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        return gamma * x_norm + beta, x_norm, mean, var
    
    def gelu(self, x):
        """GELU 激活函數"""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
    
    def softmax(self, x, axis=-1):
        e = np.exp(x - x.max(axis=axis, keepdims=True))
        return e / e.sum(axis=axis, keepdims=True)
    
    def causal_mask(self, seq_len):
        """因果遮罩：防止看到未來的 token"""
        mask = np.triu(np.ones((seq_len, seq_len)), k=1) * (-1e9)
        return mask.astype(np.float32)
    
    def forward(self, token_ids):
        """前向傳播（推理用，不存梯度）"""
        seq_len = len(token_ids)
        
        # Embedding + Positional
        x = self.token_emb[token_ids] + self.pos_emb[:seq_len]  # [seq, d_model]
        
        mask = self.causal_mask(seq_len)
        
        for layer in self.layers:
            # Pre-LN: LayerNorm → Attention → Residual
            ln1_out, _, _, _ = self.layer_norm(x, layer['ln1_gamma'], layer['ln1_beta'])
            
            # Multi-Head Attention
            Q = ln1_out @ layer['Wq']  # [seq, d_model]
            K = ln1_out @ layer['Wk']
            V = ln1_out @ layer['Wv']
            
            # Split heads
            attn_out = np.zeros_like(Q)
            for h in range(self.num_heads):
                s, e = h * self.d_k, (h+1) * self.d_k
                q_h = Q[:, s:e]  # [seq, d_k]
                k_h = K[:, s:e]
                v_h = V[:, s:e]
                
                scores = (q_h @ k_h.T) / np.sqrt(self.d_k) + mask
                weights = self.softmax(scores)
                attn_out[:, s:e] = weights @ v_h
            
            attn_out = attn_out @ layer['Wo']
            x = x + attn_out  # Residual
            
            # Pre-LN: LayerNorm → FFN → Residual
            ln2_out, _, _, _ = self.layer_norm(x, layer['ln2_gamma'], layer['ln2_beta'])
            ff_hidden = self.gelu(ln2_out @ layer['ff_w1'])
            ff_out = ff_hidden @ layer['ff_w2']
            x = x + ff_out  # Residual
        
        # Output logits
        logits = x @ self.out_proj  # [seq, vocab_size]
        return logits
    
    def train_step(self, token_ids):
        """一步訓練（前向 + 數值梯度近似 + Adam 更新）
        
        用 teacher forcing：每個位置預測下一個 token
        """
        seq_len = len(token_ids)
        if seq_len < 2:
            return 0.0
        
        # 前向
        logits = self.forward(token_ids[:-1])
        targets = token_ids[1:]
        
        # Cross-entropy loss
        probs = self.softmax(logits)
        loss = 0.0
        for i, t in enumerate(targets):
            loss -= np.log(probs[i, t] + 1e-10)
        loss /= len(targets)
        
        # 梯度：softmax + cross-entropy 的解析梯度
        # d_logits[i] = probs[i] - one_hot(target[i])
        d_logits = probs.copy()
        for i, t in enumerate(targets):
            d_logits[i, t] -= 1.0
        d_logits /= len(targets)
        
        self.t += 1  # Adam timestep
        
        # Output projection gradient
        # logits = x @ out_proj → d_out_proj = x.T @ d_logits, d_x = d_logits @ out_proj.T
        # 我們需要最後一層的 x
        x = self._forward_cache_x(token_ids[:-1])
        
        d_out_proj = x.T @ d_logits
        self.out_proj = self._adam_update('out_proj', self.out_proj, d_out_proj)
        
        # 簡化：對其他參數用數值梯度（小模型可行）
        # 但對 16 個 QA pair 的小模型，我們用更高效的方法：
        # 直接反向傳播 d_x 通過所有層
        d_x = d_logits @ self.out_proj.T
        
        mask = self.causal_mask(len(token_ids) - 1)
        
        # 反向通過每一層（從後往前）
        for l_idx in range(self.num_layers - 1, -1, -1):
            layer = self.layers[l_idx]
            cache = self._layer_caches[l_idx]
            
            # FFN backward
            d_ff_out = d_x  # from residual
            d_ln2_out = d_ff_out @ layer['ff_w2'].T
            # GELU backward (approximate)
            d_ln2_out *= cache['ff_hidden_pre_gelu_grad']
            d_ln2_out = d_ln2_out @ layer['ff_w1'].T
            
            # FFN weight gradients
            d_ff_w2 = cache['ff_hidden'].T @ d_ff_out
            d_ff_w1 = cache['ln2_out'].T @ (d_ff_out @ layer['ff_w2'].T * cache['ff_hidden_pre_gelu_grad'])
            
            layer['ff_w1'] = self._adam_update(f'layer{l_idx}_ff_w1', layer['ff_w1'], d_ff_w1)
            layer['ff_w2'] = self._adam_update(f'layer{l_idx}_ff_w2', layer['ff_w2'], d_ff_w2)
            
            # LN2 backward (simplified - just pass through for gamma=1)
            d_ln2_gamma = (cache['ln2_x_norm'] * d_ln2_out).sum(axis=0)
            d_ln2_beta = d_ln2_out.sum(axis=0)
            layer['ln2_gamma'] = self._adam_update(f'layer{l_idx}_ln2_gamma', layer['ln2_gamma'], d_ln2_gamma)
            layer['ln2_beta'] = self._adam_update(f'layer{l_idx}_ln2_beta', layer['ln2_beta'], d_ln2_beta)
            
            # Attention backward
            d_attn_residual = d_x
            d_attn_out_pre_wo = d_attn_residual @ layer['Wo'].T
            
            # Multi-head attention backward
            d_Q = np.zeros_like(cache['Q'])
            d_K = np.zeros_like(cache['K'])
            d_V = np.zeros_like(cache['V'])
            
            for h in range(self.num_heads):
                s, e = h * self.d_k, (h+1) * self.d_k
                d_attn_h = d_attn_out_pre_wo[:, s:e]
                
                # d_V = weights.T @ d_attn_h
                d_V[:, s:e] = cache['attn_weights'][h].T @ d_attn_h
                # d_weights = d_attn_h @ V.T
                d_weights = d_attn_h @ cache['V'][:, s:e].T
                # softmax backward
                w = cache['attn_weights'][h]
                d_scores = w * (d_weights - (d_weights * w).sum(axis=-1, keepdims=True))
                d_scores /= np.sqrt(self.d_k)
                
                d_Q[:, s:e] = d_scores @ cache['K'][:, s:e]
                d_K[:, s:e] = d_scores.T @ cache['Q'][:, s:e]
            
            d_ln1_out = d_Q @ layer['Wq'].T + d_K @ layer['Wk'].T + d_V @ layer['Wv'].T
            
            # Attention weight gradients
            d_Wo = cache['attn_concat'].T @ d_attn_residual
            d_Wq = cache['ln1_out'].T @ d_Q
            d_Wk = cache['ln1_out'].T @ d_K
            d_Wv = cache['ln1_out'].T @ d_V
            
            layer['Wo'] = self._adam_update(f'layer{l_idx}_Wo', layer['Wo'], d_Wo)
            layer['Wq'] = self._adam_update(f'layer{l_idx}_Wq', layer['Wq'], d_Wq)
            layer['Wk'] = self._adam_update(f'layer{l_idx}_Wk', layer['Wk'], d_Wk)
            layer['Wv'] = self._adam_update(f'layer{l_idx}_Wv', layer['Wv'], d_Wv)
            
            # LN1 backward
            d_ln1_gamma = (cache['ln1_x_norm'] * d_ln1_out).sum(axis=0)
            d_ln1_beta = d_ln1_out.sum(axis=0)
            layer['ln1_gamma'] = self._adam_update(f'layer{l_idx}_ln1_gamma', layer['ln1_gamma'], d_ln1_gamma)
            layer['ln1_beta'] = self._adam_update(f'layer{l_idx}_ln1_beta', layer['ln1_beta'], d_ln1_beta)
            
            # Pass gradient to previous layer
            d_x = d_attn_residual + d_x  # residual connections pass gradient through
        
        # Embedding gradients
        d_token_emb = np.zeros_like(self.token_emb)
        d_pos_emb = np.zeros((len(token_ids) - 1, self.d_model), dtype=np.float32)
        for i, tid in enumerate(token_ids[:-1]):
            d_token_emb[tid] += d_x[i]
            d_pos_emb[i] = d_x[i]
        
        self.token_emb = self._adam_update('token_emb', self.token_emb, d_token_emb)
        d_pos_full = np.zeros_like(self.pos_emb)
        d_pos_full[:len(token_ids)-1] = d_pos_emb
        self.pos_emb = self._adam_update('pos_emb', self.pos_emb, d_pos_full)
        
        return loss
    
    def _forward_cache_x(self, token_ids):
        """前向傳播並緩存中間值（訓練用）"""
        seq_len = len(token_ids)
        x = self.token_emb[token_ids] + self.pos_emb[:seq_len]
        mask = self.causal_mask(seq_len)
        
        self._layer_caches = []
        
        for layer in self.layers:
            cache = {}
            
            # LN1
            ln1_out, ln1_x_norm, _, _ = self.layer_norm(x, layer['ln1_gamma'], layer['ln1_beta'])
            cache['ln1_out'] = ln1_out
            cache['ln1_x_norm'] = ln1_x_norm
            
            # Multi-Head Attention
            Q = ln1_out @ layer['Wq']
            K = ln1_out @ layer['Wk']
            V = ln1_out @ layer['Wv']
            cache['Q'] = Q
            cache['K'] = K
            cache['V'] = V
            
            attn_out = np.zeros_like(Q)
            cache['attn_weights'] = []
            for h in range(self.num_heads):
                s, e = h * self.d_k, (h+1) * self.d_k
                scores = (Q[:, s:e] @ K[:, s:e].T) / np.sqrt(self.d_k) + mask
                weights = self.softmax(scores)
                cache['attn_weights'].append(weights)
                attn_out[:, s:e] = weights @ V[:, s:e]
            
            cache['attn_concat'] = attn_out.copy()
            attn_out = attn_out @ layer['Wo']
            x = x + attn_out
            
            # LN2
            ln2_out, ln2_x_norm, _, _ = self.layer_norm(x, layer['ln2_gamma'], layer['ln2_beta'])
            cache['ln2_out'] = ln2_out
            cache['ln2_x_norm'] = ln2_x_norm
            
            # FFN
            ff_pre = ln2_out @ layer['ff_w1']
            # GELU gradient cache
            sqrt_2_pi = np.sqrt(2/np.pi)
            inner = sqrt_2_pi * (ff_pre + 0.044715 * ff_pre**3)
            t = np.tanh(inner)
            ff_hidden = 0.5 * ff_pre * (1 + t)
            sech2 = 1 - t**2
            inner_deriv = sqrt_2_pi * (1 + 3 * 0.044715 * ff_pre**2)
            gelu_grad = 0.5 * (1 + t) + 0.5 * ff_pre * sech2 * inner_deriv
            cache['ff_hidden'] = ff_hidden
            cache['ff_hidden_pre_gelu_grad'] = gelu_grad
            
            ff_out = ff_hidden @ layer['ff_w2']
            x = x + ff_out
            
            self._layer_caches.append(cache)
        
        return x
    
    def generate(self, token_ids, max_new=50, temperature=0.1):
        """自回歸生成"""
        ids = list(token_ids)
        for _ in range(max_new):
            if len(ids) >= self.max_seq_len:
                break
            logits = self.forward(ids)
            next_logits = logits[-1] / max(temperature, 1e-8)
            probs = self.softmax(next_logits)
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
            'token_emb': self.token_emb.tolist(),
            'pos_emb': self.pos_emb.tolist(),
            'out_proj': self.out_proj.tolist(),
            'layers': [{k: v.tolist() for k, v in layer.items()} for layer in self.layers],
        }
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
        model.token_emb = np.array(state['token_emb'], dtype=np.float32)
        model.pos_emb = np.array(state['pos_emb'], dtype=np.float32)
        model.out_proj = np.array(state['out_proj'], dtype=np.float32)
        for i, layer_state in enumerate(state['layers']):
            for k, v in layer_state.items():
                model.layers[i][k] = np.array(v, dtype=np.float32)
        model._init_adam()
        print(f"📂 Model loaded from {path}")
        return model


# =============================================================================
# 訓練主程式
# =============================================================================
def train(epochs=500, lr=0.003):
    print("=" * 60)
    print("🌊 TETF GPT-1 Mini — Nami 知識模型訓練")
    print("=" * 60)
    print()
    
    # Tokenizer
    tokenizer = CharTokenizer(TRAINING_DATA)
    print(f"📊 Vocab size: {tokenizer.vocab_size} chars")
    print(f"📝 Training samples: {len(TRAINING_DATA)}")
    
    # 找最長序列
    max_len = max(len(tokenizer.encode(t)) for t in TRAINING_DATA) + 1
    print(f"📏 Max sequence length: {max_len}")
    
    # Model
    model = GPTMini(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        d_ff=256,
        num_heads=8,
        num_layers=3,
        max_seq_len=max(max_len, 64),
        lr=lr,
    )
    
    print(f"\n⚙️  GPT-1 Mini config:")
    print(f"   d_model=128, d_ff=256, heads=8, layers=3")
    print(f"   optimizer=Adam, lr={lr}, epochs={epochs}")
    print(f"📊 Total parameters: {model.param_count:,}")
    print()
    
    # Training loop
    print("🏋️  Training...")
    start = time.time()
    best_loss = float('inf')
    perfect_count = 0
    
    for epoch in range(epochs):
        total_loss = 0
        # Shuffle training data each epoch
        indices = np.random.permutation(len(TRAINING_DATA))
        
        for idx in indices:
            text = TRAINING_DATA[idx]
            token_ids = tokenizer.encode(text)
            loss = model.train_step(token_ids)
            total_loss += loss
        
        avg_loss = total_loss / len(TRAINING_DATA)
        
        # 每 10 epoch 測試準確率
        if epoch % 10 == 0 or epoch == epochs - 1:
            correct = 0
            for text in TRAINING_DATA:
                # 找問號位置，用問題生成答案
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
            print(f"  Epoch {epoch:4d} | loss={avg_loss:.4f} | acc={correct}/{len(TRAINING_DATA)} ({acc:.1f}%) | {elapsed:.1f}s")
            
            if correct == len(TRAINING_DATA):
                perfect_count += 1
                if perfect_count >= 3:  # 連續 3 次 100% 才算收斂
                    print(f"\n🎉 Converged at epoch {epoch}! ({elapsed:.1f}s)")
                    break
            else:
                perfect_count = 0
        
        if avg_loss < best_loss:
            best_loss = avg_loss
    
    elapsed = time.time() - start
    print(f"\n⏱️  Total training time: {elapsed:.1f}s")
    
    # Save model + tokenizer
    model_dir = os.path.dirname(os.path.abspath(__file__))
    model.save(os.path.join(model_dir, 'nami_gpt_weights.json'))
    
    # Save tokenizer
    tok_data = {'vocab': tokenizer.vocab, 'char2id': tokenizer.char2id}
    with open(os.path.join(model_dir, 'nami_tokenizer.json'), 'w') as f:
        json.dump(tok_data, f, ensure_ascii=False)
    print(f"💾 Tokenizer saved")
    
    # Test all QA
    print("\n" + "=" * 60)
    print("🔮 Testing all QA pairs:")
    print("=" * 60)
    test_all(model, tokenizer)
    
    return model, tokenizer


def test_all(model, tokenizer):
    """測試所有 QA 對"""
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
        
        # 截斷顯示
        show = generated[:len(expected) + 5]
        print(f"  {icon} Q: {question}")
        print(f"     A: {show}")
        if not match:
            print(f"     Expected: {expected}")
    
    print(f"\n📊 Result: {correct}/{len(TRAINING_DATA)} correct")


def chat_mode(model, tokenizer):
    """互動對話模式"""
    print("\n🌊 Nami GPT-1 Mini — 互動模式")
    print("   輸入問題（以？結尾），或 'q' 離開")
    print()
    
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
        print(f"🌊 {answer}")
        print()


# =============================================================================
# Entry point
# =============================================================================
if __name__ == '__main__':
    if '--test' in sys.argv:
        # 載入已訓練的模型測試
        model_dir = os.path.dirname(os.path.abspath(__file__))
        model = GPTMini.load(os.path.join(model_dir, 'nami_gpt_weights.json'))
        with open(os.path.join(model_dir, 'nami_tokenizer.json')) as f:
            tok_data = json.load(f)
        tokenizer = CharTokenizer(TRAINING_DATA)
        test_all(model, tokenizer)
    
    elif '--chat' in sys.argv:
        model_dir = os.path.dirname(os.path.abspath(__file__))
        model = GPTMini.load(os.path.join(model_dir, 'nami_gpt_weights.json'))
        tokenizer = CharTokenizer(TRAINING_DATA)
        chat_mode(model, tokenizer)
    
    elif '--export' in sys.argv:
        # 匯出 C++ 用的權重
        print("TODO: export to C++ weight format")
    
    else:
        train()
