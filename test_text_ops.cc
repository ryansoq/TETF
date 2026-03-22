/*
 * ============================================================================
 * TETF — Text Transformer Ops 測試
 * ============================================================================
 *
 * 對每個新增的 Text op 做正確性驗證：
 *   1. Forward 值正確性（手算對照）
 *   2. Backward 梯度正確性（數值梯度 vs 解析梯度）
 *
 * 數值梯度法（Finite Differences）：
 *   ∂L/∂x ≈ (L(x+ε) - L(x-ε)) / (2ε)
 *   跟 backward() 算出的解析梯度比較，相對誤差 < 5% 就 PASS
 *
 * 用法：
 *   g++ -std=c++11 -O0 -g test_text_ops.cc -DTYPE2_BACKWARD -DTYPE4_BACKWARD_CONV \
 *       -DIM2COLxGEMM -I third_party/mnist/include -I third_party/f2uc -o test_text_ops
 *   ./test_text_ops
 *
 * Authors: Ryan & Nami ✨
 * ============================================================================
 */

// 引入 main.cc（改名原本的 main 避免衝突）
#define main original_main
#include "main.cc"
#undef main

#include <cmath>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <functional>

// ============================================================================
// 測試工具
// ============================================================================

int tests_passed = 0;
int tests_failed = 0;

#define CHECK(name, cond) do { \
    if (cond) { \
        std::cout << "  ✅ " << name << std::endl; \
        tests_passed++; \
    } else { \
        std::cout << "  ❌ " << name << std::endl; \
        tests_failed++; \
    } \
} while(0)

// 把 tensor 所有 node 的 diff 和 diffs 歸零
void zero_diffs(tensor *t) {
    for (size_t i = 0; i < t->data.size(); i++) {
        t->data[i].diff = 0.0f;
        t->data[i].diffs.clear();
    }
}

// 數值梯度：對 node 的 val 做 ±ε 擾動
float numerical_grad(node &n, std::function<float()> compute_loss, float eps = 1e-4f) {
    float orig = n.val;
    n.val = orig + eps;
    float loss_plus = compute_loss();
    n.val = orig - eps;
    float loss_minus = compute_loss();
    n.val = orig;
    return (loss_plus - loss_minus) / (2.0f * eps);
}

// 相對誤差比較
bool check_grad(float analytic, float numeric, float tol = 5e-2f) {
    float diff = fabs(analytic - numeric);
    float scale = (float)std::max((double)1.0f, std::max(fabs((double)analytic), fabs((double)numeric)));
    return diff / scale < tol;
}

// ============================================================================
// Test 1: Embedding — forward 查表 + backward 梯度累加
// ============================================================================
void test_embedding() {
    std::cout << "\n═══ Test 1: Embedding ═══" << std::endl;

    int vocab_size = 4, d_model = 3, seq_len = 2;

    tensor *weight = new tensor({1, 1, vocab_size, d_model});
    tensor *output = new tensor({1, 1, seq_len, d_model});

    // 設定已知權重：vocab i = [i*3+1, i*3+2, i*3+3]
    for (int i = 0; i < vocab_size; i++)
        for (int j = 0; j < d_model; j++)
            weight->data[i * d_model + j].val = (float)(i * d_model + j + 1);

    zero_diffs(weight); zero_diffs(output);

    Embedding emb(*output, *weight, vocab_size, d_model, seq_len);

    // 輸入 token [2, 0] → 查 vocab[2] 和 vocab[0]
    emb.set_indices({2, 0});
    emb.forward();

    // Forward 檢查
    CHECK("forward: output[0,:] = weight[2,:] = [7,8,9]",
          fabs(output->data[0].val - 7.0f) < 1e-5f &&
          fabs(output->data[1].val - 8.0f) < 1e-5f &&
          fabs(output->data[2].val - 9.0f) < 1e-5f);
    CHECK("forward: output[1,:] = weight[0,:] = [1,2,3]",
          fabs(output->data[3].val - 1.0f) < 1e-5f &&
          fabs(output->data[4].val - 2.0f) < 1e-5f &&
          fabs(output->data[5].val - 3.0f) < 1e-5f);

    // Backward 檢查
    zero_diffs(weight); zero_diffs(output);
    for (int i = 0; i < seq_len * d_model; i++)
        output->data[i].diff = 1.0f;
    emb.backward();

    CHECK("backward: weight[2,:] gets gradient (token 0 used it)",
          fabs(weight->data[2 * d_model + 0].diff - 1.0f) < 1e-5f);
    CHECK("backward: weight[0,:] gets gradient (token 1 used it)",
          fabs(weight->data[0 * d_model + 0].diff - 1.0f) < 1e-5f);
    CHECK("backward: weight[1,:] = 0 (unused token)",
          fabs(weight->data[1 * d_model + 0].diff) < 1e-5f);
    CHECK("backward: weight[3,:] = 0 (unused token)",
          fabs(weight->data[3 * d_model + 0].diff) < 1e-5f);

    // 測試重複 token：[1, 1] → weight[1] 應收到兩倍梯度
    emb.set_indices({1, 1});
    emb.forward();
    zero_diffs(weight); zero_diffs(output);
    for (int i = 0; i < seq_len * d_model; i++)
        output->data[i].diff = 1.0f;
    emb.backward();
    CHECK("backward: duplicate token → gradient accumulates (2x)",
          fabs(weight->data[1 * d_model + 0].diff - 2.0f) < 1e-5f);

    delete weight; delete output;
}

// ============================================================================
// Test 2: TextMatmul — forward C=A×B + backward 數值梯度
// ============================================================================
void test_text_matmul() {
    std::cout << "\n═══ Test 2: TextMatmul ═══" << std::endl;

    int M = 2, K = 3, N = 2;

    tensor *input = new tensor({1, 1, M, K});
    tensor *weight = new tensor({1, 1, K, N});
    tensor *output = new tensor({1, 1, M, N});

    // input = [[1,2,3],[4,5,6]]
    float in_vals[] = {1,2,3, 4,5,6};
    for (int i = 0; i < M*K; i++) input->data[i].val = in_vals[i];

    // weight = [[1,4],[2,5],[3,6]]
    float w_vals[] = {1,4, 2,5, 3,6};
    for (int i = 0; i < K*N; i++) weight->data[i].val = w_vals[i];

    zero_diffs(input); zero_diffs(weight); zero_diffs(output);

    TextMatmul mm(*output, *input, *weight, M, K, N);
    mm.forward();

    // [1,2,3]×[[1,4],[2,5],[3,6]] = [1+4+9, 4+10+18] = [14, 32]
    CHECK("forward: [1,2,3]·W = [14, 32]",
          fabs(output->data[0].val - 14.0f) < 1e-4f &&
          fabs(output->data[1].val - 32.0f) < 1e-4f);
    // [4,5,6]×W = [4+10+18, 16+25+36] = [32, 77]
    CHECK("forward: [4,5,6]·W = [32, 77]",
          fabs(output->data[2].val - 32.0f) < 1e-4f &&
          fabs(output->data[3].val - 77.0f) < 1e-4f);

    // 數值梯度檢查（loss = sum(output)）
    auto compute_loss = [&]() -> float {
        mm.forward();
        float loss = 0;
        for (int i = 0; i < M*N; i++) loss += output->data[i].val;
        return loss;
    };

    // 解析梯度
    zero_diffs(input); zero_diffs(weight); zero_diffs(output);
    for (int i = 0; i < M*N; i++) output->data[i].diff = 1.0f;
    mm.backward();

    // 存下解析梯度
    std::vector<float> w_analytic(K*N), in_analytic(M*K);
    for (int i = 0; i < K*N; i++) w_analytic[i] = weight->data[i].diff;
    for (int i = 0; i < M*K; i++) in_analytic[i] = input->data[i].diff;

    // 比對數值梯度
    bool weight_ok = true;
    for (int i = 0; i < K*N; i++) {
        // 恢復原始值（numerical_grad 會自動恢復）
        float ng = numerical_grad(weight->data[i], compute_loss);
        if (!check_grad(w_analytic[i], ng)) {
            std::cout << "  ⚠️ weight[" << i << "] analytic=" << w_analytic[i]
                      << " numeric=" << ng << std::endl;
            weight_ok = false;
        }
    }
    CHECK("backward: weight gradients ≈ numerical", weight_ok);

    bool input_ok = true;
    for (int i = 0; i < M*K; i++) {
        float ng = numerical_grad(input->data[i], compute_loss);
        if (!check_grad(in_analytic[i], ng)) {
            std::cout << "  ⚠️ input[" << i << "] analytic=" << in_analytic[i]
                      << " numeric=" << ng << std::endl;
            input_ok = false;
        }
    }
    CHECK("backward: input gradients ≈ numerical", input_ok);

    delete input; delete weight; delete output;
}

// ============================================================================
// Test 3: TextAdd — forward a+b + backward 梯度直傳
// ============================================================================
void test_text_add() {
    std::cout << "\n═══ Test 3: TextAdd ═══" << std::endl;

    int size = 4;
    tensor *a = new tensor({1, 1, 1, size});
    tensor *b = new tensor({1, 1, 1, size});
    tensor *out = new tensor({1, 1, 1, size});

    for (int i = 0; i < size; i++) {
        a->data[i].val = (float)(i + 1);      // [1, 2, 3, 4]
        b->data[i].val = (float)(i + 10);     // [10, 11, 12, 13]
    }
    zero_diffs(a); zero_diffs(b); zero_diffs(out);

    TextAdd add_op(*out, *a, *b, size, true);
    add_op.forward();

    CHECK("forward: 1+10=11, 4+13=17",
          fabs(out->data[0].val - 11.0f) < 1e-5f &&
          fabs(out->data[3].val - 17.0f) < 1e-5f);

    // Backward: ∂(a+b)/∂a = 1, ∂(a+b)/∂b = 1
    zero_diffs(a); zero_diffs(b); zero_diffs(out);
    for (int i = 0; i < size; i++) out->data[i].diff = 2.0f;
    add_op.backward();

    CHECK("backward: ∂L/∂a = upstream grad = 2.0",
          fabs(a->data[0].diff - 2.0f) < 1e-5f &&
          fabs(a->data[3].diff - 2.0f) < 1e-5f);
    CHECK("backward: ∂L/∂b = upstream grad = 2.0",
          fabs(b->data[0].diff - 2.0f) < 1e-5f &&
          fabs(b->data[3].diff - 2.0f) < 1e-5f);

    delete a; delete b; delete out;
}

// ============================================================================
// Test 4: TextReLU — forward max(0,x) + backward 梯度門控
// ============================================================================
void test_text_relu() {
    std::cout << "\n═══ Test 4: TextReLU ═══" << std::endl;

    int size = 4;
    tensor *input = new tensor({1, 1, 1, size});
    tensor *output = new tensor({1, 1, 1, size});

    float vals[] = {-2.0f, 0.0f, 3.0f, -0.5f};
    for (int i = 0; i < size; i++) input->data[i].val = vals[i];
    zero_diffs(input); zero_diffs(output);

    TextReLU relu_op(*output, *input, size);
    relu_op.forward();

    CHECK("forward: relu([-2, 0, 3, -0.5]) = [0, 0, 3, 0]",
          fabs(output->data[0].val) < 1e-5f &&
          fabs(output->data[1].val) < 1e-5f &&
          fabs(output->data[2].val - 3.0f) < 1e-5f &&
          fabs(output->data[3].val) < 1e-5f);

    // Backward：正值位置梯度通過，負值位置梯度 = 0
    zero_diffs(input); zero_diffs(output);
    for (int i = 0; i < size; i++) output->data[i].diff = 1.0f;
    relu_op.backward();

    CHECK("backward: negative → grad blocked (0.0)",
          fabs(input->data[0].diff) < 1e-5f &&
          fabs(input->data[3].diff) < 1e-5f);
    CHECK("backward: positive → grad passes (1.0)",
          fabs(input->data[2].diff - 1.0f) < 1e-5f);

    delete input; delete output;
}

// ============================================================================
// Test 5: CausalAttention — forward 因果遮罩 + backward 數值梯度
// ============================================================================
void test_causal_attention() {
    std::cout << "\n═══ Test 5: CausalAttention ═══" << std::endl;

    int seq_len = 3, d_k = 2;

    tensor *Q = new tensor({1, 1, seq_len, d_k});
    tensor *K = new tensor({1, 1, seq_len, d_k});
    tensor *V = new tensor({1, 1, seq_len, d_k});
    tensor *out = new tensor({1, 1, seq_len, d_k});

    srand(123);
    for (int i = 0; i < seq_len * d_k; i++) {
        Q->data[i].val = ((float)rand() / RAND_MAX - 0.5f) * 2;
        K->data[i].val = ((float)rand() / RAND_MAX - 0.5f) * 2;
        V->data[i].val = ((float)rand() / RAND_MAX - 0.5f) * 2;
    }
    zero_diffs(Q); zero_diffs(K); zero_diffs(V); zero_diffs(out);

    CausalAttention attn(*out, *Q, *K, *V, seq_len, d_k);
    attn.forward();

    // 因果遮罩：position 0 只能看自己
    CHECK("forward: pos 0 self-attn ≈ 1.0 (can only see itself)",
          fabs(attn.attention_weights[0 * seq_len + 0] - 1.0f) < 1e-4f);
    CHECK("forward: pos 0 → pos 1,2 masked (≈ 0)",
          attn.attention_weights[0 * seq_len + 1] < 1e-5f &&
          attn.attention_weights[0 * seq_len + 2] < 1e-5f);
    CHECK("forward: pos 1 → pos 2 masked (≈ 0)",
          attn.attention_weights[1 * seq_len + 2] < 1e-5f);
    // pos 2 可以看 0,1,2（全部）
    float sum_pos2 = 0;
    for (int j = 0; j < seq_len; j++)
        sum_pos2 += attn.attention_weights[2 * seq_len + j];
    CHECK("forward: pos 2 attn weights sum to 1.0",
          fabs(sum_pos2 - 1.0f) < 1e-4f);

    // 數值梯度（loss = sum(output)）
    auto compute_loss = [&]() -> float {
        attn.forward();
        float loss = 0;
        for (int i = 0; i < seq_len * d_k; i++) loss += out->data[i].val;
        return loss;
    };

    // Q 的梯度
    zero_diffs(Q); zero_diffs(K); zero_diffs(V); zero_diffs(out);
    for (int i = 0; i < seq_len * d_k; i++) out->data[i].diff = 1.0f;
    attn.backward();

    std::vector<float> q_analytic(seq_len * d_k);
    for (int i = 0; i < seq_len * d_k; i++) q_analytic[i] = Q->data[i].diff;

    bool q_ok = true;
    for (int i = 0; i < seq_len * d_k; i++) {
        float ng = numerical_grad(Q->data[i], compute_loss);
        if (!check_grad(q_analytic[i], ng)) {
            std::cout << "  ⚠️ Q[" << i << "] analytic=" << q_analytic[i] << " numeric=" << ng << std::endl;
            q_ok = false;
        }
    }
    CHECK("backward: Q gradients ≈ numerical", q_ok);

    // K 的梯度
    zero_diffs(Q); zero_diffs(K); zero_diffs(V); zero_diffs(out);
    for (int i = 0; i < seq_len * d_k; i++) out->data[i].diff = 1.0f;
    attn.backward();

    std::vector<float> k_analytic(seq_len * d_k);
    for (int i = 0; i < seq_len * d_k; i++) k_analytic[i] = K->data[i].diff;

    bool k_ok = true;
    for (int i = 0; i < seq_len * d_k; i++) {
        float ng = numerical_grad(K->data[i], compute_loss);
        if (!check_grad(k_analytic[i], ng)) {
            std::cout << "  ⚠️ K[" << i << "] analytic=" << k_analytic[i] << " numeric=" << ng << std::endl;
            k_ok = false;
        }
    }
    CHECK("backward: K gradients ≈ numerical", k_ok);

    // V 的梯度
    zero_diffs(Q); zero_diffs(K); zero_diffs(V); zero_diffs(out);
    for (int i = 0; i < seq_len * d_k; i++) out->data[i].diff = 1.0f;
    attn.backward();

    std::vector<float> v_analytic(seq_len * d_k);
    for (int i = 0; i < seq_len * d_k; i++) v_analytic[i] = V->data[i].diff;

    bool v_ok = true;
    for (int i = 0; i < seq_len * d_k; i++) {
        float ng = numerical_grad(V->data[i], compute_loss);
        if (!check_grad(v_analytic[i], ng)) {
            std::cout << "  ⚠️ V[" << i << "] analytic=" << v_analytic[i] << " numeric=" << ng << std::endl;
            v_ok = false;
        }
    }
    CHECK("backward: V gradients ≈ numerical", v_ok);

    delete Q; delete K; delete V; delete out;
}

// ============================================================================
// Test 6: TextCrossEntropy — forward loss + backward softmax-onehot
// ============================================================================
void test_text_cross_entropy() {
    std::cout << "\n═══ Test 6: TextCrossEntropy ═══" << std::endl;

    int seq_len = 2, vocab_size = 3;

    tensor *logits = new tensor({1, 1, seq_len, vocab_size});
    tensor *loss = new tensor({1, 1, 1, 1});

    // pos 0: [2.0, 1.0, 0.1] target=0
    // pos 1: [0.1, 0.2, 3.0] target=2
    float l_vals[] = {2.0f, 1.0f, 0.1f, 0.1f, 0.2f, 3.0f};
    for (int i = 0; i < seq_len * vocab_size; i++)
        logits->data[i].val = l_vals[i];
    zero_diffs(logits); zero_diffs(loss);

    TextCrossEntropy ce(*loss, *logits, seq_len, vocab_size);
    ce.set_targets({0, 2});
    ce.forward();

    // 手算驗證
    // pos 0: softmax([2,1,0.1]) → p[0] ≈ 0.659, loss0 = -ln(0.659) ≈ 0.417
    // pos 1: softmax([0.1,0.2,3]) → p[2] ≈ 0.903, loss1 = -ln(0.903) ≈ 0.102
    // avg ≈ 0.26
    CHECK("forward: loss ≈ 0.26",
          fabs(loss->data[0].val - 0.26f) < 0.05f);
    std::cout << "    (actual loss = " << std::fixed << std::setprecision(4) 
              << loss->data[0].val << ")" << std::endl;

    // 數值梯度
    auto compute_loss = [&]() -> float {
        ce.forward();
        return loss->data[0].val;
    };

    zero_diffs(logits); zero_diffs(loss);
    ce.backward();

    std::vector<float> analytic(seq_len * vocab_size);
    for (int i = 0; i < seq_len * vocab_size; i++)
        analytic[i] = logits->data[i].diff;

    bool grad_ok = true;
    for (int i = 0; i < seq_len * vocab_size; i++) {
        float ng = numerical_grad(logits->data[i], compute_loss);
        if (!check_grad(analytic[i], ng)) {
            std::cout << "  ⚠️ logits[" << i << "] analytic=" << analytic[i]
                      << " numeric=" << ng << std::endl;
            grad_ok = false;
        }
    }
    CHECK("backward: logit gradients ≈ numerical", grad_ok);

    // target 位置梯度 = (softmax - 1)/seq_len < 0
    CHECK("backward: target pos grads < 0 (softmax - 1)",
          analytic[0] < 0 && analytic[5] < 0);
    // 非 target 位置梯度 > 0
    CHECK("backward: non-target pos grads > 0 (softmax > 0)",
          analytic[1] > 0 && analytic[2] > 0);

    delete logits; delete loss;
}

// ============================================================================
// Test 7: SGD Update — w = w - lr * grad，清除梯度
// ============================================================================
void test_sgd_update() {
    std::cout << "\n═══ Test 7: SGD Update ═══" << std::endl;

    int M = 1, K = 2, N = 2;
    tensor *input = new tensor({1, 1, M, K});
    tensor *weight = new tensor({1, 1, K, N});
    tensor *output = new tensor({1, 1, M, N});

    input->data[0].val = 1.0f; input->data[1].val = 2.0f;
    weight->data[0].val = 0.5f; weight->data[1].val = 0.3f;
    weight->data[2].val = 0.7f; weight->data[3].val = 0.1f;
    zero_diffs(input); zero_diffs(weight); zero_diffs(output);

    TextMatmul mm(*output, *input, *weight, M, K, N);
    cfg.lr = 0.1f;

    mm.forward();
    for (int i = 0; i < M*N; i++) output->data[i].diff = 1.0f;
    mm.backward();

    float w0_before = weight->data[0].val;
    float w0_grad = weight->data[0].diff;
    float expected = w0_before - cfg.lr * w0_grad;

    mm.update();

    CHECK("SGD: w_new = w_old - lr * grad",
          fabs(weight->data[0].val - expected) < 1e-5f);
    std::cout << "    (w: " << w0_before << " → " << weight->data[0].val
              << ", expected " << expected << ")" << std::endl;
    CHECK("SGD: diff cleared after update",
          fabs(weight->data[0].diff) < 1e-9f);
    CHECK("SGD: diffs vector cleared",
          weight->data[0].diffs.empty());

    delete input; delete weight; delete output;
}

// ============================================================================
// Test 8: End-to-End — 完整 pipeline forward → backward → update → loss↓
// ============================================================================
void test_end_to_end() {
    std::cout << "\n═══ Test 8: End-to-End ═══" << std::endl;

    int vocab_size = 4, d_model = 8, d_ff = 16, seq_len = 3;
    srand(42);
    cfg.lr = 0.05f;

    // 建立 tensor
    tensor *emb_w = new tensor({1, 1, vocab_size, d_model});
    tensor *pos_enc = new tensor({1, 1, seq_len, d_model});
    tensor *emb_out = new tensor({1, 1, seq_len, d_model});
    tensor *pos_out = new tensor({1, 1, seq_len, d_model});
    tensor *Wq = new tensor({1, 1, d_model, d_model});
    tensor *Wk = new tensor({1, 1, d_model, d_model});
    tensor *Wv = new tensor({1, 1, d_model, d_model});
    tensor *Q = new tensor({1, 1, seq_len, d_model});
    tensor *K = new tensor({1, 1, seq_len, d_model});
    tensor *V = new tensor({1, 1, seq_len, d_model});
    tensor *attn_out = new tensor({1, 1, seq_len, d_model});
    tensor *res1 = new tensor({1, 1, seq_len, d_model});
    tensor *ffn_w1 = new tensor({1, 1, d_model, d_ff});
    tensor *ffn_w2 = new tensor({1, 1, d_ff, d_model});
    tensor *ffn_h = new tensor({1, 1, seq_len, d_ff});
    tensor *ffn_r = new tensor({1, 1, seq_len, d_ff});
    tensor *ffn_out = new tensor({1, 1, seq_len, d_model});
    tensor *res2 = new tensor({1, 1, seq_len, d_model});
    tensor *out_w = new tensor({1, 1, d_model, vocab_size});
    tensor *logits_t = new tensor({1, 1, seq_len, vocab_size});
    tensor *loss_t = new tensor({1, 1, 1, 1});

    auto init_small = [](tensor *t) {
        for (size_t i = 0; i < t->data.size(); i++) {
            t->data[i].val = ((float)rand() / RAND_MAX - 0.5f) * 0.2f;
            t->data[i].diff = 0.0f;
            t->data[i].diffs.clear();
        }
    };
    init_small(emb_w); init_small(pos_enc);
    init_small(Wq); init_small(Wk); init_small(Wv);
    init_small(ffn_w1); init_small(ffn_w2); init_small(out_w);
    // 中間 tensor 也歸零
    tensor *intermediates[] = {emb_out, pos_out, Q, K, V, attn_out, res1,
                               ffn_h, ffn_r, ffn_out, res2, logits_t, loss_t};
    for (auto t : intermediates) init_small(t);

    // 建立 ops
    Embedding emb(*emb_out, *emb_w, vocab_size, d_model, seq_len);
    TextAdd pos_add(*pos_out, *emb_out, *pos_enc, seq_len * d_model, true);
    TextMatmul q_proj(*Q, *pos_out, *Wq, seq_len, d_model, d_model);
    TextMatmul k_proj(*K, *pos_out, *Wk, seq_len, d_model, d_model);
    TextMatmul v_proj(*V, *pos_out, *Wv, seq_len, d_model, d_model);
    CausalAttention attn(*attn_out, *Q, *K, *V, seq_len, d_model);
    TextAdd res1_op(*res1, *pos_out, *attn_out, seq_len * d_model, false);
    TextMatmul ffn1(*ffn_h, *res1, *ffn_w1, seq_len, d_model, d_ff);
    TextReLU relu(*ffn_r, *ffn_h, seq_len * d_ff);
    TextMatmul ffn2(*ffn_out, *ffn_r, *ffn_w2, seq_len, d_ff, d_model);
    TextAdd res2_op(*res2, *res1, *ffn_out, seq_len * d_model, false);
    TextMatmul out_proj(*logits_t, *res2, *out_w, seq_len, d_model, vocab_size);
    TextCrossEntropy loss_op(*loss_t, *logits_t, seq_len, vocab_size);

    std::vector<opBase*> ops = {
        &emb, &pos_add, &q_proj, &k_proj, &v_proj, &attn, &res1_op,
        &ffn1, &relu, &ffn2, &res2_op, &out_proj, &loss_op
    };

    emb.set_indices({0, 1, 2});
    loss_op.set_targets({1, 2, 3});

    // Step 1: Forward
    for (auto op : ops) op->forward();
    float loss1 = loss_t->data[0].val;
    CHECK("forward: loss is finite", std::isfinite(loss1));
    CHECK("forward: loss > 0", loss1 > 0);
    std::cout << "    (loss = " << std::fixed << std::setprecision(4) << loss1 << ")" << std::endl;

    // Step 2: Backward
    for (int i = ops.size() - 1; i >= 0; i--) ops[i]->backward();

    float total_emb_grad = 0;
    for (int i = 0; i < vocab_size * d_model; i++)
        total_emb_grad += fabs(emb_w->data[i].diff);
    CHECK("backward: gradients reach embedding weights", total_emb_grad > 1e-6f);

    float total_out_grad = 0;
    for (int i = 0; i < d_model * vocab_size; i++)
        total_out_grad += fabs(out_w->data[i].diff);
    CHECK("backward: gradients reach output weights", total_out_grad > 1e-6f);

    // Step 3: Update → loss should decrease
    for (auto op : ops) op->update();
    for (auto op : ops) op->forward();
    float loss2 = loss_t->data[0].val;
    CHECK("update: loss decreased after 1 step", loss2 < loss1);
    std::cout << "    (loss: " << loss1 << " → " << loss2 << ")" << std::endl;

    // Step 4: 多跑幾步，確認持續收斂
    for (int step = 0; step < 20; step++) {
        for (auto op : ops) op->forward();
        for (int i = ops.size() - 1; i >= 0; i--) ops[i]->backward();
        for (auto op : ops) op->update();
    }
    for (auto op : ops) op->forward();
    float loss3 = loss_t->data[0].val;
    CHECK("training: loss continues to decrease over 20 steps", loss3 < loss2);
    std::cout << "    (loss after 20 more steps: " << loss3 << ")" << std::endl;

    delete emb_w; delete pos_enc; delete emb_out; delete pos_out;
    delete Wq; delete Wk; delete Wv; delete Q; delete K; delete V; delete attn_out;
    delete res1; delete ffn_w1; delete ffn_w2; delete ffn_h; delete ffn_r;
    delete ffn_out; delete res2; delete out_w; delete logits_t; delete loss_t;
}

// ============================================================================
// Test 9: TextGELU — forward 近似值 + backward 數值梯度
// ============================================================================
void test_text_gelu() {
    std::cout << "\n═══ Test 9: TextGELU ═══" << std::endl;

    int size = 5;
    tensor *input = new tensor({1, 1, 1, size});
    tensor *output = new tensor({1, 1, 1, size});

    // GELU 特性：
    // GELU(0) = 0
    // GELU(x) ≈ x for large x > 0
    // GELU(x) ≈ 0 for large x < 0
    float vals[] = {-3.0f, -1.0f, 0.0f, 1.0f, 3.0f};
    for (int i = 0; i < size; i++) input->data[i].val = vals[i];
    zero_diffs(input); zero_diffs(output);

    TextGELU gelu_op(*output, *input, size);
    gelu_op.forward();

    CHECK("forward: GELU(0) ≈ 0",
          fabs(output->data[2].val) < 1e-5f);
    CHECK("forward: GELU(-3) ≈ 0 (negative saturates)",
          fabs(output->data[0].val) < 0.01f);
    CHECK("forward: GELU(3) ≈ 3 (positive passes)",
          fabs(output->data[4].val - 3.0f) < 0.01f);
    CHECK("forward: GELU(-1) < 0 (slightly negative)",
          output->data[1].val < 0);
    CHECK("forward: GELU(1) > 0 and < 1 (smooth gate)",
          output->data[3].val > 0.5f && output->data[3].val < 1.0f);

    // 數值梯度
    auto compute_loss = [&]() -> float {
        gelu_op.forward();
        float loss = 0;
        for (int i = 0; i < size; i++) loss += output->data[i].val;
        return loss;
    };

    zero_diffs(input); zero_diffs(output);
    for (int i = 0; i < size; i++) output->data[i].diff = 1.0f;
    gelu_op.backward();

    std::vector<float> analytic(size);
    for (int i = 0; i < size; i++) analytic[i] = input->data[i].diff;

    bool grad_ok = true;
    for (int i = 0; i < size; i++) {
        float ng = numerical_grad(input->data[i], compute_loss);
        if (!check_grad(analytic[i], ng)) {
            std::cout << "  ⚠️ input[" << i << "] x=" << vals[i]
                      << " analytic=" << analytic[i] << " numeric=" << ng << std::endl;
            grad_ok = false;
        }
    }
    CHECK("backward: gradients ≈ numerical", grad_ok);

    // GELU 導數特性：
    // x=0 → grad ≈ 0.5
    // x>>0 → grad ≈ 1
    // x<<0 → grad ≈ 0
    CHECK("backward: grad(0) ≈ 0.5", fabs(analytic[2] - 0.5f) < 0.05f);
    CHECK("backward: grad(3) ≈ 1.0", fabs(analytic[4] - 1.0f) < 0.05f);
    CHECK("backward: grad(-3) ≈ 0.0", fabs(analytic[0]) < 0.05f);

    delete input; delete output;
}

// ============================================================================
// Test 10: TextMultiHeadAttention — 多頭因果注意力 + 數值梯度
// ============================================================================
void test_text_multi_head_attention() {
    std::cout << "\n═══ Test 10: TextMultiHeadAttention ═══" << std::endl;

    int seq_len = 3, d_model = 4, num_heads = 2;
    // d_k = d_model / num_heads = 2

    tensor *input = new tensor({1, 1, seq_len, d_model});
    tensor *output = new tensor({1, 1, seq_len, d_model});
    tensor *Wq = new tensor({1, 1, d_model, d_model});
    tensor *Wk = new tensor({1, 1, d_model, d_model});
    tensor *Wv = new tensor({1, 1, d_model, d_model});
    tensor *Wo = new tensor({1, 1, d_model, d_model});

    srand(456);
    auto init = [](tensor *t) {
        for (size_t i = 0; i < t->data.size(); i++) {
            t->data[i].val = ((float)rand() / RAND_MAX - 0.5f) * 0.5f;
            t->data[i].diff = 0; t->data[i].diffs.clear();
        }
    };
    init(input); init(Wq); init(Wk); init(Wv); init(Wo);
    zero_diffs(output);

    TextMultiHeadAttention mha(*output, *input, *Wq, *Wk, *Wv, *Wo, seq_len, d_model, num_heads);
    mha.forward();

    // 基本檢查：output 不為 NaN
    bool finite = true;
    for (int i = 0; i < seq_len * d_model; i++)
        if (!std::isfinite(output->data[i].val)) finite = false;
    CHECK("forward: output is finite (no NaN)", finite);

    // 因果遮罩：head 0, pos 0 → 只看自己
    CHECK("forward: head 0 pos 0 self-attn ≈ 1.0",
          fabs(mha.attn_weights[0][0 * seq_len + 0] - 1.0f) < 1e-4f);
    CHECK("forward: head 0 pos 0 → pos 1,2 masked",
          mha.attn_weights[0][0 * seq_len + 1] < 1e-5f);

    // head 1 也有因果遮罩
    CHECK("forward: head 1 pos 0 self-attn ≈ 1.0",
          fabs(mha.attn_weights[1][0 * seq_len + 0] - 1.0f) < 1e-4f);

    // 數值梯度（loss = sum(output)）
    auto compute_loss = [&]() -> float {
        mha.forward();
        float loss = 0;
        for (int i = 0; i < seq_len * d_model; i++) loss += output->data[i].val;
        return loss;
    };

    // input 梯度
    zero_diffs(input); zero_diffs(output);
    zero_diffs(Wq); zero_diffs(Wk); zero_diffs(Wv); zero_diffs(Wo);
    for (int i = 0; i < seq_len * d_model; i++) output->data[i].diff = 1.0f;
    mha.backward();

    std::vector<float> in_analytic(seq_len * d_model);
    for (int i = 0; i < seq_len * d_model; i++) in_analytic[i] = input->data[i].diff;

    bool input_ok = true;
    for (int i = 0; i < seq_len * d_model; i++) {
        float ng = numerical_grad(input->data[i], compute_loss);
        if (!check_grad(in_analytic[i], ng)) {
            std::cout << "  ⚠️ input[" << i << "] analytic=" << in_analytic[i]
                      << " numeric=" << ng << std::endl;
            input_ok = false;
        }
    }
    CHECK("backward: input gradients ≈ numerical", input_ok);

    // Wq 梯度（抽樣檢查前幾個）
    zero_diffs(input); zero_diffs(output);
    zero_diffs(Wq); zero_diffs(Wk); zero_diffs(Wv); zero_diffs(Wo);
    for (int i = 0; i < seq_len * d_model; i++) output->data[i].diff = 1.0f;
    mha.backward();

    std::vector<float> wq_analytic(d_model * d_model);
    for (int i = 0; i < d_model * d_model; i++) wq_analytic[i] = Wq->data[i].diff;

    bool wq_ok = true;
    for (int i = 0; i < d_model * d_model; i++) {
        float ng = numerical_grad(Wq->data[i], compute_loss);
        if (!check_grad(wq_analytic[i], ng)) {
            std::cout << "  ⚠️ Wq[" << i << "] analytic=" << wq_analytic[i]
                      << " numeric=" << ng << std::endl;
            wq_ok = false;
        }
    }
    CHECK("backward: Wq gradients ≈ numerical", wq_ok);

    // Wo 梯度
    zero_diffs(input); zero_diffs(output);
    zero_diffs(Wq); zero_diffs(Wk); zero_diffs(Wv); zero_diffs(Wo);
    for (int i = 0; i < seq_len * d_model; i++) output->data[i].diff = 1.0f;
    mha.backward();

    std::vector<float> wo_analytic(d_model * d_model);
    for (int i = 0; i < d_model * d_model; i++) wo_analytic[i] = Wo->data[i].diff;

    bool wo_ok = true;
    for (int i = 0; i < d_model * d_model; i++) {
        float ng = numerical_grad(Wo->data[i], compute_loss);
        if (!check_grad(wo_analytic[i], ng)) {
            std::cout << "  ⚠️ Wo[" << i << "] analytic=" << wo_analytic[i]
                      << " numeric=" << ng << std::endl;
            wo_ok = false;
        }
    }
    CHECK("backward: Wo gradients ≈ numerical", wo_ok);

    delete input; delete output; delete Wq; delete Wk; delete Wv; delete Wo;
}

// ============================================================================
// Test 11: Adam optimizer — momentum + adaptive lr
// ============================================================================
void test_adam() {
    std::cout << "\n═══ Test 11: Adam Optimizer ═══" << std::endl;

    int size = 4;
    tensor *w = new tensor({1, 1, 1, size});
    for (int i = 0; i < size; i++) w->data[i].val = 1.0f;
    zero_diffs(w);

    AdamState adam;
    adam.init(size);

    // 模擬固定梯度 = 0.1，跑幾步
    float lr = 0.01f;
    for (int t = 1; t <= 10; t++) {
        for (int i = 0; i < size; i++) w->data[i].diff = 0.1f;
        adam.step(w, size, t, lr);
    }

    // Adam 更新後，w 應該比 1.0 小（因為梯度一直是正的）
    CHECK("Adam: w decreased after 10 steps with positive grad",
          w->data[0].val < 1.0f);
    // 所有 w 應該一樣（因為梯度相同）
    CHECK("Adam: all weights equal (same grad)",
          fabs(w->data[0].val - w->data[1].val) < 1e-7f &&
          fabs(w->data[0].val - w->data[3].val) < 1e-7f);
    // diff 應該被清零
    CHECK("Adam: diff cleared", fabs(w->data[0].diff) < 1e-9f);

    // Adam 的自適應特性：大梯度和小梯度的更新幅度不會差太多
    tensor *w2 = new tensor({1, 1, 1, 2});
    w2->data[0].val = 1.0f; w2->data[1].val = 1.0f;
    zero_diffs(w2);
    AdamState adam2;
    adam2.init(2);

    for (int t = 1; t <= 50; t++) {
        w2->data[0].diff = 0.01f;   // 小梯度
        w2->data[1].diff = 10.0f;   // 大梯度
        adam2.step(w2, 2, t, lr);
    }

    float delta_small = 1.0f - w2->data[0].val;  // 小梯度的累積更新
    float delta_large = 1.0f - w2->data[1].val;  // 大梯度的累積更新

    // Adam 的核心特性：自適應 lr 讓兩者更新幅度在同一數量級
    // SGD 的話 delta_large / delta_small = 1000，Adam 會壓到 ~1-10x
    float ratio = delta_large / delta_small;
    CHECK("Adam: adaptive lr (large/small grad ratio < 20x)",
          ratio > 0.5f && ratio < 20.0f);
    std::cout << "    (delta_small=" << delta_small << ", delta_large=" << delta_large
              << ", ratio=" << ratio << ")" << std::endl;

    delete w; delete w2;
}

// ============================================================================
// Test 12: TextLayerNorm — forward 正規化 + backward 數值梯度
// ============================================================================
void test_text_layer_norm() {
    std::cout << "\n═══ Test 12: TextLayerNorm ═══" << std::endl;

    int seq_len = 2, d_model = 4;

    tensor *input = new tensor({1, 1, seq_len, d_model});
    tensor *output = new tensor({1, 1, seq_len, d_model});
    tensor *gamma = new tensor({1, 1, 1, d_model});
    tensor *beta = new tensor({1, 1, 1, d_model});

    // 設定已知輸入
    // row 0: [1, 2, 3, 4]  mean=2.5, var=1.25
    // row 1: [5, 3, 7, 1]  mean=4.0, var=5.0
    float in_vals[] = {1, 2, 3, 4, 5, 3, 7, 1};
    for (int i = 0; i < seq_len * d_model; i++)
        input->data[i].val = in_vals[i];
    zero_diffs(input); zero_diffs(output); zero_diffs(gamma); zero_diffs(beta);

    TextLayerNorm ln(*output, *input, *gamma, *beta, seq_len, d_model);
    ln.forward();

    // row 0: (x - 2.5) / sqrt(1.25 + 1e-5)
    // normalized[0] = (1-2.5)/1.118 ≈ -1.342
    // normalized[3] = (4-2.5)/1.118 ≈ 1.342
    CHECK("forward: row 0 normalized (mean-centered)",
          output->data[0].val < -1.0f && output->data[3].val > 1.0f);

    // row 1: [5,3,7,1] mean=4.0, var=5.0
    // normalized[4] = (5-4)/sqrt(5+ε) ≈ 0.447
    // normalized[7] = (1-4)/sqrt(5+ε) ≈ -1.342
    CHECK("forward: row 1 normalized correctly",
          output->data[4].val > 0.3f && output->data[7].val < -1.0f);

    // 正規化後每 row 的 mean ≈ 0（beta=0 時）
    float row0_mean = 0;
    for (int j = 0; j < d_model; j++) row0_mean += output->data[j].val;
    row0_mean /= d_model;
    CHECK("forward: normalized row mean ≈ 0",
          fabs(row0_mean) < 1e-4f);

    // 數值梯度檢查
    auto compute_loss = [&]() -> float {
        ln.forward();
        float loss = 0;
        for (int i = 0; i < seq_len * d_model; i++) loss += output->data[i].val;
        return loss;
    };

    // 解析梯度
    zero_diffs(input); zero_diffs(output); zero_diffs(gamma); zero_diffs(beta);
    for (int i = 0; i < seq_len * d_model; i++) output->data[i].diff = 1.0f;
    ln.backward();

    // 存下解析梯度
    std::vector<float> in_analytic(seq_len * d_model);
    std::vector<float> g_analytic(d_model);
    for (int i = 0; i < seq_len * d_model; i++) in_analytic[i] = input->data[i].diff;
    for (int j = 0; j < d_model; j++) g_analytic[j] = gamma->data[j].diff;

    // input 梯度
    bool input_ok = true;
    for (int i = 0; i < seq_len * d_model; i++) {
        float ng = numerical_grad(input->data[i], compute_loss);
        if (!check_grad(in_analytic[i], ng)) {
            std::cout << "  ⚠️ input[" << i << "] analytic=" << in_analytic[i]
                      << " numeric=" << ng << std::endl;
            input_ok = false;
        }
    }
    CHECK("backward: input gradients ≈ numerical", input_ok);

    // gamma 梯度
    bool gamma_ok = true;
    for (int j = 0; j < d_model; j++) {
        float ng = numerical_grad(gamma->data[j], compute_loss);
        if (!check_grad(g_analytic[j], ng)) {
            std::cout << "  ⚠️ gamma[" << j << "] analytic=" << g_analytic[j]
                      << " numeric=" << ng << std::endl;
            gamma_ok = false;
        }
    }
    CHECK("backward: gamma gradients ≈ numerical", gamma_ok);

    // beta 梯度 = sum of upstream grads per position
    // 我們設 output.diff = 1 for all，所以 beta.diff = seq_len = 2
    zero_diffs(input); zero_diffs(output); zero_diffs(gamma); zero_diffs(beta);
    for (int i = 0; i < seq_len * d_model; i++) output->data[i].diff = 1.0f;
    ln.backward();
    CHECK("backward: beta grads = seq_len (sum over positions)",
          fabs(beta->data[0].diff - (float)seq_len) < 1e-4f);

    delete input; delete output; delete gamma; delete beta;
}

// ============================================================================
// main
// ============================================================================
int main() {
    std::cout << "╔═══════════════════════════════════════════╗" << std::endl;
    std::cout << "║  TETF Text Transformer Ops — Unit Tests   ║" << std::endl;
    std::cout << "║  Authors: Ryan & Nami ✨                   ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════╝" << std::endl;

    test_embedding();
    test_text_matmul();
    test_text_add();
    test_text_relu();
    test_causal_attention();
    test_text_cross_entropy();
    test_sgd_update();
    test_text_layer_norm();
    test_text_gelu();
    test_text_multi_head_attention();
    test_adam();
    test_end_to_end();

    std::cout << "\n═══════════════════════════════════════════" << std::endl;
    std::cout << "Results: " << tests_passed << " passed, " << tests_failed << " failed" << std::endl;

    if (tests_failed == 0)
        std::cout << "🎉 All tests passed!" << std::endl;
    else
        std::cout << "❌ " << tests_failed << " tests failed!" << std::endl;

    return tests_failed > 0 ? 1 : 0;
}
