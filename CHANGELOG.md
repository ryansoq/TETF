# CHANGELOG - TETF Code Review Fixes

Branch: `fix/code-review`

## Bug Fixes

### 1. `index_` uninitialized in Max_pool::forward (Critical)
- **Problem**: If all pixels are skipped due to padding, `index_` is used uninitialized → UB
- **Fix**: Initialize `index_ = 0`

### 2. `im2col_get_pixel` uses `calloc` for C++ objects (Critical)
- **Problem**: `calloc` doesn't call constructors for `node` objects that contain `std::string` and `std::vector` → UB/crash
- **Fix**: Replace `calloc` with `new node()` and `free()` with `delete`

### 3. Conv::update only updates partial weights (Critical)
- **Problem**: Weight update loop iterates `m_out_c * ks * ks` instead of `m_out_c * m_c * ks * ks`, missing all input channel weights
- **Fix**: Changed loop bound to `m_out_c * m_c * ks * ks` in update(), clear(), and backward()

### 4. Max_pool::update applies SGD to input activations (Critical)
- **Problem**: MaxPool has no trainable weights, but `update()` was modifying input activation values with SGD, corrupting upstream outputs
- **Fix**: `update()` now only clears diffs (same as `clear()` but with QAT quantization)

### 5. Added Softmax + Cross-Entropy loss (Enhancement)
- **Problem**: Using Sigmoid output + MSE loss for classification is suboptimal
- **Fix**: Added `Loss_CrossEntropy` class with numerically stable softmax, `tir_loss_cross_entropy()` wrapper, and updated LeNet model to use logits → cross-entropy instead of sigmoid → MSE

### 6. Better weight initialization (Enhancement)
- **Problem**: `rand() % 1000 * 0.001 - 0.5` gives poor uniform [-0.5, 0.5] init regardless of layer size
- **Fix**: Default Xavier (Glorot) uniform init in tensor constructor; added `init_he()` method with Box-Muller normal for conv layers

### 7. Removed global variables (Refactor)
- **Problem**: `Accuracy`, `lr`, `START_QUANTIZATION`, `Acc_ok`, `global_num`, `tensor_num` were bare globals
- **Fix**: Moved into `TrainConfig` struct with single `cfg` instance; all references updated to `cfg.*`

### 8. `bias`/`group` assignment in function call args (Bug)
- **Problem**: `bias = 0.0` and `group = 1` used as assignments inside function call argument lists — technically defined in C++ but confusing and fragile
- **Fix**: Assign before the function call, pass by value
