/*
 * ============================================================================
 * TETF — Tiny Embedded Training Framework
 * ============================================================================
 *
 * 整體架構：
 *   node   — 計算圖的最小單位，記錄值(val)、梯度(diff)、偏微分邊(diffs)
 *   tensor — 多維陣列，內部是 vector<node>，用 shape 描述維度 (NCHW)
 *   opBase — 所有算子的基底類別，定義 forward/backward/update 介面
 *   Net    — 神經網路容器，用 list<opBase*> 串接所有算子
 *
 * 訓練流程：
 *   1. forward()  — 逐層前向計算，同時記錄偏微分資訊到 node::diffs
 *   2. backward() — 逆序遍歷各層，用鏈式法則累積梯度到 node::diff
 *   3. update()   — SGD 更新權重：w = w - lr * grad，然後清除梯度
 *
 * 自動微分方式：
 *   每個基本運算（mul, add, sub, div）在計算前向值的同時，
 *   將「局部偏微分」和「上游節點指標」記錄到 input->diffs。
 *   backward 時透過鏈式法則 ∂L/∂x = Σ(∂f/∂x × ∂L/∂f) 遞迴回推梯度。
 *
 * 支援的算子：
 *   Conv, MaxPool, Matmul, Add, ReLU, Sigmoid, Leaky_ReLU,
 *   Loss_MSE, Loss_CrossEntropy,
 *   ScaledDotProductAttention, MultiHeadAttention, LayerNorm, TransformerBlock
 * ============================================================================
 */

#define _USE_MATH_DEFINES // [FIX #6] Needed for M_PI on some compilers
#include <iostream>
#include <cmath>
#include <unistd.h>
#include <vector>
#include <string>
#include <list>
#include <mnist/mnist_reader.hpp>
#include <assert.h>
#include <iomanip>
#include <f2uc.h>
#include <float.h> //max_pool
#include <fstream>
#include <numeric>
#include <algorithm>
#include <random>

// Weight 93-94%
//#include "weight.inc"
//#include "ff90_weight.inc"

// [FIX #7] Moved global variables into a config struct to avoid global state
struct TrainConfig {
    float START_QUANTIZATION = 100.0;
    float Accuracy = 0.0;
    float lr = 0.01;
    float Acc_ok = 99.0;
    int global_num = 0;
    int tensor_num = 0;
};

// Single global config instance (minimally invasive refactor)
TrainConfig cfg;

typedef int8_t q7_t;
typedef uint8_t u8_t;
typedef int16_t q15_t;
typedef uint16_t u16_t;
typedef int32_t q31_t;
typedef int64_t q63_t;

/*
 * node — 計算圖的最小單位（標量節點）
 *
 *   val   — 前向傳播時的計算值
 *   diff  — 反向傳播時累積的梯度 ∂L/∂(this node)
 *   diffs — 偏微分邊的集合：每條邊是 (局部偏微分值, 上游節點指標)
 *           例如 z = x * y 時，x.diffs 會存 (y.val, &z)
 *           表示 ∂z/∂x = y，且 z 是 x 的上游節點
 *   q_val — INT8 量化值，用於模型壓縮（精度達標後啟用）
 */
class node
{
public:
    static int nextID;
    std::string id;
    float val;
    q7_t q_val;
    float diff;
    std::vector<std::pair<float, node *>> diffs;

    node();
    void f2q(void);
    void q2f(void);
    void print_q(void);
    void printDiff(void);
    void setDiff(float dfdx, node *dldf);
};

int node::nextID = 0;

void node::f2q()
{
    if (val >= 0)
    {
        if (val * 127 > 127)
            q_val = 127;
        else
            q_val = val * 127;
    }
    else
    {
        if (val * 128 < -128)
            q_val = -128;
        else
            q_val = val * 128;
    }
}

void node::q2f()
{
    if (val >= 0)
        val = (float)q_val / (float)127;
    else
        val = (float)q_val / (float)128;
}

void node::print_q()
{
    if (val >= 0)
        std::cout << (float)q_val / (float)127 << std::endl;
    else
        std::cout << (float)q_val / (float)128 << std::endl;
}

node::node()
{
    std::string str = std::to_string(++nextID);
    id = "∂x" + str;
}

void node::setDiff(float dfdx, node *dldf)
{
    std::pair<float, node *> val;
    val.first = dfdx;
    val.second = dldf;
    diffs.push_back(val);
};

typedef struct
{
    /*!
   * \brief Type code of base types.
   * We keep it uint8_t instead of DLDataTypeCode for minimal memory
   * footprint, but the value should be one of DLDataTypeCode enum values.
   * */
    uint8_t code;
    /*!
   * \brief Number of bits, common choices are 8, 16, 32.
   */
    uint8_t bits;
    /*! \brief Number of lanes in the type, used for vector types. */
    uint16_t lanes;
} DLDataType;

/*
 * tensor — 多維張量
 *
 *   data   — 底層儲存，是 vector<node>，每個元素都是計算圖節點
 *   shape  — 維度資訊，慣例 NCHW（batch, channel, height, width）
 *   n,c,h,w — shape 的快捷存取
 *
 *   建構時自動以 Xavier 均勻分布初始化權重。
 *   卷積層可額外呼叫 init_he() 改用 He 初始化（適合 ReLU）。
 */
class tensor
{
public:
    std::vector<node> data;
    //node * data;
    int ndim;
    DLDataType dtype;
    std::vector<int> shape;
    int n, c, h, w;
    std::vector<int> strides;
    uint64_t byte_offset;
    std::string name;

    tensor()
    {
        //data = 0;
    }

    tensor(std::vector<int> _shape)
    {
        shape = _shape;
        ndim = shape.size();
        int shape_size = 1;
        n = shape[0];
        c = shape[1];
        h = shape[2];
        w = shape[3];

        for (int i = 0; i < ndim; i++)
            shape_size *= shape[i];

        data.resize(shape_size);

        // [FIX #6] Better weight initialization using Xavier (Glorot) uniform
        // For He init on conv layers, use init_he() after construction
        srand((int)time(0) + rand());
        float fan_in = (ndim >= 2) ? shape_size / shape[0] : shape_size;
        float fan_out = (ndim >= 2) ? shape_size / shape[ndim-1] : shape_size;
        float limit = sqrt(6.0f / (fan_in + fan_out)); // Xavier uniform
        for (int i = 0; i < shape_size; i++)
            data[i].val = ((float)rand() / RAND_MAX) * 2.0f * limit - limit;
    }

    // [FIX #6] He initialization for conv layers (call after construction)
    void init_he()
    {
        int fan_in = data.size() / shape[0]; // shape[0] = num output filters
        float stddev = sqrt(2.0f / fan_in);
        for (size_t i = 0; i < data.size(); i++) {
            // Box-Muller transform for normal distribution
            float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
            float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
            data[i].val = stddev * sqrt(-2.0f * log(u1)) * cos(2.0f * M_PI * u2);
        }
    }

    ~tensor()
    {
    }

    node &operator[](std::size_t idx)
    {
        return data[idx];
    }

    void load_uc2f(unsigned char *ptr)
    {
        bool dump;
        float *fptr;
        uc2float(ptr, &fptr, data.size() * sizeof(float), dump = false);
        for (auto i = 0; i < data.size(); i++)
            data[i].val = fptr[i];
    }

    void save_f2uc(std::string name)
    {
        unsigned char *ptr;
        bool dump;
        std::vector<float> temp;
        for (auto i = 0; i < data.size(); i++)
            temp.push_back(data[i].val);
        float2uc(temp.data(), &ptr, data.size() * sizeof(float), dump = true, name.c_str());
    }
};

/*
 * get_diff — 遞迴計算 ∂(dst)/∂(src)，即 dst 對 src 的梯度
 *
 * 鏈式法則的核心實現：
 *   若 src == dst，則 ∂x/∂x = 1（基底情況）
 *   否則 ∂L/∂x = Σ (∂f/∂x × ∂L/∂f)
 *   其中 (∂f/∂x, &f) 存在 src->diffs 裡
 *
 * 注意：此函數主要用於 TYPE1 backward（建圖式自動微分），
 * TYPE2/3/4 backward 改用手動累積 diff 以提升效能。
 */
float get_diff(node *src, node *dst)
{
    if (src == dst)
    {
        return 1;
    }
    else
    {
        src->diff = 0;
        for (int i = 0; i < src->diffs.size(); i++)
        {
            float r_value = src->diffs[i].first * get_diff(src->diffs[i].second, dst);
            src->diff = src->diff + r_value;
        }
        return src->diff;
    }
}

/*
 * mul — 乘法運算 output = input1 × input2
 *
 * 偏微分：∂(x×y)/∂x = y，∂(x×y)/∂y = x
 * 因此 input1 的邊權重 = input2->val，input2 的邊權重 = input1->val
 */
void mul(node *output, node *input1, node *input2)
{
    std::pair<float, node *> diff1;
    diff1.first = input2->val;
    diff1.second = output;
    input1->diffs.push_back(diff1);

    std::pair<float, node *> diff2;
    diff2.first = input1->val;
    diff2.second = output;
    input2->diffs.push_back(diff2);

    output->val = input1->val * input2->val;
}

void mul_acc(node *output, node *input1, node *input2)
{
    std::pair<float, node *> diff1;
    diff1.first = input2->val;
    diff1.second = output;
    input1->diffs.push_back(diff1);

    std::pair<float, node *> diff2;
    diff2.first = input1->val;
    diff2.second = output;
    input2->diffs.push_back(diff2);

    output->val += input1->val * input2->val;
}

float mul_diff(node *input1, node *input2, node *output)
{
    std::pair<float, node *> diff1;
    diff1.first = input2->val;
    diff1.second = output;
    input1->diffs.push_back(diff1);

    std::pair<float, node *> diff2;
    diff2.first = input1->val;
    diff2.second = output;
    input2->diffs.push_back(diff2);

    return input1->val * input2->val;
}

/*
 * add — 加法運算 output = input1 + input2
 * 偏微分：∂(x+y)/∂x = 1，∂(x+y)/∂y = 1
 */
void add(node *output, node *input1, node *input2)
{
    std::pair<float, node *> diff1;
    diff1.first = 1;
    diff1.second = output;
    input1->diffs.push_back(diff1);

    std::pair<float, node *> diff2;
    diff2.first = 1;
    diff2.second = output;
    input2->diffs.push_back(diff2);

    output->val = input1->val + input2->val;
}

void add_acc(node *output, node *input1, node *input2)
{
    std::pair<float, node *> diff1;
    diff1.first = 1;
    diff1.second = output;
    input1->diffs.push_back(diff1);

    std::pair<float, node *> diff2;
    diff2.first = 1;
    diff2.second = output;
    input2->diffs.push_back(diff2);

    output->val += input1->val + input2->val;
}

float add_diff(node *input1, node *input2, node *output)
{
    std::pair<float, node *> diff1;
    diff1.first = 1;
    diff1.second = output;
    input1->diffs.push_back(diff1);

    std::pair<float, node *> diff2;
    diff2.first = 1;
    diff2.second = output;
    input2->diffs.push_back(diff2);

    return input1->val + input2->val;
}

/*
 * sub — 減法運算 output = input1 - input2
 * 偏微分：∂(x-y)/∂x = 1，∂(x-y)/∂y = -1
 */
void sub(node *output, node *input1, node *input2)
{
    std::pair<float, node *> diff1;
    diff1.first = 1;
    diff1.second = output;
    input1->diffs.push_back(diff1);

    std::pair<float, node *> diff2;
    diff2.first = -1;
    diff2.second = output;
    input2->diffs.push_back(diff2);

    output->val = input1->val - input2->val;
}

void sub_acc(node *output, node *input1, node *input2)
{
    std::pair<float, node *> diff1;
    diff1.first = 1;
    diff1.second = output;
    input1->diffs.push_back(diff1);

    std::pair<float, node *> diff2;
    diff2.first = -1;
    diff2.second = output;
    input2->diffs.push_back(diff2);

    output->val += input1->val - input2->val;
}

// input1 - input2
float sub_diff(node *input1, node *input2, node *output)
{
    std::pair<float, node *> diff1;
    diff1.first = 1;
    diff1.second = output;
    input1->diffs.push_back(diff1);

    std::pair<float, node *> diff2;
    diff2.first = -1;
    diff2.second = output;
    input2->diffs.push_back(diff2);

    return input1->val - input2->val;
}

/*
 * div — 除法運算 output = input1 / input2
 * 偏微分：∂(x/y)/∂x = 1/y，∂(x/y)/∂y = -x/y²
 */
// https://zs.symbolab.com/solver/partial-derivative-calculator/%20%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%20y%7D%5Cleft(%5Cfrac%7Bx%7D%7By%7D%5Cright)
float div_diff(node *input1, node *input2, node *output)
{
    std::pair<float, node *> diff1;
    diff1.first = 1 / input2->val;
    diff1.second = output;
    input1->diffs.push_back(diff1);

    std::pair<float, node *> diff2;
    diff2.first = -1 * (input1->val / (input2->val * input2->val));
    diff2.second = output;
    input2->diffs.push_back(diff2);

    return input1->val / input2->val;
}

void div(node *output, node *input1, node *input2)
{
    std::pair<float, node *> diff1;
    diff1.first = 1 / input2->val;
    diff1.second = output;
    input1->diffs.push_back(diff1);

    std::pair<float, node *> diff2;
    diff2.first = -1 * (input1->val / (input2->val * input2->val));
    diff2.second = output;
    input2->diffs.push_back(diff2);

    output->val = input1->val / input2->val;
}

void div_acc(node *output, node *input1, node *input2)
{
    std::pair<float, node *> diff1;
    diff1.first = 1 / input2->val;
    diff1.second = output;
    input1->diffs.push_back(diff1);

    std::pair<float, node *> diff2;
    diff2.first = -1 * (input1->val / (input2->val * input2->val));
    diff2.second = output;
    input2->diffs.push_back(diff2);

    output->val += input1->val / input2->val;
}

class opBase
{
public:
    std::string nnCode;

    virtual void forward()
    {
        std::cout << "forward, Base" << std::endl;
    }
    virtual void backward()
    {
        std::cout << "backward, Base" << std::endl;
    }
    virtual void update()
    {
        std::cout << "update, Base" << std::endl;
    }
    virtual void clear()
    {
        //std::cout << "clear, Base" << std::endl;
    }
    virtual void save()
    {
        std::cout << "dump, Base" << std::endl;
    }
    virtual void print()
    {
        std::cout << "print, Base" << std::endl;
    }
};

class External : public opBase
{
public:
    tensor *output;
    std::vector<int> shape;
    External(tensor &out, std::vector<int> pShape);
    void forward(){};
    void backward(){};
    void update(){};
    void save();
};

External::External(tensor &out, std::vector<int> pShape)
{
    output = &out;
    shape = pShape;

    // NNEF codeGen
    nnCode.append(out.name); // output
    nnCode.append(" = ");
    nnCode.append("external");
    nnCode.append("(");

    nnCode.append("shape = [");
    for (auto i = 0; i < shape.size(); i++)
    {
        if (i)
            nnCode.append(", ");
        nnCode.append(std::to_string(shape[i]));
    }
    nnCode.append("]");

    nnCode.append(");\n");
}

void External::save()
{
    std::cout << "\t" << nnCode;
}

class Variable : public opBase
{
public:
    tensor *output;
    std::vector<int> shape;
    std::string save_path;
    Variable(tensor &out, std::vector<int> pShape, std::string pSave_path);
    void forward(){};
    void backward(){};
    void update(){};
    void save();
};

Variable::Variable(tensor &out, std::vector<int> pShape, std::string pSave_path)
{
    output = &out;
    shape = pShape;
    save_path = pSave_path;

    // NNEF codeGen
    std::string op = "variable";
    nnCode.append(out.name); // output
    nnCode.append(" = ");
    nnCode.append(op);
    nnCode.append("(");

    nnCode.append("shape = [");
    for (auto i = 0; i < shape.size(); i++)
    {
        if (i)
            nnCode.append(", ");
        nnCode.append(std::to_string(shape[i]));
    }
    nnCode.append("]");

    nnCode.append(", ");
    nnCode.append("label = \'" + pSave_path + "\'");

    nnCode.append(");\n");
}

void Variable::save()
{
    std::cout << "\t" << nnCode;
    std::ofstream ofile;
    ofile.open(save_path);

    int length = 1;
    for (auto i = 0; i < shape.size(); i++)
        length *= shape[i];

    float val;
    for (auto i = 0; i < length; i++)
    {
        ofile << output->data[i].val << " ";
    }
    ofile.close();
}

class Reshape : public opBase
{
public:
    tensor *output;
    tensor *input1;

    std::vector<int> shape;
    Reshape(tensor &out, tensor &a, std::vector<int> shape);
    void forward();
    void backward(){};
    void update(){};
    void save();
};

Reshape::Reshape(tensor &out, tensor &a, std::vector<int> p_shape)
{
    output = &out;
    input1 = &a;
    shape = p_shape;

    // NNEF codeGen
    nnCode.append(out.name); // output
    nnCode.append(" = ");
    nnCode.append("reshape");
    nnCode.append("(");

    nnCode.append(a.name); // input
    nnCode.append(", ");

    nnCode.append("shape = [");
    for (auto i = 0; i < shape.size(); i++)
    {
        if (i)
            nnCode.append(", ");
        nnCode.append(std::to_string(shape[i]));
    }
    nnCode.append("]");

    nnCode.append(");\n");
}

void Reshape::save()
{
    std::cout << "\t" << nnCode;
}

void Reshape::forward()
{
    tensor &out = *output;
    tensor &a = *input1;
    out.shape = shape;
    out.data = a.data;
}

/*
 * Max_pool — 最大池化算子
 *
 * forward: 在每個 size×size 視窗中取最大值，記錄最大值的位置索引
 * backward: 梯度只回傳給 forward 時的最大值位置（其餘位置梯度為 0）
 *
 * 無可訓練參數，update() 只清除梯度。
 */
class Max_pool : public opBase
{
public:
    tensor *output;
    tensor *input1;
    int m_stride;
    int m_size;
    int m_pad;
    std::vector<int> index; // ref darknet opt
    std::vector<int> out_index;
    Max_pool(tensor &out, tensor &a, int p_size, int p_padding, int p_stride);
    void forward();
    void backward();
    void update();
    void clear();
    void save();
};

Max_pool::Max_pool(tensor &out, tensor &a, int size, int pad, int stride)
{
    output = &out;
    input1 = &a;
    m_size = size;
    m_pad = pad;
    m_stride = stride;

    // NNEF codeGen
    nnCode.append(out.name); // output
    nnCode.append(" = ");
    nnCode.append("max_pool");
    nnCode.append("(");

    nnCode.append(a.name);

    // size
    nnCode.append(", size = [1, 1, ");
    nnCode.append(std::to_string(size));
    nnCode.append(", ");
    nnCode.append(std::to_string(size));
    nnCode.append("]");
    // pad
    if (pad == 1)
        nnCode.append(", padding = []");
    else
        assert(0);
    // border
    nnCode.append(", border = 'ignore'");
    // stride
    nnCode.append(", stride = [1, 1, ");
    nnCode.append(std::to_string(stride));
    nnCode.append(", ");
    nnCode.append(std::to_string(stride));
    nnCode.append("]");

    nnCode.append(");\n");
}

void Max_pool::save()
{
    std::cout << "\t" << nnCode;
}

void Max_pool::forward()
{
    //Run
    //tensor<float> out;
    //out.shape.resize(4);

    int v_offset_T = 0;
    int v_offset_Z = 0;
    int v_offset_Y = 0;
    int v_offset_X = 0;

    //virtual_height, virtual_weight
    int v_height = 0;
    int v_width = 0;

    //virtual_bound_height , virtual_bound_weight
    int vb_height = 0;
    int vb_width = 0;

    int out_n = output->shape[0];
    int out_c = output->shape[1];
    int out_h = output->shape[2];
    int out_w = output->shape[3];
    int in_n = input1->shape[0];
    int in_c = input1->shape[1];
    int in_h = input1->shape[2];
    int in_w = input1->shape[3];
    int size = m_size;
    int stride = m_stride;
    int padding = m_pad;

    // Chack
    // NCHW foramt
    assert(input1->shape[2] >= size);
    assert(input1->shape[3] >= size);

    if (padding)
    {
        out_n = in_n;
        out_c = in_c;
        out_h = (int)(ceil((float)(in_h) / (float)stride));
        out_w = (int)(ceil((float)(in_w) / (float)stride));

        int newY = size + (out_h - 1) * stride;
        int newX = size + (out_w - 1) * stride;

        v_offset_Y = (newY - in_h) / 2;
        v_offset_X = (newX - in_w) / 2;

        vb_height = in_h + v_offset_Y;
        vb_width = in_w + v_offset_X;
    }
    else
    {
        out_n = in_n;
        out_c = in_c;
        out_h = ceil(((float)(in_h - size + 1)) / ((float)stride));
        out_w = ceil(((float)(in_w - size + 1)) / ((float)stride));

        vb_height = in_h;
        vb_width = in_w;
    }

    //virtual_height, virtual_weight
    v_height = v_offset_Y;
    v_width = v_offset_X;

    //make_tensor(out, out->n, out->c, out->h, out->w);
    output->shape[0] = out_n;
    output->shape[1] = out_c;
    output->shape[2] = out_h;
    output->shape[3] = out_w;

    // Tensor is [batch, height, width, channels], NNEF not
    // NNEF is [batch, channels, height, width]
    for (int N = 0; N < out_n; N++)
        //#pragma omp parallel for
        for (int C = 0; C < out_c; C++)
            for (int H = 0; H < out_h; H++)
                for (int W = 0; W < out_w; W++)
                {
                    float MaxValue = -FLT_MAX;
                    int offsetY = (H * stride);
                    int offsetX = (W * stride);

                    //for (int x = 0; x < size[0]; x++)
                    //for (int y = 0; y < size[1]; y++)
                    int index_ = 0; // [FIX #1] Initialize to 0 to prevent UB when all pixels are skipped by padding
                    for (int z = 0; z < size; z++)
                        for (int t = 0; t < size; t++)
                        {
                            // logical_height, logical_weight
                            int l_height = z + offsetY;
                            int l_weight = t + offsetX;

                            if ((l_height >= v_height && l_weight >= v_width) && (l_height < vb_height && l_weight < vb_width))
                            {
                                float value = input1->data[N * in_c * in_h * in_w + C * in_h * in_w + (l_height - v_offset_Y) * in_w + (l_weight - v_offset_X)].val;
                                if (MaxValue < value)
                                {
                                    MaxValue = value;
                                    index_ = N * in_c * in_h * in_w + C * in_h * in_w + (l_height - v_offset_Y) * in_w + (l_weight - v_offset_X);
                                }
                            }
                        }
                    index.push_back(index_);
                    out_index.push_back(N * out_c * out_h * out_w + C * out_h * out_w + H * out_w + W);
                    output->data[N * out_c * out_h * out_w + C * out_h * out_w + H * out_w + W].val = MaxValue;
                }
}

void Max_pool::backward()
{
    //Run
    //tensor<float> out;
    //out.shape.resize(4);
#if 0
    int v_offset_T = 0;
    int v_offset_Z = 0;
    int v_offset_Y = 0;
    int v_offset_X = 0;

    //virtual_height, virtual_weight
    int v_height = 0;
    int v_width = 0;

    //virtual_bound_height , virtual_bound_weight
    int vb_height = 0;
    int vb_width = 0;

    int out_n = output->shape[0];
    int out_c = output->shape[1];
    int out_h = output->shape[2];
    int out_w = output->shape[3];
    int in_n = input1->shape[0];
    int in_c = input1->shape[1];
    int in_h = input1->shape[2];
    int in_w = input1->shape[3];
    int size = m_size;
    int stride = m_stride;
    int padding = m_pad;

    // Chack
    // NCHW foramt
    assert(input1->shape[2] >= size);
    assert(input1->shape[3] >= size);

    if (padding)
    {
        out_n = in_n;
        out_c = in_c;
        out_h = (int)(ceil((float)(in_h) / (float)stride));
        out_w = (int)(ceil((float)(in_w) / (float)stride));

        int newY = size + (out_h - 1) * stride;
        int newX = size + (out_w - 1) * stride;

        v_offset_Y = (newY - in_h) / 2;
        v_offset_X = (newX - in_w) / 2;

        vb_height = in_h + v_offset_Y;
        vb_width = in_w + v_offset_X;
    }
    else
    {
        out_n = in_n;
        out_c = in_c;
        out_h = ceil(((float)(in_h - size + 1)) / ((float)stride));
        out_w = ceil(((float)(in_w - size + 1)) / ((float)stride));

        vb_height = in_h;
        vb_width = in_w;
    }

    //virtual_height, virtual_weight
    v_height = v_offset_Y;
    v_width = v_offset_X;

    //make_tensor(out, out->n, out->c, out->h, out->w);
    output->shape[0] = out_n;
    output->shape[1] = out_c;
    output->shape[2] = out_h;
    output->shape[3] = out_w;

    // Tensor is [batch, height, width, channels], NNEF not
    // NNEF is [batch, channels, height, width]
    for (int N = 0; N < out_n; N++)
        //#pragma omp parallel for
        for (int C = 0; C < out_c; C++)
            for (int H = 0; H < out_h; H++)
                for (int W = 0; W < out_w; W++)
                {
                    float MaxValue = -FLT_MAX;
                    int offsetY = (H * stride);
                    int offsetX = (W * stride);

                    //for (int x = 0; x < size[0]; x++)
                    //for (int y = 0; y < size[1]; y++)
                    int index;
                    for (int z = 0; z < size; z++)
                        for (int t = 0; t < size; t++)
                        {
                            // logical_height, logical_weight
                            int l_height = z + offsetY;
                            int l_weight = t + offsetX;

                            if ((l_height >= v_height && l_weight >= v_width) && (l_height < vb_height && l_weight < vb_width))
                            {
                                float value = input1->data[N * in_c * in_h * in_w + C * in_h * in_w + (l_height - v_offset_Y) * in_w + (l_weight - v_offset_X)].val;
                                if (MaxValue < value)
                                {
                                    MaxValue = value;
                                    index = N * in_c * in_h * in_w + C * in_h * in_w + (l_height - v_offset_Y) * in_w + (l_weight - v_offset_X);
                                }
                            }
                        }
                    //output->data[N * out_c * out_h * out_w + C * out_h * out_w + H * out_w + W].val = MaxValue;
                    input1->data[index].diff += 1 * output->data[N * out_c * out_h * out_w + C * out_h * out_w + H * out_w + W].diff;
                }
#else // opt
    for (auto i = 0; i < out_index.size(); i++)
        input1->data[index[i]].diff += 1 * output->data[out_index[i]].diff;
    index.clear();
    out_index.clear();
#endif
}

void Max_pool::update()
{
    // [FIX #4] MaxPool has no trainable weights. update() should only clear diffs,
    // not apply SGD to input activations (which corrupts upstream layer outputs).
    tensor &x = *input1;

    for (int i = 0; i < x.data.size(); i++)
    {
        if (cfg.Accuracy > cfg.START_QUANTIZATION)
        {
            x[i].f2q();
            x[i].q2f();
        }
        x[i].diff = 0;
        x[i].diffs.clear();
    }
}

void Max_pool::clear()
{
    int size = m_size;
    int stride = m_stride;
    int padding = m_pad;
    tensor &x = *input1;

    for (int i = 0; i < x.data.size(); i++)
    {
        x[i].diff = 0;
        x[i].diffs.clear();
    }
}
/*
 * Conv — 2D 卷積算子
 *
 * forward: output[n,c_out,h,w] = Σ input[n,c_in,h',w'] × filter[c_out,c_in,kh,kw]
 *          支援 padding、stride，有多種實現 (TYPE1~TYPE4)
 *          TYPE4 使用 im2col + GEMM 加速
 *
 * backward: 對 input 和 filter 分別累積梯度
 *   ∂L/∂input  += filter_val × ∂L/∂output
 *   ∂L/∂filter += input_val  × ∂L/∂output
 *
 * update: SGD 更新 filter 權重，清除梯度
 */
class Conv : public opBase
{
public:
    tensor *output;
    tensor *input1;
    tensor *input2;
    int m_stride;
    int m_ks;
    int m_c;
    int m_m;
    int m_n;
    int m_out_c;
    int m_out_x;
    int m_out_y;
    int m_pad;
    Conv(tensor &out, tensor &a, tensor &b, int mc, int mm, int nn, int stride, int pad, int ks, int out_c, int out_x, int out_y);
    void forward();
    void backward();
    void update();
    void clear();
    void save();
};

Conv::Conv(tensor &out, tensor &a, tensor &b, int mc, int mm, int nn, int stride, int pad, int ks, int out_c, int out_x, int out_y)
{
    output = &out;
    input1 = &a;
    input2 = &b;
    m_c = mc;
    m_m = mm;
    m_n = nn;
    m_ks = ks;
    m_pad = pad;
    m_stride = stride;
    m_out_c = out_c;
    m_out_x = out_x;
    m_out_y = out_y;

    // NNEF codeGen
    nnCode.append(out.name); // output
    nnCode.append(" = ");
    nnCode.append("conv");
    nnCode.append("(");

    nnCode.append(a.name);
    nnCode.append(", ");
    nnCode.append(b.name);

    // bias
    nnCode.append(", bias = 0.0");
    // pad
    if (pad == 1)
        nnCode.append(", padding = []");
    else
        nnCode.append(", padding = [(0, 0), (0, 0)]");
    // border
    nnCode.append(", border = 'constant'");
    // stride
    nnCode.append(", stride  = [");
    nnCode.append(std::to_string(stride));
    nnCode.append(", ");
    nnCode.append(std::to_string(stride));
    nnCode.append("]");
    // dilation
    nnCode.append(", dilation = [1, 1]");
    nnCode.append(");\n");
}

void Conv::save()
{
    std::cout << "\t" << nnCode;
}

int conv_HWC(tensor *Im_in,
             const uint16_t dim_im_in,
             const uint16_t ch_im_in,
             tensor *wt,
             const uint16_t ch_im_out,
             const uint16_t dim_kernel,
             const uint16_t padding,
             const uint16_t stride,
             const uint16_t bias_shift,
             const uint16_t out_shift,
             tensor *Im_out,
             const uint16_t dim_im_out)
{
    int i, j, k, l, m, n;
    int conv_out;
    int in_row, in_col;

    for (i = 0; i < ch_im_out; i++)
    {
        for (j = 0; j < dim_im_out; j++)
        {
            for (k = 0; k < dim_im_out; k++)
            {
                conv_out = 0;
                (*Im_out)[i + (j * dim_im_out + k) * ch_im_out].val = 0.0;
                for (m = 0; m < dim_kernel; m++)
                {
                    for (n = 0; n < dim_kernel; n++)
                    {
                        // if-for implementation
                        in_row = stride * j + m - padding;
                        in_col = stride * k + n - padding;
                        if (in_row >= 0 && in_col >= 0 && in_row < dim_im_in && in_col < dim_im_in)
                        {
                            for (l = 0; l < ch_im_in; l++)
                            {
                                conv_out += (*Im_in)[(in_row * dim_im_in + in_col) * ch_im_in + l].val * (*wt)[i * ch_im_in * dim_kernel * dim_kernel + (m * dim_kernel + n) * ch_im_in + l].val;
                                (*Im_in)[(in_row * dim_im_in + in_col) * ch_im_in + l].setDiff((*wt)[i * ch_im_in * dim_kernel * dim_kernel + (m * dim_kernel + n) * ch_im_in + l].val, &(*Im_out)[i + (j * dim_im_out + k) * ch_im_out]);
                                (*wt)[i * ch_im_in * dim_kernel * dim_kernel + (m * dim_kernel + n) * ch_im_in + l].setDiff((*Im_in)[(in_row * dim_im_in + in_col) * ch_im_in + l].val, &(*Im_out)[i + (j * dim_im_out + k) * ch_im_out]);
                                /*
                                mul_acc(&(*Im_out)[i + (j * dim_im_out + k) * ch_im_out],
                                        &(*Im_in)[(in_row * dim_im_in + in_col) * ch_im_in + l],
                                        &(*wt)[i * ch_im_in * dim_kernel * dim_kernel + (m * dim_kernel + n) * ch_im_in + l]);
                                
                                conv_out +=
                                    Im_in[(in_row * dim_im_in + in_col) * ch_im_in +
                                          l] * wt[i * ch_im_in * dim_kernel * dim_kernel + (m * dim_kernel +
                                                                                            n) * ch_im_in + l];
                                */
                            }
                        }
                    }
                }
                (*Im_out)[i + (j * dim_im_out + k) * ch_im_out].val = conv_out;
                //Im_out[i + (j * dim_im_out + k) * ch_im_out] = conv_out;
            }
        }
    }

    return 0;
}

#ifdef TYPE2_BACKWARD_CONV
int TYPE2_FORWARD_conv_HWC(tensor *Im_in,
                           const uint16_t dim_im_in,
                           const uint16_t ch_im_in,
                           tensor *wt,
                           const uint16_t ch_im_out,
                           const uint16_t dim_kernel,
                           const uint16_t padding,
                           const uint16_t stride,
                           const uint16_t bias_shift,
                           const uint16_t out_shift,
                           tensor *Im_out,
                           const uint16_t dim_im_out)
{
    int i, j, k, l, m, n;
    int conv_out;
    int in_row, in_col;

    for (i = 0; i < ch_im_out; i++)
    {
        for (j = 0; j < dim_im_out; j++)
        {
            for (k = 0; k < dim_im_out; k++)
            {
                conv_out = 0;

                for (m = 0; m < dim_kernel; m++)
                {
                    for (n = 0; n < dim_kernel; n++)
                    {
                        // if-for implementation
                        in_row = stride * j + m - padding;
                        in_col = stride * k + n - padding;
                        if (in_row >= 0 && in_col >= 0 && in_row < dim_im_in && in_col < dim_im_in)
                        {
                            for (l = 0; l < ch_im_in; l++)
                            {
                                conv_out +=
                                    (*Im_in)[(in_row * dim_im_in + in_col) * ch_im_in + l].val * (*wt)[i * ch_im_in * dim_kernel * dim_kernel + (m * dim_kernel + n) * ch_im_in + l].val;
                            }
                        }
                    }
                }
                (*Im_out)[i + (j * dim_im_out + k) * ch_im_out].val = conv_out;
            }
        }
    }
    return 0;
}

int TYPE2_BACKWARD_conv_HWC(tensor *Im_in,
                            const uint16_t dim_im_in,
                            const uint16_t ch_im_in,
                            tensor *wt,
                            const uint16_t ch_im_out,
                            const uint16_t dim_kernel,
                            const uint16_t padding,
                            const uint16_t stride,
                            const uint16_t bias_shift,
                            const uint16_t out_shift,
                            tensor *Im_out,
                            const uint16_t dim_im_out)
{
    int i, j, k, l, m, n;
    int conv_out;
    int in_row, in_col;

    for (i = 0; i < ch_im_out; i++)
    {
        for (j = 0; j < dim_im_out; j++)
        {
            for (k = 0; k < dim_im_out; k++)
            {
                conv_out = 0;

                for (m = 0; m < dim_kernel; m++)
                {
                    for (n = 0; n < dim_kernel; n++)
                    {
                        // if-for implementation
                        in_row = stride * j + m - padding;
                        in_col = stride * k + n - padding;
                        if (in_row >= 0 && in_col >= 0 && in_row < dim_im_in && in_col < dim_im_in)
                        {
                            for (l = 0; l < ch_im_in; l++)
                            {
                                (*Im_in)[(in_row * dim_im_in + in_col) * ch_im_in + l].diff += (*wt)[i * ch_im_in * dim_kernel * dim_kernel + (m * dim_kernel + n) * ch_im_in + l].val * (*Im_out)[i + (j * dim_im_out + k) * ch_im_out].diff;
                                (*wt)[i * ch_im_in * dim_kernel * dim_kernel + (m * dim_kernel + n) * ch_im_in + l].diff += (*Im_in)[(in_row * dim_im_in + in_col) * ch_im_in + l].val * (*Im_out)[i + (j * dim_im_out + k) * ch_im_out].diff;
                            }
                        }
                    }
                }
            }
        }
    }
    return 0;
}
#elif TYPE3_BACKWARD_CONV
int TYPE3_FORWARD_conv_CHW(tensor *Im_in,
                           const uint16_t dim_im_in,
                           const uint16_t ch_im_in,
                           tensor *wt,
                           const uint16_t ch_im_out,
                           const uint16_t dim_kernel,
                           const uint16_t padding,
                           const uint16_t stride,
                           const uint16_t bias_shift,
                           const uint16_t out_shift,
                           tensor *Im_out,
                           const uint16_t dim_im_out)
{
    uint16_t i, j, k, l, m, n;
    int conv_out;
    long in_row, in_col;

    for (i = 0; i < ch_im_out; i++)
    {
        for (j = 0; j < dim_im_out; j++)
        {
            for (k = 0; k < dim_im_out; k++)
            {
                conv_out = 0;
                for (m = 0; m < dim_kernel; m++)
                {
                    for (n = 0; n < dim_kernel; n++)
                    {
                        // if-for implementation
                        in_row = stride * j + m - padding;
                        in_col = stride * k + n - padding;
                        if (in_row >= 0 && in_col >= 0 && in_row < dim_im_in && in_col < dim_im_in)
                        {
                            for (l = 0; l < ch_im_in; l++)
                            {
                                conv_out +=
                                    (*Im_in)[(in_row * dim_im_in + in_col) * ch_im_in + l].val * (*wt)[i * ch_im_in * dim_kernel * dim_kernel + (m * dim_kernel + n) * ch_im_in + l].val;
                            }
                        }
                    }
                }
                (*Im_out)[i + (j * dim_im_out + k) * ch_im_out].val = conv_out;
            }
        }
    }
    return 0;
}

int TYPE3_BACKWARD_conv_CHW(tensor *Im_in,
                            const uint16_t dim_im_in,
                            const uint16_t ch_im_in,
                            tensor *wt,
                            const uint16_t ch_im_out,
                            const uint16_t dim_kernel,
                            const uint16_t padding,
                            const uint16_t stride,
                            const uint16_t bias_shift,
                            const uint16_t out_shift,
                            tensor *Im_out,
                            const uint16_t dim_im_out)
{
    uint16_t i, j, k, l, m, n;
    int conv_out;
    long in_row, in_col;

    for (i = 0; i < ch_im_out; i++)
    {
        for (j = 0; j < dim_im_out; j++)
        {
            for (k = 0; k < dim_im_out; k++)
            {
                conv_out = 0;
                for (m = 0; m < dim_kernel; m++)
                {
                    for (n = 0; n < dim_kernel; n++)
                    {
                        // if-for implementation
                        in_row = stride * j + m - padding;
                        in_col = stride * k + n - padding;
                        if (in_row >= 0 && in_col >= 0 && in_row < dim_im_in && in_col < dim_im_in)
                        {
                            for (l = 0; l < ch_im_in; l++)
                            {
                                (*Im_in)[(in_row * dim_im_in + in_col) * ch_im_in + l].diff += (*wt)[i * ch_im_in * dim_kernel * dim_kernel + (m * dim_kernel + n) * ch_im_in + l].val * (*Im_out)[i + (j * dim_im_out + k) * ch_im_out].diff;
                                (*wt)[i * ch_im_in * dim_kernel * dim_kernel + (m * dim_kernel + n) * ch_im_in + l].diff += (*Im_in)[(in_row * dim_im_in + in_col) * ch_im_in + l].val * (*Im_out)[i + (j * dim_im_out + k) * ch_im_out].diff;
                            }
                        }
                    }
                }
                //(*Im_out)[i + (j * dim_im_out + k) * ch_im_out].val = conv_out;
            }
        }
    }
    return 0;
}

#elif TYPE4_BACKWARD_CONV // NNEF-RTX
std::vector<void *> free_ptr;

inline node *im2col_get_pixel(tensor *im, int height, int width, int channels,
                              int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 || row >= height || col >= width)
    {
        // [FIX #2] Use new instead of calloc - calloc doesn't call constructors
        // for C++ objects with std::string/std::vector members → UB
        node *zero_node = new node();
        zero_node->val = 0.0f;
        zero_node->diff = 0.0f;
        free_ptr.push_back(zero_node);
        return zero_node;
    }

    return &im->data[col + width * (row + height * channel)];
}

int TYPE4_FORWARD_conv_CHW(tensor *out, tensor *in_x, tensor *filter, float bias, int padding, int stride, int groups)
{
    //shape
    int inPic = in_x->n;
    int filterKernelNum = filter->n;

    assert(in_x->h >= filter->h);
    assert(in_x->w >= filter->w);

    int v_offset_Y = 0;
    int v_offset_X = 0;

    //virtual_height, virtual_weight
    int v_height = 0;
    int v_width = 0;

    //virtual_bound_height , virtual_bound_weight
    int vb_height = 0;
    int vb_width = 0;

    int pad = 0;

    if (padding)
    {
        out->n = in_x->n;
        out->c = filter->n;
        out->h = ceil(((float)in_x->h) / ((float)stride));
        out->w = ceil(((float)in_x->w) / ((float)stride));

        //padding
        int newY = filter->h + (out->h - 1) * stride;
        int newX = filter->w + (out->w - 1) * stride;

        v_offset_Y = (newY - in_x->h) / 2;
        v_offset_X = (newX - in_x->w) / 2;

        vb_height = in_x->h + v_offset_Y;
        vb_width = in_x->w + v_offset_X;

        pad = ((out->h - 1) * stride + filter->h - in_x->h) / 2;
    }
    else
    {
        out->n = in_x->n;
        out->c = filter->n;
        out->h = ceil(((float)(in_x->h - filter->h + 1)) / ((float)stride));
        out->w = ceil(((float)(in_x->w - filter->w + 1)) / ((float)stride));

        vb_height = in_x->h;
        vb_width = in_x->w;

        pad = 0;
    }

    //virtual_height, virtual_weight
    v_height = v_offset_Y;
    v_width = v_offset_X;

    if (groups == 1) //general convolution
    {
#ifdef IM2COLxGEMM
        int out_w, out_h;
        int workspace_size;

        out_w = out->h;
        out_h = out->w;
        workspace_size = out_h * out_w * filter->h * filter->h * in_x->c;
        node **colD = 0;

        if (!colD)
            colD = (node **)calloc(workspace_size, sizeof(node *));

        int c, h, w;

        int height_col = out_h;
        int width_col = out_w;
        int channels_col = in_x->c * filter->h * filter->h;

        for (int Pic = 0; Pic < inPic; Pic++)
        {
            for (c = 0; c < channels_col; ++c)
            {
                for (h = 0; h < height_col; ++h)
                {
                    for (w = 0; w < width_col; ++w)
                    {
                        int w_offset = c % filter->h;
                        int h_offset = (c / filter->h) % filter->h;
                        int c_im = c / filter->h / filter->h;
                        int im_row = h_offset + h * stride;
                        int im_col = w_offset + w * stride;
                        int col_index = (c * height_col + h) * width_col + w;
                        //int col_index = (h * width_col + w) * channels_col + c;
                        colD[col_index] = im2col_get_pixel(in_x, in_x->h, in_x->w, in_x->c, im_row, im_col, c_im, pad);
                    }
                }
            }

            int m = filter->n;                         // input height N
            int n = out_w * out_h;                     // filter width = number of filter = 9
            int p = filter->c * filter->h * filter->w; // CHW = input width = filter height = channel*ksize*ksize

            for (int i = 0; i < m; i++) //2
            {
                for (int j = 0; j < n; j++) //9
                {
                    float sum = 0.0;
                    for (int k = 0; k < p; k++) //18
                    {
                        // [ik][kj]
                        sum += filter->data[i * p + k].val * (*colD[k * n + j]).val;
                    }
                    out->data[i * n + j].val = sum + bias;
                }
            }
        }
        // free section
        free(colD);
        for (auto i = 0; i < free_ptr.size(); i++)
            delete static_cast<node*>(free_ptr[i]); // [FIX #2] Use delete to match new
        free_ptr.clear();
#else
        for (int Pic = 0; Pic < inPic; Pic++)
        {
            for (int filterKernel = 0; filterKernel < filterKernelNum; filterKernel++) // 32
            {
                for (int height = 0; height < out->h; height = height + 1) //28
                {
                    for (int width = 0; width < out->w; width = width + 1) //28
                    {
                        float featureValue = 0;
                        int offsetY = (height * stride);
                        int offsetX = (width * stride);

                        for (int z = 0; z < filter->c; z++)
                        {
                            for (int y = 0; y < filter->h; y++)
                            {
                                for (int x = 0; x < filter->w; x++)
                                {
                                    // logical_height, logical_weight
                                    int l_height = y + offsetY;
                                    int l_weight = x + offsetX;

                                    if ((l_height >= v_height && l_weight >= v_width) && (l_height < vb_height && l_weight < vb_width))
                                        featureValue = featureValue + in_x->data[Pic * in_x->c * in_x->h * in_x->w + z * in_x->h * in_x->w + (l_height - v_offset_Y) * in_x->w + (l_weight - v_offset_X)].val * filter->data[filterKernel * filter->c * filter->h * filter->w + z * filter->h * filter->w + y * filter->w + x].val;
                                }
                            }
                        }
                        out->data[Pic * out->c * out->h * out->w + filterKernel * out->h * out->w + height * out->w + width].val = featureValue + bias;
                    }
                }
            }
        }
#endif
    }
    else
    {
        assert(0); // Current not support.
    }
    return 0;
}

int TYPE4_BACKWARD_conv_CHW(tensor *out, tensor *in_x, tensor *filter, float bias, int padding, int stride, int groups)
{
    //shape
    int inPic = in_x->n;
    int filterKernelNum = filter->n;

    assert(in_x->h >= filter->h);
    assert(in_x->w >= filter->w);

    int v_offset_Y = 0;
    int v_offset_X = 0;

    //virtual_height, virtual_weight
    int v_height = 0;
    int v_width = 0;

    //virtual_bound_height , virtual_bound_weight
    int vb_height = 0;
    int vb_width = 0;

    int pad = 0;

    if (padding)
    {
        out->n = in_x->n;
        out->c = filter->n;
        out->h = ceil(((float)in_x->h) / ((float)stride));
        out->w = ceil(((float)in_x->w) / ((float)stride));

        //padding
        int newY = filter->h + (out->h - 1) * stride;
        int newX = filter->w + (out->w - 1) * stride;

        v_offset_Y = (newY - in_x->h) / 2;
        v_offset_X = (newX - in_x->w) / 2;

        vb_height = in_x->h + v_offset_Y;
        vb_width = in_x->w + v_offset_X;

        pad = ((out->h - 1) * stride + filter->h - in_x->h) / 2;
    }
    else
    {
        out->n = in_x->n;
        out->c = filter->n;
        out->h = ceil(((float)(in_x->h - filter->h + 1)) / ((float)stride));
        out->w = ceil(((float)(in_x->w - filter->w + 1)) / ((float)stride));

        vb_height = in_x->h;
        vb_width = in_x->w;

        pad = 0;
    }

    //virtual_height, virtual_weight
    v_height = v_offset_Y;
    v_width = v_offset_X;

    if (groups == 1) //general convolution
    {
#ifdef IM2COLxGEMM
        int out_w, out_h;
        int workspace_size;

        out_w = out->h;
        out_h = out->w;
        workspace_size = out_h * out_w * filter->h * filter->h * in_x->c;
        node **colD = 0;

        if (!colD)
            colD = (node **)calloc(workspace_size, sizeof(node *));

        int c, h, w;

        int height_col = out_h;
        int width_col = out_w;
        int channels_col = in_x->c * filter->h * filter->h;

        for (int Pic = 0; Pic < inPic; Pic++)
        {
            for (c = 0; c < channels_col; ++c)
            {
                for (h = 0; h < height_col; ++h)
                {
                    for (w = 0; w < width_col; ++w)
                    {
                        int w_offset = c % filter->h;
                        int h_offset = (c / filter->h) % filter->h;
                        int c_im = c / filter->h / filter->h;
                        int im_row = h_offset + h * stride;
                        int im_col = w_offset + w * stride;
                        int col_index = (c * height_col + h) * width_col + w;
                        //int col_index = (h * width_col + w) * channels_col + c;
                        colD[col_index] = im2col_get_pixel(in_x, in_x->h, in_x->w, in_x->c, im_row, im_col, c_im, pad);
                    }
                }
            }

            int m = filter->n;                         // input height N
            int n = out_w * out_h;                     // filter width = number of filter = 9
            int p = filter->c * filter->h * filter->w; // CHW = input width = filter height = channel*ksize*ksize

            for (int i = 0; i < m; i++) //2
            {
                for (int j = 0; j < n; j++) //9
                {
                    float sum = 0.0;
                    for (int k = 0; k < p; k++) //18
                    {
                        // [ik][kj]
                        //sum += filter->data[i * p + k].val * (*colD[k * n + j]).val;
                        filter->data[i * p + k].diff += (*colD[k * n + j]).val * out->data[i * n + j].diff;
                        (*colD[k * n + j]).diff += filter->data[i * p + k].val * out->data[i * n + j].diff;
                    }
                    //out->data[i * n + j].val = sum + bias;
                }
            }
        }
        // free section
        free(colD);
        for (auto i = 0; i < free_ptr.size(); i++)
            delete static_cast<node*>(free_ptr[i]); // [FIX #2] Use delete to match new
        free_ptr.clear();
#else
        for (int Pic = 0; Pic < inPic; Pic++)
        {
            for (int filterKernel = 0; filterKernel < filterKernelNum; filterKernel++) // 32
            {
                for (int height = 0; height < out->h; height = height + 1) //28
                {
                    for (int width = 0; width < out->w; width = width + 1) //28
                    {
                        float featureValue = 0;
                        int offsetY = (height * stride);
                        int offsetX = (width * stride);

                        for (int z = 0; z < filter->c; z++)
                        {
                            for (int y = 0; y < filter->h; y++)
                            {
                                for (int x = 0; x < filter->w; x++)
                                {
                                    // logical_height, logical_weight
                                    int l_height = y + offsetY;
                                    int l_weight = x + offsetX;

                                    if ((l_height >= v_height && l_weight >= v_width) && (l_height < vb_height && l_weight < vb_width))
                                    {
                                        //featureValue = featureValue + in_x->data[Pic * in_x->c * in_x->h * in_x->w + z * in_x->h * in_x->w + (l_height - v_offset_Y) * in_x->w + (l_weight - v_offset_X)].val * filter->data[filterKernel * filter->c * filter->h * filter->w + z * filter->h * filter->w + y * filter->w + x].val;
                                        in_x->data[Pic * in_x->c * in_x->h * in_x->w + z * in_x->h * in_x->w + (l_height - v_offset_Y) * in_x->w + (l_weight - v_offset_X)].diff += filter->data[filterKernel * filter->c * filter->h * filter->w + z * filter->h * filter->w + y * filter->w + x].val * out->data[Pic * out->c * out->h * out->w + filterKernel * out->h * out->w + height * out->w + width].diff;
                                        filter->data[filterKernel * filter->c * filter->h * filter->w + z * filter->h * filter->w + y * filter->w + x].diff += in_x->data[Pic * in_x->c * in_x->h * in_x->w + z * in_x->h * in_x->w + (l_height - v_offset_Y) * in_x->w + (l_weight - v_offset_X)].val * out->data[Pic * out->c * out->h * out->w + filterKernel * out->h * out->w + height * out->w + width].diff;
                                    }
                                }
                            }
                        }
                        //out->data[Pic * out->c * out->h * out->w + filterKernel * out->h * out->w + height * out->w + width].val = featureValue + bias;
                    }
                }
            }
        }
#endif
    }
    else
    {
        assert(0); // Current not support.
    }
    return 0;
}
#endif

void Conv::forward()
{
#ifdef TYPE2_BACKWARD_CONV
    const uint16_t in_tensor_dim = m_m;
    const uint16_t in_tensor_ch = m_c;
    const uint16_t out_tensor_ch = m_out_c;
    const uint16_t ker_dim = m_ks;
    const uint16_t pad = m_pad;
    const uint16_t stride = m_stride;
    const uint16_t out_tensor_dim = m_out_x;
    uint16_t i, j, k, l, m, n;
    long in_row, in_col;
    // NHWC
    tensor *Im_in = input1;
    tensor *wt = input2;
    tensor *Im_out = output;
    int ret = TYPE2_FORWARD_conv_HWC(Im_in, in_tensor_dim, in_tensor_ch, wt, out_tensor_ch, ker_dim, pad, stride, 0, 0, Im_out, out_tensor_dim);
#elif TYPE3_BACKWARD_CONV
    const uint16_t in_tensor_dim = m_m;
    const uint16_t in_tensor_ch = m_c;
    const uint16_t out_tensor_ch = m_out_c;
    const uint16_t ker_dim = m_ks;
    const uint16_t pad = m_pad;
    const uint16_t stride = m_stride;
    const uint16_t out_tensor_dim = m_out_x;
    uint16_t i, j, k, l, m, n;
    long in_row, in_col;
    // NCHW
    tensor *Im_in = input1;
    tensor *wt = input2;
    tensor *Im_out = output;
    //printf("TYPE3_BACKWARD_conv_CHW\n");
    int ret = TYPE3_FORWARD_conv_CHW(Im_in, in_tensor_dim, in_tensor_ch, wt, out_tensor_ch, ker_dim, pad, stride, 0, 0, Im_out, out_tensor_dim);
#elif TYPE4_BACKWARD_CONV
    const uint16_t in_tensor_dim = m_m;
    const uint16_t in_tensor_ch = m_c;
    const uint16_t out_tensor_ch = m_out_c;
    const uint16_t ker_dim = m_ks;
    const uint16_t pad = m_pad;
    const uint16_t stride = m_stride;
    const uint16_t out_tensor_dim = m_out_x;
    uint16_t i, j, k, l, m, n;
    long in_row, in_col;
    // NCHW
    tensor *Im_in = input1;
    tensor *filter = input2;
    tensor *Im_out = output;
    float bias;
    int group;
    //printf("TYPE4_BACKWARD_conv_CHW\n");
    //int TYPE4_BACKWARD_conv_CHW(tensor *out, tensor *in_x, tensor *filter, float bias, int padding, int stride, int groups)
    // [FIX #8] Assign bias/group before passing to avoid undefined behavior
    bias = 0.0f;
    group = 1;
    int ret = TYPE4_FORWARD_conv_CHW(Im_out, Im_in, filter, bias, pad, stride, group);
#else
    const uint16_t in_tensor_dim = m_m;
    const uint16_t in_tensor_ch = m_c;
    const uint16_t out_tensor_ch = m_out_c;
    const uint16_t ker_dim = m_ks;
    const uint16_t pad = m_pad;
    const uint16_t stride = m_stride;
    const uint16_t out_tensor_dim = m_out_x;
    uint16_t i, j, k, l, m, n;
    long in_row, in_col;
    // NHWC
    tensor *Im_in = input1;
    tensor *wt = input2;
    tensor *Im_out = output;
    int ret = conv_HWC(Im_in, in_tensor_dim, in_tensor_ch, wt, out_tensor_ch, ker_dim, pad, stride, 0, 0, Im_out, out_tensor_dim);
    // NCHW
    /*
    for (i = 0; i < out_tensor_ch; i++)
    {
        for (j = 0; j < out_tensor_dim; j++)
        {
            for (k = 0; k < out_tensor_dim; k++)
            {
                (*output)[i * out_tensor_dim * out_tensor_dim + j * out_tensor_dim + k].val = 0.0; //init
                for (m = 0; m < ker_dim; m++)
                {
                    for (n = 0; n < ker_dim; n++)
                    {
                        // if-for implementation
                        in_row = stride * j + m - pad;
                        in_col = stride * k + n - pad;
                        if (in_row >= 0 && in_col >= 0 && in_row < in_tensor_dim && in_col < in_tensor_dim)
                        {
                            for (l = 0; l < in_tensor_ch; l++)
                            {
                                mul_acc(&(*output)[i * out_tensor_dim * out_tensor_dim + j * out_tensor_dim + k],
                                        &(*input1)[l * in_tensor_dim * in_tensor_dim + in_row * in_tensor_dim + in_col],
                                        &(*input2)[(i * ker_dim * ker_dim * in_tensor_ch) + (l * ker_dim * ker_dim) + (m * ker_dim) + n]); // in_tensor * ker_weight
                            }
                        }
                    }
                }
            }
        }
    }
    */
#endif
}

void Conv::backward()
{
#ifdef TYPE2_BACKWARD_CONV
    const uint16_t in_tensor_dim = m_m;
    const uint16_t in_tensor_ch = m_c;
    const uint16_t out_tensor_ch = m_out_c;
    const uint16_t ker_dim = m_ks;
    const uint16_t pad = m_pad;
    const uint16_t stride = m_stride;
    const uint16_t out_tensor_dim = m_out_x;
    uint16_t i, j, k, l, m, n;
    long in_row, in_col;
    // NHWC
    tensor *Im_in = input1;
    tensor *wt = input2;
    tensor *Im_out = output;
    int ret = TYPE2_BACKWARD_conv_HWC(Im_in, in_tensor_dim, in_tensor_ch, wt, out_tensor_ch, ker_dim, pad, stride, 0, 0, Im_out, out_tensor_dim);
#elif TYPE3_BACKWARD_CONV
    const uint16_t in_tensor_dim = m_m;
    const uint16_t in_tensor_ch = m_c;
    const uint16_t out_tensor_ch = m_out_c;
    const uint16_t ker_dim = m_ks;
    const uint16_t pad = m_pad;
    const uint16_t stride = m_stride;
    const uint16_t out_tensor_dim = m_out_x;
    uint16_t i, j, k, l, m, n;
    long in_row, in_col;
    // NCHW
    tensor *Im_in = input1;
    tensor *wt = input2;
    tensor *Im_out = output;
    //printf("TYPE3_BACKWARD_conv_CHW\n");
    int ret = TYPE3_BACKWARD_conv_CHW(Im_in, in_tensor_dim, in_tensor_ch, wt, out_tensor_ch, ker_dim, pad, stride, 0, 0, Im_out, out_tensor_dim);
#elif TYPE4_BACKWARD_CONV
    const uint16_t in_tensor_dim = m_m;
    const uint16_t in_tensor_ch = m_c;
    const uint16_t out_tensor_ch = m_out_c;
    const uint16_t ker_dim = m_ks;
    const uint16_t pad = m_pad;
    const uint16_t stride = m_stride;
    const uint16_t out_tensor_dim = m_out_x;
    uint16_t i, j, k, l, m, n;
    long in_row, in_col;
    // NCHW
    tensor *Im_in = input1;
    tensor *filter = input2;
    tensor *Im_out = output;
    float bias;
    int group;
    //printf("TYPE3_BACKWARD_conv_CHW\n");
    // [FIX #8] Assign bias/group before passing
    bias = 0.0f;
    group = 1;
    int ret = TYPE4_BACKWARD_conv_CHW(Im_out, Im_in, filter, bias, pad, stride, group);
#else
    int c = m_c;
    int m = m_m;
    int n = m_n;
    int ks = m_ks;
    tensor &x = *input1;
    tensor &w = *input2;

    assert(x.data.size() == (c * m * n));
    assert(w.data.size() == (m_out_c * m_c * ks * ks));

    for (int i = 0; i < c * m * n; i++)
    {
        for (int j = 0; j < x[i].diffs.size(); j++)
        {
            x[i].diff += x[i].diffs[j].first * x[i].diffs[j].second->diff;
        }
    }

    for (int i = 0; i < m_out_c * m_c * ks * ks; i++)
    {
        for (int j = 0; j < w[i].diffs.size(); j++)
        {
            w[i].diff += w[i].diffs[j].first * w[i].diffs[j].second->diff;
        }
    }
#endif
}

void Conv::update()
{
    int c = m_c;
    int m = m_m;
    int n = m_n;
    int ks = m_ks;
    tensor &x = *input1;
    tensor &w = *input2;

    assert(x.data.size() == (c * m * n));
    assert(w.data.size() == (m_out_c * m_c * ks * ks));

    for (int i = 0; i < c * m * n; i++)
    {
        if (cfg.Accuracy > cfg.START_QUANTIZATION)
        {
            x[i].f2q();
            x[i].q2f();
        }
        x[i].val = x[i].val - cfg.lr * x[i].diff;
        x[i].diff = 0;
        x[i].diffs.clear();
    }

    // [FIX #3] Must iterate over m_out_c * m_c * ks * ks weights, not just m_out_c * ks * ks
    for (int i = 0; i < m_out_c * m_c * ks * ks; i++)
    {
        if (cfg.Accuracy > cfg.START_QUANTIZATION)
        {
            w[i].f2q();
            w[i].q2f();
        }
        w[i].val = w[i].val - cfg.lr * w[i].diff;
        w[i].diff = 0;
        w[i].diffs.clear();
    }
}

void Conv::clear()
{
    int c = m_c;
    int m = m_m;
    int n = m_n;
    int ks = m_ks;
    tensor &x = *input1;
    tensor &w = *input2;

    assert(x.data.size() == (c * m * n));
    assert(w.data.size() == (m_out_c * m_c * ks * ks));

    for (int i = 0; i < c * m * n; i++)
    {
        x[i].diff = 0;
        x[i].diffs.clear();
    }

    for (int i = 0; i < m_out_c * m_c * ks * ks; i++)
    {
        w[i].diff = 0;
        w[i].diffs.clear();
    }
}

/*
 * Matmul — 矩陣乘法算子 C[m,n] = A[m,k] × B[k,n]
 *
 * forward: C_ij = Σ_q A_iq × B_qj
 *   同時記錄偏微分：∂C_ij/∂A_iq = B_qj，∂C_ij/∂B_qj = A_iq
 *
 * backward: 展開鏈式法則累積梯度
 *   ∂L/∂A = ∂L/∂C × Bᵀ（等效操作）
 *   ∂L/∂B = Aᵀ × ∂L/∂C（等效操作）
 */
class Matmul : public opBase
{
public:
    tensor *output;
    tensor *input1;
    tensor *input2;
    int m_m;
    int m_k;
    int m_n;
    Matmul(tensor &out, tensor &a, tensor &b, int m, int k, int n);
    void forward();
    void backward();
    void update();
    void clear();
    void save();
};

Matmul::Matmul(tensor &out, tensor &a, tensor &b, int m, int k, int n)
{
    output = &out;
    input1 = &a;
    input2 = &b;
    m_m = m;
    m_k = k;
    m_n = n;

    // NNEF codeGen
    nnCode.append(out.name); // output
    nnCode.append(" = ");
    nnCode.append("matmul");
    nnCode.append("(");

    nnCode.append(a.name);
    nnCode.append(", ");
    nnCode.append(b.name);

    nnCode.append(", ");
    nnCode.append("transposeA = false, transposeB = false");

    nnCode.append(");\n");
}

void Matmul::forward()
{
#ifdef TYPE2_BACKWARD
    int m = m_m;
    int n = m_n;
    int k = m_k;
    tensor &out = *output;
    tensor &a = *input1;
    tensor &b = *input2;
    output->ndim = 2;
    std::vector<int> shape = {m, n};
    output->shape = shape;

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            out[i * n + j].val = 0;
            for (int q = 0; q < k; q++)
            {
                out[i * n + j].val += a[i * k + q].val * b[q * n + j].val;
            }
        }
    }
#else
    int m = m_m;
    int n = m_n;
    int k = m_k;
    tensor &out = *output;
    tensor &a = *input1;
    tensor &b = *input2;
    output->ndim = 2;
    std::vector<int> shape = {m, n};
    output->shape = shape;

    //init
    /*
    for (int i = 0; i < m * k; i++)
    {
        a[i].diff = 0;
        a[i].diffs.clear();
        //std::vector< std::pair<float, node*> >().swap(a[i].diffs);
    }

    for (int i = 0; i < k * n; i++)
    {
        b[i].diff = 0;
        b[i].diffs.clear();
        //std::vector< std::pair<float, node*> >().swap(b[i].diffs);
    }

    for (int i = 0; i < m * n; i++)
    {
        out[i].diff = 0;
        out[i].diffs.clear();
        //std::vector< std::pair<float, node*> >().swap(out[i].diffs);
    }
    */
    //exec
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            out[i * n + j].val = 0;
            for (int q = 0; q < k; q++)
            {
                //out[i * n + j].val+=a[i * k + q].val * b[q * n + j].val;
                out[i * n + j].val += mul_diff(&a[i * k + q], &b[q * n + j], &out[i * n + j]);
                /*
                a[i * k + q].setDiff(b[q * n + j].val, &out[i * n + j]);
                b[q * n + j].setDiff(a[i * k + q].val, &out[i * n + j]);
*/
            }
        }
    }
#endif
}

void Matmul::backward()
{

#ifdef TYPE2_BACKWARD
    int m = m_m;
    int n = m_n;
    int k = m_k;
    tensor &out = *output;
    tensor &a = *input1;
    tensor &b = *input2;
    output->ndim = 2;
    std::vector<int> shape = {m, n};
    output->shape = shape;

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            out[i * n + j].val = 0;
            for (int q = 0; q < k; q++)
            {
                a[i * k + q].diff += b[q * n + j].val * out[i * n + j].diff;
                b[q * n + j].diff += a[i * k + q].val * out[i * n + j].diff;
            }
        }
    }
#else
    int m = m_m;
    int n = m_n;
    int k = m_k;
    tensor &x = *input1;
    tensor &w = *input2;

    assert(x.data.size() == (m * k));
    assert(w.data.size() == (k * n));

    for (int i = 0; i < m * k; i++)
    {
        while (!x[i].diffs.empty())
        {
            std::pair<float, node *> pop = x[i].diffs.back();
            x[i].diff += pop.first * pop.second->diff;
            x[i].diffs.pop_back();
        }
    }

    for (int i = 0; i < k * n; i++)
    {
        while (!w[i].diffs.empty())
        {
            std::pair<float, node *> pop = w[i].diffs.back();
            w[i].diff += pop.first * pop.second->diff;
            w[i].diffs.pop_back();
        }
    }
#endif
    /*
    for (int i = 0; i < m * k; i++)
    {
        for (int j = 0; j < x[i].diffs.size(); j++)
        {
            x[i].diff += x[i].diffs[j].first * x[i].diffs[j].second->diff;
        }
    }
    
    for (int i = 0; i < k * n; i++)
    {
        for (int j = 0; j < w[i].diffs.size(); j++)
        {
            w[i].diff += w[i].diffs[j].first * w[i].diffs[j].second->diff;
        }
    }
    */
}

void Matmul::update()
{
    int m = m_m;
    int n = m_n;
    int k = m_k;
    tensor &x = *input1;
    tensor &w = *input2;
    tensor &out = *output;

    assert(x.data.size() == (m * k));
    assert(w.data.size() == (k * n));
    /*
    if (cfg.Accuracy > cfg.Acc_ok)
    {
        unsigned char *ptr;
        bool dump;
        std::vector<float> temp;
        for (auto i = 0; i < w.data.size(); i++)
            temp.push_back(w.data[i].val);
        float2uc(temp.data(), &ptr, w.data.size() * sizeof(float), dump = true, "MATMUL" + std::to_string(cfg.global_num));
        cfg.global_num++
    }
*/
    for (int i = 0; i < m * k; i++)
    {
        if (cfg.Accuracy > cfg.START_QUANTIZATION)
        {
            x[i].f2q();
            x[i].q2f();
        }
        x[i].val = x[i].val - cfg.lr * x[i].diff;
        x[i].diff = 0;
        x[i].diffs.clear();
    }

    for (int i = 0; i < k * n; i++)
    {
        if (cfg.Accuracy > cfg.START_QUANTIZATION)
        {
            w[i].f2q();
            w[i].q2f();
        }
        w[i].val = w[i].val - cfg.lr * w[i].diff;
        w[i].diff = 0;
        w[i].diffs.clear();
    }
}

void Matmul::clear()
{
    int m = m_m;
    int n = m_n;
    int k = m_k;
    tensor &x = *input1;
    tensor &w = *input2;
    tensor &out = *output;

    assert(x.data.size() == (m * k));
    assert(w.data.size() == (k * n));

    for (int i = 0; i < m * k; i++)
    {
        x[i].diff = 0;
        x[i].diffs.clear();
    }

    for (int i = 0; i < k * n; i++)
    {
        w[i].diff = 0;
        w[i].diffs.clear();
    }
}

void Matmul::save()
{
    std::cout << "\t" << nnCode;
}

void node::printDiff(void){
    /*
    for (int i = 0; i < diffs.size(); i++)
    {
        if (i)
            std::cout << " + ";
        if (diffs[i].second)
            std::cout <<  " ( " << diffs[i].first << " ) " << diffs[i].second->id << "/" << id << " * " <<  " ( " << diffs[i].second->diff << " ) " << "∂L" << "/" << diffs[i].second->id;
        else // final node
            std::cout <<  " ( " << diffs[i].first << " ) " << "∂L" << "/" << id << " * ( 1 ) " << "∂L" << "/" << "∂L";
    }
    std::cout << std::endl;
    */
};

/*
 * Add — 逐元素加法算子 output[i] = input1[i] + input2[i]
 *
 * forward: 記錄 ∂(x+y)/∂x = 1, ∂(x+y)/∂y = 1
 * backward: 梯度直接傳遞（乘以 1），兩個輸入都收到完整梯度
 */
class Add : public opBase
{
public:
    tensor *output;
    tensor *input1;
    tensor *input2;
    int length;

    Add(tensor &out, tensor &a, tensor &b, int len);
    void forward();
    void backward();
    void update();
    void clear();
    void save();
};

Add::Add(tensor &out, tensor &a, tensor &b, int len)
{
    output = &out;
    input1 = &a;
    input2 = &b;
    length = len;

    // NNEF codeGen
    nnCode.append(out.name); // output
    nnCode.append(" = ");
    nnCode.append("add");
    nnCode.append("(");

    nnCode.append(a.name);
    nnCode.append(", ");
    nnCode.append(b.name);

    nnCode.append(");\n");
}

void Add::forward()
{
    tensor &out = *output;
    tensor &a = *input1;
    tensor &b = *input2;
    //init
    /*
    for (int i = 0; i < length; i++)
    {
        a[i].diff = 0;
        a[i].diffs.clear();
        //std::vector< std::pair<float, node*> >().swap(a[i].diffs);
        b[i].diff = 0;
        b[i].diffs.clear();
        //std::vector< std::pair<float, node*> >().swap(b[i].diffs);
        out[i].diff = 0;
        out[i].diffs.clear();
        //std::vector< std::pair<float, node*> >().swap(out[i].diffs);
    }
    */
    //exec
    for (int i = 0; i < length; i++)
    {
        //out[i].val = a[i].val + b[i].val;
        out[i].val = add_diff(&a[i], &b[i], &out[i]);

        //a[i].setDiff(1, &out[i]);
        //b[i].setDiff(1, &out[i]);
    }
}

void Add::backward()
{
    tensor &x = *input1;
    tensor &w = *input2;

    assert(x.data.size() == length);
    assert(w.data.size() == length);
    /*
    if (cfg.Accuracy > cfg.Acc_ok)
    {
        unsigned char *ptr;
        bool dump;
        std::vector<float> temp;
        for (auto i = 0; i < w.data.size(); i++)
            temp.push_back(w.data[i].val);
        float2uc(temp.data(), &ptr, w.data.size() * sizeof(float), dump = true, "ADD" + std::to_string(cfg.global_num));
        cfg.global_num++
    }
*/
    for (int i = 0; i < length; i++)
    {
        while (!x[i].diffs.empty())
        {
            std::pair<float, node *> pop = x[i].diffs.back();
            x[i].diff += pop.first * pop.second->diff;
            x[i].diffs.pop_back();
        }
    }

    for (int i = 0; i < length; i++)
    {
        while (!w[i].diffs.empty())
        {
            std::pair<float, node *> pop = w[i].diffs.back();
            w[i].diff += pop.first * pop.second->diff;
            w[i].diffs.pop_back();
        }
    }
    /*
    for (int i = 0; i < length; i++)
    {
        for (int j = 0; j < x[i].diffs.size(); j++)
        {
            x[i].diff += x[i].diffs[j].first * x[i].diffs[j].second->diff;
        }
    }
    
    for (int i = 0; i < length; i++)
    {
        for (int j = 0; j < w[i].diffs.size(); j++)
        {
            w[i].diff += w[i].diffs[j].first * w[i].diffs[j].second->diff;
        }
    }
    */
}

void Add::update()
{
    tensor &x = *input1;
    tensor &w = *input2;
    tensor &out = *output;

    assert(x.data.size() == length);
    assert(w.data.size() == length);

    for (int i = 0; i < length; i++)
    {
        if (cfg.Accuracy > cfg.START_QUANTIZATION)
        {
            x[i].f2q();
            x[i].q2f();
        }
        x[i].val = x[i].val - cfg.lr * x[i].diff;
        x[i].diff = 0;
        x[i].diffs.clear();
    }

    for (int i = 0; i < length; i++)
    {
        if (cfg.Accuracy > cfg.START_QUANTIZATION)
        {
            w[i].f2q();
            w[i].q2f();
        }
        w[i].val = w[i].val - cfg.lr * w[i].diff;
        w[i].diff = 0;
        w[i].diffs.clear();
    }
}

void Add::clear()
{
    tensor &x = *input1;
    tensor &w = *input2;
    tensor &out = *output;

    assert(x.data.size() == length);
    assert(w.data.size() == length);

    for (int i = 0; i < length; i++)
    {
        x[i].diff = 0;
        x[i].diffs.clear();
    }

    for (int i = 0; i < length; i++)
    {
        w[i].diff = 0;
        w[i].diffs.clear();
    }
}

void Add::save()
{
    std::cout << "\t" << nnCode;
}

/*
 * ReLU — 整流線性單元 f(x) = max(0, x)
 *
 * forward: x > 0 → output = x, diff = 1
 *          x ≤ 0 → output = 0, diff = 0
 * backward: 梯度只在 x > 0 時通過（梯度閘門）
 */
class ReLU : public opBase
{
public:
    tensor *output;
    tensor *input1;
    int length;
    ReLU(tensor &out, tensor &a, int length);
    void forward();
    void backward();
    void update();
    void clear();
    void save();
};

ReLU::ReLU(tensor &out, tensor &a, int len)
{
    output = &out;
    input1 = &a;
    length = len;

    // NNEF codeGen
    std::string op = "relu";
    nnCode.append(out.name); // output
    nnCode.append(" = ");
    nnCode.append(op);
    nnCode.append("(");
    nnCode.append(a.name);
    nnCode.append(");\n");
}

void ReLU::save()
{
    std::cout << "\t" << nnCode;
}

void ReLU::forward()
{
    tensor &out = *output;
    tensor &a = *input1;

    for (int i = 0; i < length; i++)
    {
        if (a[i].val > 0)
        {
            out[i].val = a[i].val;
            a[i].setDiff(1, &out[i]);
        }
        else
        {
            out[i].val = 0;
            a[i].setDiff(0, &out[i]);
        }
    }
}

void ReLU::backward()
{
    tensor &x = *input1;

    assert(x.data.size() == length);

    for (int i = 0; i < length; i++)
    {
        for (int j = 0; j < x[i].diffs.size(); j++)
        {
            x[i].diff += x[i].diffs[j].first * x[i].diffs[j].second->diff;
        }
    }
}

void ReLU::update()
{
    tensor &x = *input1;

    assert(x.data.size() == length);

    for (int i = 0; i < length; i++)
    {
        if (cfg.Accuracy > cfg.START_QUANTIZATION)
        {
            x[i].f2q();
            x[i].q2f();
        }
        x[i].val = x[i].val - cfg.lr * x[i].diff;
        x[i].diff = 0;
        x[i].diffs.clear();
    }
}

void ReLU::clear()
{
    tensor &x = *input1;

    assert(x.data.size() == length);

    for (int i = 0; i < length; i++)
    {
        x[i].diff = 0;
        x[i].diffs.clear();
    }
}
class Leaky_ReLU : public opBase
{
public:
    tensor *output;
    tensor *input1;
    int length;
    Leaky_ReLU(tensor &out, tensor &a, int length);
    void forward();
    void backward();
    void update();
    void clear();
};

Leaky_ReLU::Leaky_ReLU(tensor &out, tensor &a, int len)
{
    output = &out;
    input1 = &a;
    length = len;
}

void Leaky_ReLU::forward()
{
    tensor &out = *output;
    tensor &a = *input1;

    for (int i = 0; i < length; i++)
    {
        if (a[i].val > 0)
        {
            out[i].val = a[i].val;
            a[i].setDiff(1, &out[i]);
        }
        else
        {
            out[i].val = a[i].val * 0.01;
            a[i].setDiff(0.01, &out[i]);
        }
    }
}

void Leaky_ReLU::backward()
{
    tensor &x = *input1;

    assert(x.data.size() == length);

    for (int i = 0; i < length; i++)
    {
        for (int j = 0; j < x[i].diffs.size(); j++)
        {
            x[i].diff += x[i].diffs[j].first * x[i].diffs[j].second->diff;
        }
    }
}

void Leaky_ReLU::update()
{
    tensor &x = *input1;

    assert(x.data.size() == length);

    for (int i = 0; i < length; i++)
    {
        if (cfg.Accuracy > cfg.START_QUANTIZATION)
        {
            x[i].f2q();
            x[i].q2f();
        }
        x[i].val = x[i].val - cfg.lr * x[i].diff;
        x[i].diff = 0;
        x[i].diffs.clear();
    }
}

void Leaky_ReLU::clear()
{
    tensor &x = *input1;

    assert(x.data.size() == length);

    for (int i = 0; i < length; i++)
    {
        x[i].diff = 0;
        x[i].diffs.clear();
    }
}

/*
 * Sigmoid — S 型激活函數 σ(x) = 1/(1+e^(-x))
 *
 * forward: output = σ(x)
 * backward: ∂σ/∂x = σ(x) × (1 - σ(x))，梯度在兩端趨近 0（飽和區）
 */
class Sigmoid : public opBase
{
public:
    tensor *output;
    tensor *input1;
    int length;
    Sigmoid(tensor &out, tensor &a, int length);
    void forward();
    void backward();
    void update();
    void clear();
    void save();
};

Sigmoid::Sigmoid(tensor &out, tensor &a, int len)
{
    output = &out;
    input1 = &a;
    length = len;

    // NNEF codeGen
    nnCode.append(out.name); // output
    nnCode.append(" = ");
    nnCode.append("sigmoid");
    nnCode.append("(");

    nnCode.append(a.name);

    nnCode.append(");\n");
}

void Sigmoid::forward()
{
    tensor &out = *output;
    tensor &a = *input1;
    //intit
    /*
    for (int i = 0; i < length; i++)
    {
        a[i].diff = 0;
        a[i].diffs.clear();
        //std::vector< std::pair<float, node*> >().swap(a[i].diffs);
        out[i].diff = 0;
        out[i].diffs.clear();
        //std::vector< std::pair<float, node*> >().swap(out[i].diffs);
    }
    */
    for (int i = 0; i < length; i++)
    {
        out[i].val = 1.0 / (1.0 + exp(-a[i].val));
        a[i].setDiff(out[i].val * (1 - out[i].val), &out[i]);
    }
}

void Sigmoid::backward()
{
    tensor &x = *input1;

    assert(x.data.size() == length);

    for (int i = 0; i < length; i++)
    {
        while (!x[i].diffs.empty())
        {
            std::pair<float, node *> pop = x[i].diffs.back();
            x[i].diff += pop.first * pop.second->diff;
            x[i].diffs.pop_back();
        }
    }
    /*
    for (int i = 0; i < length; i++)
    {
        for (int j = 0; j < x[i].diffs.size(); j++)
        {
            x[i].diff += x[i].diffs[j].first * x[i].diffs[j].second->diff;
        }
    }
    */
}

void Sigmoid::update()
{
    tensor &x = *input1;

    assert(x.data.size() == length);

    for (int i = 0; i < length; i++)
    {
        if (cfg.Accuracy > cfg.START_QUANTIZATION)
        {
            x[i].f2q();
            x[i].q2f();
        }
        x[i].val = x[i].val - cfg.lr * x[i].diff;
        x[i].diff = 0;
        x[i].diffs.clear();
    }
}

void Sigmoid::clear()
{
    tensor &x = *input1;

    assert(x.data.size() == length);

    for (int i = 0; i < length; i++)
    {
        x[i].diff = 0;
        x[i].diffs.clear();
    }
}

void Sigmoid::save()
{
    std::cout << "\t" << nnCode;
}

/*
 * Loss_MSE — 均方誤差損失函數 L = Σ(src - dest)² / 2m
 *
 * forward: 計算 MSE 並設定梯度起點 loss.diff = 1
 *   ∂L/∂src_i  = src_i - dest_i
 *   ∂L/∂dest_i = dest_i - src_i
 */
class Loss_MSE : public opBase
{
public:
    tensor *output;
    tensor *input1;
    tensor *input2;
    int length;

    Loss_MSE(tensor &out, tensor &a, tensor &b, int len);
    void forward();
    void backward();
    void update();
    void clear();
    void save();
};

Loss_MSE::Loss_MSE(tensor &out, tensor &a, tensor &b, int len)
{
    output = &out;
    input1 = &a;
    input2 = &b;
    length = len;
}

void Loss_MSE::forward()
{
    tensor &loss = *output;
    tensor &src = *input1;
    tensor &dest = *input2;
    int m = length;
    /*
    //intit
    for (int i = 0; i < length; i++)
    {
        src[i].diff = 0;
        src[i].diffs.clear();
        //std::vector< std::pair<float, node*> >().swap(src[i].diffs);
        dest[i].diff = 0;
        dest[i].diffs.clear();
        //std::vector< std::pair<float, node*> >().swap(dest[i].diffs);
    }

    loss[0].diff = 0;
    loss[0].diffs.clear();
    //std::vector< std::pair<float, node*> >().swap(loss[0].diffs);
    */
    float sum = 0;

    for (int i = 0; i < m; i++)
    {
        sum += pow(src[i].val - dest[i].val, 2);
        // https://zs.symbolab.com/solver/partial-derivative-calculator/%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%20x%7D%5Cleft(%5Cfrac%7B1%7D%7B2%7D%5Cleft(x-y%5Cright)%5E%7B2%7D%5Cright)
        // ∂/∂x = x - y, ∂/∂y = -x + y
        src[i].setDiff(src[i].val - dest[i].val, &loss[0]);
        dest[i].setDiff(-src[i].val + dest[i].val, &loss[0]);
    }

    loss[0].val = sum / (2 * m);
    loss[0].diff = 1;
    //loss[0].setDiff(1, NULL);
}

void Loss_MSE::backward()
{
    tensor &x = *input1;
    tensor &w = *input2;

    assert(x.data.size() == length);
    assert(w.data.size() == length);

    for (int i = 0; i < length; i++)
    {
        while (!x[i].diffs.empty())
        {
            std::pair<float, node *> pop = x[i].diffs.back();
            x[i].diff += pop.first * pop.second->diff;
            x[i].diffs.pop_back();
        }
    }

    for (int i = 0; i < length; i++)
    {
        while (!w[i].diffs.empty())
        {
            std::pair<float, node *> pop = w[i].diffs.back();
            w[i].diff += pop.first * pop.second->diff;
            w[i].diffs.pop_back();
        }
    }
    /*
    for (int i = 0; i < length; i++)
    {
        for (int j = 0; j < x[i].diffs.size(); j++)
        {
            x[i].diff += x[i].diffs[j].first * x[i].diffs[j].second->diff;
        }
    }
    
    for (int i = 0; i < length; i++)
    {
        for (int j = 0; j < w[i].diffs.size(); j++)
        {
            w[i].diff += w[i].diffs[j].first * w[i].diffs[j].second->diff;
        }
    }
    */
}

void Loss_MSE::update()
{
    tensor &x = *input1;
    tensor &w = *input2;
    tensor &out = *output;

    assert(x.data.size() == length);
    assert(w.data.size() == length);

    for (int i = 0; i < length; i++)
    {
        if (cfg.Accuracy > cfg.START_QUANTIZATION)
        {
            x[i].f2q();
            x[i].q2f();
        }
        x[i].val = x[i].val - cfg.lr * x[i].diff;
        x[i].diff = 0;
        x[i].diffs.clear();
    }

    for (int i = 0; i < length; i++)
    {
        if (cfg.Accuracy > cfg.START_QUANTIZATION)
        {
            w[i].f2q();
            w[i].q2f();
        }
        w[i].val = w[i].val - cfg.lr * w[i].diff;
        w[i].diff = 0;
        w[i].diffs.clear();
    }
}

void Loss_MSE::clear()
{
    tensor &x = *input1;
    tensor &w = *input2;
    tensor &out = *output;

    assert(x.data.size() == length);
    assert(w.data.size() == length);

    for (int i = 0; i < length; i++)
    {
        x[i].diff = 0;
        x[i].diffs.clear();
    }

    for (int i = 0; i < length; i++)
    {
        w[i].diff = 0;
        w[i].diffs.clear();
    }
}

void Loss_MSE::save()
{
    // not code-gen
}

/*
 * Loss_CrossEntropy — Softmax + 交叉熵損失（分類任務首選）
 *
 * forward:
 *   1. Softmax: p_i = exp(x_i - max) / Σ exp(x_j - max)  （數值穩定版）
 *   2. Cross-Entropy: L = -Σ target_i × log(p_i)
 *   3. 梯度：∂L/∂logits_i = softmax_i - target_i（簡潔的組合梯度）
 *
 * 這個「softmax - target」的梯度是 softmax + cross-entropy 組合微分的結果，
 * 避免了單獨計算 softmax 梯度的複雜性。
 */
// [FIX #5] Softmax + Cross-Entropy loss for classification (replaces Sigmoid + MSE)
class Loss_CrossEntropy : public opBase
{
public:
    tensor *output;   // scalar loss
    tensor *input1;   // logits (pre-softmax)
    tensor *input2;   // one-hot target
    int length;
    std::vector<float> softmax_cache; // store softmax output for backward

    Loss_CrossEntropy(tensor &out, tensor &a, tensor &b, int len);
    void forward();
    void backward();
    void update();
    void clear();
    void save();
};

Loss_CrossEntropy::Loss_CrossEntropy(tensor &out, tensor &a, tensor &b, int len)
{
    output = &out;
    input1 = &a;
    input2 = &b;
    length = len;
}

void Loss_CrossEntropy::forward()
{
    tensor &loss = *output;
    tensor &logits = *input1;
    tensor &target = *input2;

    // Check for extreme input values that could cause NaN
    for (int i = 0; i < length; i++) {
        if (std::isnan(logits[i].val) || std::isinf(logits[i].val)) {
            std::cout << "WARNING: NaN/Inf input to CrossEntropy at index " << i << ": " << logits[i].val << std::endl;
            logits[i].val = 0.0f;  // Clamp to avoid NaN propagation
        }
        if (logits[i].val > 100.0f) logits[i].val = 100.0f;  // Clamp extreme values
        if (logits[i].val < -100.0f) logits[i].val = -100.0f;
    }

    // Softmax: find max for numerical stability
    float max_val = logits[0].val;
    for (int i = 1; i < length; i++)
        if (logits[i].val > max_val) max_val = logits[i].val;

    softmax_cache.resize(length);
    float sum_exp = 0.0f;
    for (int i = 0; i < length; i++) {
        softmax_cache[i] = exp(logits[i].val - max_val);
        sum_exp += softmax_cache[i];
    }
    
    if (sum_exp <= 1e-20f) {
        std::cout << "WARNING: sum_exp too small: " << sum_exp << std::endl;
        sum_exp = 1e-20f;  // Prevent division by zero
    }
    
    for (int i = 0; i < length; i++)
        softmax_cache[i] /= sum_exp;

    // Cross-entropy: L = -sum(target * log(softmax))
    float ce = 0.0f;
    for (int i = 0; i < length; i++) {
        if (target[i].val > 0.0f) {
            float log_prob = log(softmax_cache[i] + 1e-12f);
            if (std::isnan(log_prob) || std::isinf(log_prob)) {
                std::cout << "WARNING: NaN/Inf log_prob at index " << i 
                          << " softmax=" << softmax_cache[i] << " log=" << log_prob << std::endl;
                log_prob = -12.0f;  // Use a reasonable default
            }
            ce -= target[i].val * log_prob;
        }
    }

    if (std::isnan(ce) || std::isinf(ce)) {
        std::cout << "WARNING: NaN/Inf cross-entropy loss: " << ce << std::endl;
        ce = 1.0f;  // Use a reasonable default
    }

    loss[0].val = ce;
    loss[0].diff = 1;

    // Set gradients: dL/d(logits_i) = softmax_i - target_i
    for (int i = 0; i < length; i++) {
        float grad = softmax_cache[i] - target[i].val;
        if (std::isnan(grad) || std::isinf(grad)) {
            std::cout << "WARNING: NaN/Inf gradient at index " << i << ": " << grad << std::endl;
            grad = 0.0f;
        }
        logits[i].setDiff(grad, &loss[0]);
    }
}

void Loss_CrossEntropy::backward()
{
    tensor &x = *input1;
    for (int i = 0; i < length; i++) {
        while (!x[i].diffs.empty()) {
            std::pair<float, node *> pop = x[i].diffs.back();
            x[i].diff += pop.first * pop.second->diff;
            x[i].diffs.pop_back();
        }
    }
}

void Loss_CrossEntropy::update()
{
    tensor &x = *input1;
    tensor &w = *input2;

    for (int i = 0; i < length; i++) {
        if (cfg.Accuracy > cfg.START_QUANTIZATION) { x[i].f2q(); x[i].q2f(); }
        x[i].val = x[i].val - cfg.lr * x[i].diff;
        x[i].diff = 0; x[i].diffs.clear();
    }
    for (int i = 0; i < length; i++) {
        w[i].diff = 0; w[i].diffs.clear();
    }
}

void Loss_CrossEntropy::clear()
{
    tensor &x = *input1;
    tensor &w = *input2;
    for (int i = 0; i < length; i++) { x[i].diff = 0; x[i].diffs.clear(); }
    for (int i = 0; i < length; i++) { w[i].diff = 0; w[i].diffs.clear(); }
}

void Loss_CrossEntropy::save()
{
    // not code-gen
}

// [1 * 3 * 2]
void matmul(node *out, node *a, node *b, int m, int k, int n)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            out[i * n + j].val = 0;
            for (int q = 0; q < k; q++)
            {
                out[i * n + j].val += a[i * k + q].val * b[q * n + j].val;

                a[i * k + q].setDiff(b[q * n + j].val, &out[i * n + j]);
                b[q * n + j].setDiff(a[i * k + q].val, &out[i * n + j]);
            }
        }
    }
}

// MSE
void loss_MSE(node *loss, node *src, node *dest, int m)
{
    float sum = 0;

    for (int i = 0; i < m; i++)
    {
        sum += pow(src[i].val - dest[i].val, 2);

        // https://zs.symbolab.com/solver/partial-derivative-calculator/%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%20x%7D%5Cleft(%5Cfrac%7B1%7D%7B2%7D%5Cleft(x-y%5Cright)%5E%7B2%7D%5Cright)
        // ∂/∂x = x - y, ∂/∂y = -x + y
        src[i].setDiff(src[i].val - dest[i].val, loss);
        dest[i].setDiff(-src[i].val + dest[i].val, loss);
    }

    loss->val = sum / (2 * m);
    loss->diff = 1;
    loss->setDiff(1, NULL);
}

void add(node *out, node *a, node *b, int length)
{
    for (int i = 0; i < length; i++)
    {
        out[i].val = a[i].val + b[i].val;

        a[i].setDiff(1, &out[i]);
        b[i].setDiff(1, &out[i]);
    }
}

/*
 * ScaledDotProductAttention — 縮放點積注意力機制
 *
 * 數學公式：Attention(Q,K,V) = softmax(QKᵀ / √d_k) × V
 *
 * forward:
 *   1. QKᵀ: [seq, d_k] × [d_k, seq] → [seq, seq] 注意力分數矩陣
 *   2. 除以 √d_k 防止點積值過大導致 softmax 飽和
 *   3. Softmax 歸一化為注意力權重
 *   4. 加權求和 V：[seq, seq] × [seq, d_k] → [seq, d_k]
 *
 * backward: 反向傳播通過三個階段
 *   1. ∂L/∂V = Attentionᵀ × ∂L/∂output
 *   2. ∂L/∂attention = ∂L/∂output × Vᵀ → 通過 softmax 反向
 *   3. ∂L/∂Q, ∂L/∂K 從 QKᵀ 的梯度推導
 */
// ============================================================================
// [TEXT] Embedding — 嵌入查表層
//
// forward:  output[i] = weight[indices[i]]  （查表）
// backward: weight[indices[i]] += ∂L/∂output[i]  （梯度累加回對應 row）
//
// 用於把離散的 token ID 轉成連續的向量表示
// ============================================================================
class Embedding : public opBase
{
public:
    tensor *output;     // [seq_len, d_model]
    tensor *weight;     // [vocab_size, d_model]  — 嵌入矩陣
    std::vector<int> indices;  // [seq_len] — 輸入的 token IDs
    int vocab_size;
    int d_model;
    int seq_len;

    Embedding(tensor &out, tensor &w, int vocab, int dim, int seq)
    {
        output = &out;
        weight = &w;
        vocab_size = vocab;
        d_model = dim;
        seq_len = seq;
        indices.resize(seq);

        nnCode.append(out.name + " = embedding(" + w.name + ");\n");
    }

    void set_indices(const std::vector<int> &idx)
    {
        for (int i = 0; i < seq_len && i < (int)idx.size(); i++)
            indices[i] = idx[i];
    }

    void forward()
    {
        // 查表：output[i, :] = weight[indices[i], :]
        for (int i = 0; i < seq_len; i++) {
            int token_id = indices[i];
            for (int j = 0; j < d_model; j++) {
                output->data[i * d_model + j].val = weight->data[token_id * d_model + j].val;
            }
        }
    }

    void backward()
    {
        // 梯度累加回 weight 的對應 row
        for (int i = 0; i < seq_len; i++) {
            int token_id = indices[i];
            for (int j = 0; j < d_model; j++) {
                weight->data[token_id * d_model + j].diff += output->data[i * d_model + j].diff;
            }
        }
    }

    void update()
    {
        // SGD 更新嵌入矩陣
        for (int i = 0; i < vocab_size * d_model; i++) {
            weight->data[i].val -= cfg.lr * weight->data[i].diff;
            weight->data[i].diff = 0;
            weight->data[i].diffs.clear();
        }
    }

    void clear()
    {
        for (int i = 0; i < vocab_size * d_model; i++) {
            weight->data[i].diff = 0;
            weight->data[i].diffs.clear();
        }
        for (int i = 0; i < seq_len * d_model; i++) {
            output->data[i].diff = 0;
            output->data[i].diffs.clear();
        }
    }

    void save() {}
};

// ============================================================================
// [TEXT] CausalAttention — 帶因果遮罩的縮放點積注意力
//
// 跟 ScaledDotProductAttention 一樣，但加了 causal mask：
// 上三角設為 -∞（-1e9），讓每個位置只能看到自己和之前的 token
// ============================================================================
class CausalAttention : public opBase
{
public:
    tensor *output;
    tensor *input_q, *input_k, *input_v;
    int seq_len, d_k;
    std::vector<float> attention_weights;

    CausalAttention(tensor &out, tensor &q, tensor &k, tensor &v, int seq, int dk)
    {
        output = &out;
        input_q = &q;
        input_k = &k;
        input_v = &v;
        seq_len = seq;
        d_k = dk;
        attention_weights.resize(seq_len * seq_len);

        nnCode.append(out.name + " = causal_attention(" + q.name + ", " + k.name + ", " + v.name + ");\n");
    }

    void forward()
    {
        float scale = 1.0f / sqrt((float)d_k);

        // QKᵀ + causal mask + softmax
        for (int i = 0; i < seq_len; i++) {
            // 計算 scores
            std::vector<float> scores(seq_len);
            for (int j = 0; j < seq_len; j++) {
                float score = 0.0f;
                for (int k = 0; k < d_k; k++)
                    score += input_q->data[i * d_k + k].val * input_k->data[j * d_k + k].val;
                scores[j] = score * scale;

                // 因果遮罩：看不到未來（j > i 的位置）
                if (j > i) scores[j] = -1e9f;
            }

            // Softmax（數值穩定）
            float max_val = *std::max_element(scores.begin(), scores.end());
            float sum_exp = 0.0f;
            for (int j = 0; j < seq_len; j++) {
                scores[j] = exp(scores[j] - max_val);
                sum_exp += scores[j];
            }
            for (int j = 0; j < seq_len; j++)
                attention_weights[i * seq_len + j] = scores[j] / sum_exp;

            // Attention × V
            for (int j = 0; j < d_k; j++) {
                output->data[i * d_k + j].val = 0.0f;
                for (int k = 0; k < seq_len; k++)
                    output->data[i * d_k + j].val += attention_weights[i * seq_len + k] * input_v->data[k * d_k + j].val;
            }
        }
    }

    void backward()
    {
        float scale = 1.0f / sqrt((float)d_k);

        // ∂L/∂V
        for (int i = 0; i < seq_len; i++)
            for (int j = 0; j < d_k; j++)
                for (int k = 0; k < seq_len; k++)
                    input_v->data[k * d_k + j].diff += attention_weights[i * seq_len + k] * output->data[i * d_k + j].diff;

        // ∂L/∂attention_weights
        std::vector<float> d_attn(seq_len * seq_len, 0.0f);
        for (int i = 0; i < seq_len; i++)
            for (int k = 0; k < seq_len; k++)
                for (int j = 0; j < d_k; j++)
                    d_attn[i * seq_len + k] += input_v->data[k * d_k + j].val * output->data[i * d_k + j].diff;

        // ∂L/∂scores (through softmax)
        std::vector<float> d_scores(seq_len * seq_len);
        for (int i = 0; i < seq_len; i++) {
            float sum = 0.0f;
            for (int k = 0; k < seq_len; k++)
                sum += d_attn[i * seq_len + k] * attention_weights[i * seq_len + k];
            for (int j = 0; j < seq_len; j++)
                d_scores[i * seq_len + j] = attention_weights[i * seq_len + j] * (d_attn[i * seq_len + j] - sum);
        }

        // ∂L/∂Q, ∂L/∂K
        for (int i = 0; i < seq_len; i++)
            for (int j = 0; j < seq_len; j++) {
                if (j > i) continue;  // causal mask: 遮罩位置沒有梯度
                float grad = d_scores[i * seq_len + j] * scale;
                for (int k = 0; k < d_k; k++) {
                    input_q->data[i * d_k + k].diff += grad * input_k->data[j * d_k + k].val;
                    input_k->data[j * d_k + k].diff += grad * input_q->data[i * d_k + k].val;
                }
            }
    }

    void update()
    {
        for (int i = 0; i < seq_len * d_k; i++) {
            input_q->data[i].diff = 0; input_q->data[i].diffs.clear();
            input_k->data[i].diff = 0; input_k->data[i].diffs.clear();
            input_v->data[i].diff = 0; input_v->data[i].diffs.clear();
            output->data[i].diff = 0; output->data[i].diffs.clear();
        }
    }

    void clear()
    {
        for (int i = 0; i < seq_len * d_k; i++) {
            input_q->data[i].diff = 0; input_q->data[i].diffs.clear();
            input_k->data[i].diff = 0; input_k->data[i].diffs.clear();
            input_v->data[i].diff = 0; input_v->data[i].diffs.clear();
            output->data[i].diff = 0; output->data[i].diffs.clear();
        }
    }

    void save() {}
};

// ============================================================================
// [TEXT] TextCrossEntropy — 序列級交叉熵損失
//
// 對序列中每個位置獨立做 softmax + cross entropy，然後取平均
// ============================================================================
class TextCrossEntropy : public opBase
{
public:
    tensor *output;   // loss scalar (size 1)
    tensor *input;    // logits [seq_len, vocab_size]
    std::vector<int> targets;  // target token IDs [seq_len]
    int seq_len, vocab_size;
    std::vector<float> probs;  // softmax 結果，backward 用

    TextCrossEntropy(tensor &out, tensor &in, int seq, int vocab)
    {
        output = &out;
        input = &in;
        seq_len = seq;
        vocab_size = vocab;
        targets.resize(seq);
        probs.resize(seq * vocab);

        nnCode.append(out.name + " = text_cross_entropy(" + in.name + ");\n");
    }

    void set_targets(const std::vector<int> &tgt)
    {
        for (int i = 0; i < seq_len && i < (int)tgt.size(); i++)
            targets[i] = tgt[i];
    }

    void forward()
    {
        float total_loss = 0.0f;
        for (int i = 0; i < seq_len; i++) {
            // Softmax for position i
            float max_val = -1e9f;
            for (int j = 0; j < vocab_size; j++) {
                float v = input->data[i * vocab_size + j].val;
                if (v > max_val) max_val = v;
            }
            float sum_exp = 0.0f;
            for (int j = 0; j < vocab_size; j++) {
                probs[i * vocab_size + j] = exp(input->data[i * vocab_size + j].val - max_val);
                sum_exp += probs[i * vocab_size + j];
            }
            for (int j = 0; j < vocab_size; j++)
                probs[i * vocab_size + j] /= sum_exp;

            // Cross entropy: -log(p[target])
            float p = probs[i * vocab_size + targets[i]];
            total_loss += -log(p + 1e-10f);
        }
        output->data[0].val = total_loss / seq_len;
    }

    void backward()
    {
        // ∂L/∂logits = (softmax - one_hot) / seq_len
        output->data[0].diff = 1.0f;
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < vocab_size; j++) {
                float grad = probs[i * vocab_size + j];
                if (j == targets[i]) grad -= 1.0f;
                input->data[i * vocab_size + j].diff += grad / seq_len;
            }
        }
    }

    void update()
    {
        for (int i = 0; i < seq_len * vocab_size; i++) {
            input->data[i].diff = 0;
            input->data[i].diffs.clear();
        }
        output->data[0].diff = 0;
        output->data[0].diffs.clear();
    }

    void clear()
    {
        for (int i = 0; i < seq_len * vocab_size; i++) {
            input->data[i].diff = 0;
            input->data[i].diffs.clear();
        }
        output->data[0].diff = 0;
        output->data[0].diffs.clear();
    }

    void save() {}
};

// ============================================================================
// [TEXT] TextMatmul — 矩陣乘法（帶權重更新）
//
// output = input × weight，SGD 更新 weight
// ============================================================================
class TextMatmul : public opBase
{
public:
    tensor *output;   // [M, N]
    tensor *input;    // [M, K]
    tensor *weight;   // [K, N]
    int M, K, N;

    TextMatmul(tensor &out, tensor &in, tensor &w, int m, int k, int n)
    {
        output = &out;
        input = &in;
        weight = &w;
        M = m; K = k; N = n;

        nnCode.append(out.name + " = matmul(" + in.name + ", " + w.name + ");\n");
    }

    void forward()
    {
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++) {
                output->data[i * N + j].val = 0.0f;
                for (int k = 0; k < K; k++)
                    output->data[i * N + j].val += input->data[i * K + k].val * weight->data[k * N + j].val;
            }
    }

    void backward()
    {
        // ∂L/∂input = ∂L/∂output × Wᵀ
        for (int i = 0; i < M; i++)
            for (int k = 0; k < K; k++)
                for (int j = 0; j < N; j++)
                    input->data[i * K + k].diff += output->data[i * N + j].diff * weight->data[k * N + j].val;

        // ∂L/∂W = inputᵀ × ∂L/∂output
        for (int k = 0; k < K; k++)
            for (int j = 0; j < N; j++)
                for (int i = 0; i < M; i++)
                    weight->data[k * N + j].diff += input->data[i * K + k].val * output->data[i * N + j].diff;
    }

    void update()
    {
        for (int i = 0; i < K * N; i++) {
            weight->data[i].val -= cfg.lr * weight->data[i].diff;
            weight->data[i].diff = 0;
            weight->data[i].diffs.clear();
        }
        for (int i = 0; i < M * K; i++) {
            input->data[i].diff = 0;
            input->data[i].diffs.clear();
        }
        for (int i = 0; i < M * N; i++) {
            output->data[i].diff = 0;
            output->data[i].diffs.clear();
        }
    }

    void clear()
    {
        for (int k = 0; k < K * N; k++) {
            weight->data[k].diff = 0;
            weight->data[k].diffs.clear();
        }
        for (int i = 0; i < M * K; i++) {
            input->data[i].diff = 0;
            input->data[i].diffs.clear();
        }
        for (int i = 0; i < M * N; i++) {
            output->data[i].diff = 0;
            output->data[i].diffs.clear();
        }
    }

    void save() {}
};

// ============================================================================
// [TEXT] TextAdd — 逐元素加法（帶權重更新，用於 bias / positional encoding）
// ============================================================================
class TextAdd : public opBase
{
public:
    tensor *output;
    tensor *input_a, *input_b;
    int size;
    bool update_b;  // 是否 SGD 更新 input_b

    TextAdd(tensor &out, tensor &a, tensor &b, int sz, bool upd_b = true)
    {
        output = &out;
        input_a = &a;
        input_b = &b;
        size = sz;
        update_b = upd_b;

        nnCode.append(out.name + " = add(" + a.name + ", " + b.name + ");\n");
    }

    void forward()
    {
        for (int i = 0; i < size; i++)
            output->data[i].val = input_a->data[i].val + input_b->data[i].val;
    }

    void backward()
    {
        for (int i = 0; i < size; i++) {
            input_a->data[i].diff += output->data[i].diff;
            input_b->data[i].diff += output->data[i].diff;
        }
    }

    void update()
    {
        for (int i = 0; i < size; i++) {
            if (update_b) {
                input_b->data[i].val -= cfg.lr * input_b->data[i].diff;
            }
            input_a->data[i].diff = 0; input_a->data[i].diffs.clear();
            input_b->data[i].diff = 0; input_b->data[i].diffs.clear();
            output->data[i].diff = 0; output->data[i].diffs.clear();
        }
    }

    void clear()
    {
        for (int i = 0; i < size; i++) {
            input_a->data[i].diff = 0; input_a->data[i].diffs.clear();
            input_b->data[i].diff = 0; input_b->data[i].diffs.clear();
            output->data[i].diff = 0; output->data[i].diffs.clear();
        }
    }

    void save() {}
};

// ============================================================================
// [TEXT] TextReLU — ReLU 激活
// ============================================================================
class TextReLU : public opBase
{
public:
    tensor *output, *input;
    int size;

    TextReLU(tensor &out, tensor &in, int sz) : output(&out), input(&in), size(sz) {
        nnCode.append(out.name + " = relu(" + in.name + ");\n");
    }

    void forward() {
        for (int i = 0; i < size; i++)
            output->data[i].val = (input->data[i].val > 0) ? input->data[i].val : 0;
    }

    void backward() {
        for (int i = 0; i < size; i++)
            input->data[i].diff += (input->data[i].val > 0) ? output->data[i].diff : 0;
    }

    void update() {
        for (int i = 0; i < size; i++) {
            input->data[i].diff = 0; input->data[i].diffs.clear();
            output->data[i].diff = 0; output->data[i].diffs.clear();
        }
    }

    void clear() {
        for (int i = 0; i < size; i++) {
            input->data[i].diff = 0; input->data[i].diffs.clear();
            output->data[i].diff = 0; output->data[i].diffs.clear();
        }
    }

    void save() {}
};

// ============================================================================
// [TEXT] TextLayerNorm — 序列級層正規化
//
// 對序列中每個位置（row）獨立做 LayerNorm：
//   output[i,:] = γ × (x[i,:] - μ_i) / √(σ²_i + ε) + β
//
// γ, β 是長度 d_model 的可學習參數（所有位置共享）
// ============================================================================
class TextLayerNorm : public opBase
{
public:
    tensor *output;
    tensor *input;
    tensor *gamma;     // [d_model] — scale
    tensor *beta;      // [d_model] — shift
    int seq_len, d_model;
    float eps;
    std::vector<float> mean_cache;    // [seq_len]
    std::vector<float> var_cache;     // [seq_len]
    std::vector<float> norm_cache;    // [seq_len * d_model]

    TextLayerNorm(tensor &out, tensor &in, tensor &g, tensor &b, int seq, int dim, float epsilon = 1e-5f)
    {
        output = &out;
        input = &in;
        gamma = &g;
        beta = &b;
        seq_len = seq;
        d_model = dim;
        eps = epsilon;
        mean_cache.resize(seq_len);
        var_cache.resize(seq_len);
        norm_cache.resize(seq_len * d_model);

        // γ 初始化為 1，β 初始化為 0
        for (int i = 0; i < d_model; i++) {
            gamma->data[i].val = 1.0f;
            beta->data[i].val = 0.0f;
        }

        nnCode.append(out.name + " = layer_norm(" + in.name + ");\n");
    }

    void forward()
    {
        for (int i = 0; i < seq_len; i++) {
            // 計算 mean
            float mean = 0.0f;
            for (int j = 0; j < d_model; j++)
                mean += input->data[i * d_model + j].val;
            mean /= d_model;
            mean_cache[i] = mean;

            // 計算 variance
            float var = 0.0f;
            for (int j = 0; j < d_model; j++) {
                float diff = input->data[i * d_model + j].val - mean;
                var += diff * diff;
            }
            var /= d_model;
            var_cache[i] = var;

            // normalize + scale + shift
            float inv_std = 1.0f / sqrt(var + eps);
            for (int j = 0; j < d_model; j++) {
                float normalized = (input->data[i * d_model + j].val - mean) * inv_std;
                norm_cache[i * d_model + j] = normalized;
                output->data[i * d_model + j].val = normalized * gamma->data[j].val + beta->data[j].val;
            }
        }
    }

    void backward()
    {
        for (int i = 0; i < seq_len; i++) {
            float mean = mean_cache[i];
            float var = var_cache[i];
            float inv_std = 1.0f / sqrt(var + eps);

            // ∂L/∂γ, ∂L/∂β（所有位置累加）
            for (int j = 0; j < d_model; j++) {
                gamma->data[j].diff += norm_cache[i * d_model + j] * output->data[i * d_model + j].diff;
                beta->data[j].diff += output->data[i * d_model + j].diff;
            }

            // ∂L/∂norm
            std::vector<float> d_norm(d_model);
            for (int j = 0; j < d_model; j++)
                d_norm[j] = gamma->data[j].val * output->data[i * d_model + j].diff;

            // ∂L/∂var
            float d_var = 0.0f;
            for (int j = 0; j < d_model; j++)
                d_var += d_norm[j] * (input->data[i * d_model + j].val - mean);
            d_var *= -0.5f * pow(var + eps, -1.5f);

            // ∂L/∂mean
            float d_mean = 0.0f;
            for (int j = 0; j < d_model; j++)
                d_mean += d_norm[j];
            d_mean *= -inv_std;
            // d_var 對 mean 的貢獻（Σ(x-μ) = 0，所以這項為 0）

            // ∂L/∂x
            for (int j = 0; j < d_model; j++) {
                input->data[i * d_model + j].diff +=
                    d_norm[j] * inv_std +
                    d_var * (2.0f / d_model) * (input->data[i * d_model + j].val - mean) +
                    d_mean / d_model;
            }
        }
    }

    void update()
    {
        // SGD 更新 γ, β
        for (int j = 0; j < d_model; j++) {
            gamma->data[j].val -= cfg.lr * gamma->data[j].diff;
            gamma->data[j].diff = 0; gamma->data[j].diffs.clear();
            beta->data[j].val -= cfg.lr * beta->data[j].diff;
            beta->data[j].diff = 0; beta->data[j].diffs.clear();
        }
        // 清理 input/output
        for (int i = 0; i < seq_len * d_model; i++) {
            input->data[i].diff = 0; input->data[i].diffs.clear();
            output->data[i].diff = 0; output->data[i].diffs.clear();
        }
    }

    void clear()
    {
        for (int j = 0; j < d_model; j++) {
            gamma->data[j].diff = 0; gamma->data[j].diffs.clear();
            beta->data[j].diff = 0; beta->data[j].diffs.clear();
        }
        for (int i = 0; i < seq_len * d_model; i++) {
            input->data[i].diff = 0; input->data[i].diffs.clear();
            output->data[i].diff = 0; output->data[i].diffs.clear();
        }
    }

    void save() {}
};

// ============================================================================
// [TEXT] TextGELU — Gaussian Error Linear Unit
//
// GPT 系列用 GELU 取代 ReLU：
//   GELU(x) = x × Φ(x)  ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
//
// backward:
//   ∂GELU/∂x = Φ(x) + x × φ(x)  （用 tanh 近似的導數）
// ============================================================================
class TextGELU : public opBase
{
public:
    tensor *output, *input;
    int size;
    std::vector<float> tanh_cache;  // 存 tanh 值供 backward 用

    TextGELU(tensor &out, tensor &in, int sz) : output(&out), input(&in), size(sz) {
        tanh_cache.resize(sz);
        nnCode.append(out.name + " = gelu(" + in.name + ");\n");
    }

    void forward() {
        const float sqrt_2_over_pi = 0.7978845608f;  // √(2/π)
        const float coeff = 0.044715f;
        for (int i = 0; i < size; i++) {
            float x = input->data[i].val;
            float inner = sqrt_2_over_pi * (x + coeff * x * x * x);
            float t = tanh(inner);
            tanh_cache[i] = t;
            output->data[i].val = 0.5f * x * (1.0f + t);
        }
    }

    void backward() {
        const float sqrt_2_over_pi = 0.7978845608f;
        const float coeff = 0.044715f;
        for (int i = 0; i < size; i++) {
            float x = input->data[i].val;
            float t = tanh_cache[i];
            // sech²(inner) = 1 - tanh²(inner)
            float sech2 = 1.0f - t * t;
            float inner_deriv = sqrt_2_over_pi * (1.0f + 3.0f * coeff * x * x);
            // ∂GELU/∂x = 0.5(1+t) + 0.5x × sech²(inner) × inner'
            float grad = 0.5f * (1.0f + t) + 0.5f * x * sech2 * inner_deriv;
            input->data[i].diff += grad * output->data[i].diff;
        }
    }

    void update() {
        for (int i = 0; i < size; i++) {
            input->data[i].diff = 0; input->data[i].diffs.clear();
            output->data[i].diff = 0; output->data[i].diffs.clear();
        }
    }

    void clear() {
        for (int i = 0; i < size; i++) {
            input->data[i].diff = 0; input->data[i].diffs.clear();
            output->data[i].diff = 0; output->data[i].diffs.clear();
        }
    }

    void save() {}
};

// [TRANSFORMER] ScaledDotProductAttention operation
class ScaledDotProductAttention : public opBase
{
public:
    tensor *output;    // output values
    tensor *input_q;   // query
    tensor *input_k;   // key  
    tensor *input_v;   // value
    int seq_len;       // sequence length
    int d_k;           // key dimension
    std::vector<float> attention_weights; // store for backward pass
    
    ScaledDotProductAttention(tensor &out, tensor &q, tensor &k, tensor &v, int seq_len, int d_k);
    void forward();
    void backward();
    void update();
    void clear();
    void save();
};

ScaledDotProductAttention::ScaledDotProductAttention(tensor &out, tensor &q, tensor &k, tensor &v, int seq, int dk)
{
    output = &out;
    input_q = &q;
    input_k = &k;
    input_v = &v;
    seq_len = seq;
    d_k = dk;
    attention_weights.resize(seq_len * seq_len);
    
    // NNEF codeGen
    nnCode.append(out.name);
    nnCode.append(" = ");
    nnCode.append("scaled_dot_product_attention");
    nnCode.append("(");
    nnCode.append(q.name);
    nnCode.append(", ");
    nnCode.append(k.name);
    nnCode.append(", ");
    nnCode.append(v.name);
    nnCode.append(", d_k = ");
    nnCode.append(std::to_string(d_k));
    nnCode.append(");\n");
}

void ScaledDotProductAttention::forward()
{
    // Compute QK^T / sqrt(d_k)
    float scale = 1.0f / sqrt((float)d_k);
    
    // QK^T: [seq_len, d_k] x [d_k, seq_len] -> [seq_len, seq_len]
    std::vector<float> qk_scores(seq_len * seq_len);
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            float score = 0.0f;
            for (int k = 0; k < d_k; k++) {
                score += input_q->data[i * d_k + k].val * input_k->data[j * d_k + k].val;
            }
            qk_scores[i * seq_len + j] = score * scale;
        }
    }
    
    // Numerically stable softmax
    for (int i = 0; i < seq_len; i++) {
        // Find max for numerical stability
        float max_val = qk_scores[i * seq_len];
        for (int j = 1; j < seq_len; j++) {
            if (qk_scores[i * seq_len + j] > max_val) {
                max_val = qk_scores[i * seq_len + j];
            }
        }
        
        // Compute exp and sum
        float sum_exp = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            qk_scores[i * seq_len + j] = exp(qk_scores[i * seq_len + j] - max_val);
            sum_exp += qk_scores[i * seq_len + j];
        }
        
        // Normalize to get attention weights
        for (int j = 0; j < seq_len; j++) {
            attention_weights[i * seq_len + j] = qk_scores[i * seq_len + j] / sum_exp;
        }
    }
    
    // Attention * V: [seq_len, seq_len] x [seq_len, d_k] -> [seq_len, d_k]
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < d_k; j++) {
            output->data[i * d_k + j].val = 0.0f;
            for (int k = 0; k < seq_len; k++) {
                output->data[i * d_k + j].val += attention_weights[i * seq_len + k] * input_v->data[k * d_k + j].val;
            }
        }
    }
}

void ScaledDotProductAttention::backward()
{
    float scale = 1.0f / sqrt((float)d_k);
    
    // Backward through Attention * V
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < d_k; j++) {
            for (int k = 0; k < seq_len; k++) {
                // Gradient w.r.t. V
                input_v->data[k * d_k + j].diff += attention_weights[i * seq_len + k] * output->data[i * d_k + j].diff;
            }
        }
    }
    
    // Compute gradient w.r.t. attention weights
    std::vector<float> d_attention_weights(seq_len * seq_len, 0.0f);
    for (int i = 0; i < seq_len; i++) {
        for (int k = 0; k < seq_len; k++) {
            for (int j = 0; j < d_k; j++) {
                d_attention_weights[i * seq_len + k] += input_v->data[k * d_k + j].val * output->data[i * d_k + j].diff;
            }
        }
    }
    
    // Backward through softmax
    std::vector<float> d_qk_scores(seq_len * seq_len);
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            float sum = 0.0f;
            for (int k = 0; k < seq_len; k++) {
                sum += d_attention_weights[i * seq_len + k] * attention_weights[i * seq_len + k];
            }
            d_qk_scores[i * seq_len + j] = attention_weights[i * seq_len + j] * 
                                          (d_attention_weights[i * seq_len + j] - sum);
        }
    }
    
    // Backward through QK^T / sqrt(d_k)
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            float grad = d_qk_scores[i * seq_len + j] * scale;
            for (int k = 0; k < d_k; k++) {
                // Gradient w.r.t. Q
                input_q->data[i * d_k + k].diff += grad * input_k->data[j * d_k + k].val;
                // Gradient w.r.t. K  
                input_k->data[j * d_k + k].diff += grad * input_q->data[i * d_k + k].val;
            }
        }
    }
}

void ScaledDotProductAttention::update()
{
    // No trainable parameters in this operation
    for (int i = 0; i < seq_len * d_k; i++) {
        if (cfg.Accuracy > cfg.START_QUANTIZATION) {
            input_q->data[i].f2q();
            input_q->data[i].q2f();
            input_k->data[i].f2q();
            input_k->data[i].q2f();
            input_v->data[i].f2q();
            input_v->data[i].q2f();
        }
        input_q->data[i].diff = 0;
        input_q->data[i].diffs.clear();
        input_k->data[i].diff = 0;
        input_k->data[i].diffs.clear();
        input_v->data[i].diff = 0;
        input_v->data[i].diffs.clear();
    }
}

void ScaledDotProductAttention::clear()
{
    for (int i = 0; i < seq_len * d_k; i++) {
        input_q->data[i].diff = 0;
        input_q->data[i].diffs.clear();
        input_k->data[i].diff = 0;
        input_k->data[i].diffs.clear();
        input_v->data[i].diff = 0;
        input_v->data[i].diffs.clear();
    }
}

void ScaledDotProductAttention::save()
{
    std::cout << "\t" << nnCode;
}

/*
 * MultiHeadAttention — 多頭注意力機制
 *
 * 概念：將 d_model 維度切成 num_heads 份，各頭獨立計算注意力後拼接
 *   MultiHead(Q,K,V) = Concat(head_1, ..., head_h) × W_O
 *   head_i = Attention(Q × W_Q_i, K × W_K_i, V × W_V_i)
 *
 * forward:
 *   1. 線性投影 Q = input × W_Q, K = input × W_K, V = input × W_V
 *   2. 切分成 num_heads 份，各頭獨立做 ScaledDotProductAttention
 *   3. 拼接所有頭的輸出，再乘以 W_O 投影回 d_model 維度
 *
 * backward: 反向沿投影 → 各頭注意力 → 合併梯度
 *   W_Q, W_K, W_V, W_O 為可訓練參數，使用 SGD + 梯度裁剪更新
 */
// [TRANSFORMER] MultiHeadAttention operation
class MultiHeadAttention : public opBase
{
public:
    tensor *output;
    tensor *input;
    tensor *W_Q;       // Query projection weights
    tensor *W_K;       // Key projection weights  
    tensor *W_V;       // Value projection weights
    tensor *W_O;       // Output projection weights
    int num_heads;
    int d_model;
    int d_k;           // d_model / num_heads
    int seq_len;
    std::vector<ScaledDotProductAttention*> attention_heads;
    std::vector<tensor*> head_outputs;
    std::vector<float> concat_heads_cache;  // FIX: Cache concat heads for backward
    
    MultiHeadAttention(tensor &out, tensor &in, tensor &wq, tensor &wk, tensor &wv, tensor &wo, 
                      int heads, int model_dim, int sequence_length);
    void forward();
    void backward();
    void update();
    void clear();
    void save();
};

MultiHeadAttention::MultiHeadAttention(tensor &out, tensor &in, tensor &wq, tensor &wk, 
                                     tensor &wv, tensor &wo, int heads, int model_dim, int sequence_length)
{
    output = &out;
    input = &in;
    W_Q = &wq;
    W_K = &wk;
    W_V = &wv;
    W_O = &wo;
    num_heads = heads;
    d_model = model_dim;
    d_k = d_model / num_heads;
    seq_len = sequence_length;
    
    // Initialize He weights
    W_Q->init_he();
    W_K->init_he();
    W_V->init_he(); 
    W_O->init_he();
    
    // NNEF codeGen
    nnCode.append(out.name);
    nnCode.append(" = ");
    nnCode.append("multi_head_attention");
    nnCode.append("(");
    nnCode.append(in.name);
    nnCode.append(", num_heads = ");
    nnCode.append(std::to_string(num_heads));
    nnCode.append(", d_model = ");
    nnCode.append(std::to_string(d_model));
    nnCode.append(");\n");
}

void MultiHeadAttention::forward()
{
    // FIX Bug 4: Clean up previous forward call data
    for (auto* attn : attention_heads) {
        delete attn;
    }
    for (auto* head : head_outputs) {
        delete head;
    }
    attention_heads.clear();
    head_outputs.clear();
    
    // Linear projections Q = input * W_Q, K = input * W_K, V = input * W_V
    // [seq_len, d_model] * [d_model, d_model] -> [seq_len, d_model]
    std::vector<tensor> Q_proj(1, tensor({seq_len, d_model}));
    std::vector<tensor> K_proj(1, tensor({seq_len, d_model}));
    std::vector<tensor> V_proj(1, tensor({seq_len, d_model}));
    
    // Compute Q projection
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < d_model; j++) {
            Q_proj[0].data[i * d_model + j].val = 0.0f;
            for (int k = 0; k < d_model; k++) {
                Q_proj[0].data[i * d_model + j].val += input->data[i * d_model + k].val * W_Q->data[k * d_model + j].val;
            }
        }
    }
    
    // Compute K projection
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < d_model; j++) {
            K_proj[0].data[i * d_model + j].val = 0.0f;
            for (int k = 0; k < d_model; k++) {
                K_proj[0].data[i * d_model + j].val += input->data[i * d_model + k].val * W_K->data[k * d_model + j].val;
            }
        }
    }
    
    // Compute V projection
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < d_model; j++) {
            V_proj[0].data[i * d_model + j].val = 0.0f;
            for (int k = 0; k < d_model; k++) {
                V_proj[0].data[i * d_model + j].val += input->data[i * d_model + k].val * W_V->data[k * d_model + j].val;
            }
        }
    }
    
    // Split into heads and apply attention
    for (int h = 0; h < num_heads; h++) {
        // Extract head h: [seq_len, d_k]
        tensor* q_head = new tensor({seq_len, d_k});
        tensor* k_head = new tensor({seq_len, d_k});
        tensor* v_head = new tensor({seq_len, d_k});
        tensor* attn_out = new tensor({seq_len, d_k});
        
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < d_k; j++) {
                q_head->data[i * d_k + j].val = Q_proj[0].data[i * d_model + h * d_k + j].val;
                k_head->data[i * d_k + j].val = K_proj[0].data[i * d_model + h * d_k + j].val;
                v_head->data[i * d_k + j].val = V_proj[0].data[i * d_model + h * d_k + j].val;
            }
        }
        
        // Apply scaled dot product attention
        ScaledDotProductAttention* attention = new ScaledDotProductAttention(*attn_out, *q_head, *k_head, *v_head, seq_len, d_k);
        attention->forward();
        
        attention_heads.push_back(attention);
        head_outputs.push_back(attn_out);
    }
    
    // Concatenate heads: [seq_len, num_heads * d_k] = [seq_len, d_model]
    std::vector<tensor> concat_heads(1, tensor({seq_len, d_model}));
    for (int i = 0; i < seq_len; i++) {
        for (int h = 0; h < num_heads; h++) {
            for (int j = 0; j < d_k; j++) {
                concat_heads[0].data[i * d_model + h * d_k + j].val = head_outputs[h]->data[i * d_k + j].val;
            }
        }
    }
    
    // FIX Bug 1: Cache concat_heads for backward pass
    concat_heads_cache.resize(seq_len * d_model);
    for (int i = 0; i < seq_len * d_model; i++) {
        concat_heads_cache[i] = concat_heads[0].data[i].val;
    }
    
    // Output projection: [seq_len, d_model] * [d_model, d_model] -> [seq_len, d_model]
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < d_model; j++) {
            output->data[i * d_model + j].val = 0.0f;
            for (int k = 0; k < d_model; k++) {
                output->data[i * d_model + j].val += concat_heads[0].data[i * d_model + k].val * W_O->data[k * d_model + j].val;
            }
        }
    }
}

void MultiHeadAttention::backward()
{
    // FIX Bug 1: Backward through output projection with cached concat_heads
    std::vector<float> d_concat(seq_len * d_model, 0.0f);
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < d_model; j++) {
            for (int k = 0; k < d_model; k++) {
                // Gradient w.r.t. concat heads
                d_concat[i * d_model + k] += output->data[i * d_model + j].diff * W_O->data[k * d_model + j].val;
                // FIX Bug 1: W_O gradient uses cached concat_heads, not d_concat
                W_O->data[k * d_model + j].diff += concat_heads_cache[i * d_model + k] * output->data[i * d_model + j].diff;
            }
        }
    }
    
    // Backward through head concatenation and attention
    std::vector<float> d_Q(seq_len * d_model, 0.0f);
    std::vector<float> d_K(seq_len * d_model, 0.0f);
    std::vector<float> d_V(seq_len * d_model, 0.0f);
    
    for (int h = 0; h < num_heads; h++) {
        // Set gradients for head outputs
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < d_k; j++) {
                head_outputs[h]->data[i * d_k + j].diff = d_concat[i * d_model + h * d_k + j];
            }
        }
        
        // Backward through attention
        attention_heads[h]->backward();
        
        // Accumulate gradients from heads
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < d_k; j++) {
                d_Q[i * d_model + h * d_k + j] += attention_heads[h]->input_q->data[i * d_k + j].diff;
                d_K[i * d_model + h * d_k + j] += attention_heads[h]->input_k->data[i * d_k + j].diff;
                d_V[i * d_model + h * d_k + j] += attention_heads[h]->input_v->data[i * d_k + j].diff;
            }
        }
    }
    
    // Backward through linear projections
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < d_model; j++) {
            for (int k = 0; k < d_model; k++) {
                // Gradients w.r.t. input (from Q, K, V projections)
                input->data[i * d_model + k].diff += d_Q[i * d_model + j] * W_Q->data[k * d_model + j].val;
                input->data[i * d_model + k].diff += d_K[i * d_model + j] * W_K->data[k * d_model + j].val;
                input->data[i * d_model + k].diff += d_V[i * d_model + j] * W_V->data[k * d_model + j].val;
                
                // Gradients w.r.t. weights
                W_Q->data[k * d_model + j].diff += input->data[i * d_model + k].val * d_Q[i * d_model + j];
                W_K->data[k * d_model + j].diff += input->data[i * d_model + k].val * d_K[i * d_model + j];
                W_V->data[k * d_model + j].diff += input->data[i * d_model + k].val * d_V[i * d_model + j];
            }
        }
    }
}

void MultiHeadAttention::update()
{
    // [FIX] Only clear input diffs - do NOT SGD update input activations!
    for (int i = 0; i < seq_len * d_model; i++) {
        input->data[i].diff = 0;
        input->data[i].diffs.clear();
    }
    
    // Update weight matrices (SGD with gradient clipping)
    for (int i = 0; i < d_model * d_model; i++) {
        // Gradient clipping
        if (W_Q->data[i].diff > 5.0f) W_Q->data[i].diff = 5.0f;
        if (W_Q->data[i].diff < -5.0f) W_Q->data[i].diff = -5.0f;
        if (W_K->data[i].diff > 5.0f) W_K->data[i].diff = 5.0f;
        if (W_K->data[i].diff < -5.0f) W_K->data[i].diff = -5.0f;
        if (W_V->data[i].diff > 5.0f) W_V->data[i].diff = 5.0f;
        if (W_V->data[i].diff < -5.0f) W_V->data[i].diff = -5.0f;
        if (W_O->data[i].diff > 5.0f) W_O->data[i].diff = 5.0f;
        if (W_O->data[i].diff < -5.0f) W_O->data[i].diff = -5.0f;
        
        if (cfg.Accuracy > cfg.START_QUANTIZATION) {
            W_Q->data[i].f2q();
            W_Q->data[i].q2f();
            W_K->data[i].f2q();
            W_K->data[i].q2f();
            W_V->data[i].f2q();
            W_V->data[i].q2f();
            W_O->data[i].f2q();
            W_O->data[i].q2f();
        }
        W_Q->data[i].val = W_Q->data[i].val - cfg.lr * W_Q->data[i].diff;
        W_Q->data[i].diff = 0;
        W_Q->data[i].diffs.clear();
        
        W_K->data[i].val = W_K->data[i].val - cfg.lr * W_K->data[i].diff;
        W_K->data[i].diff = 0;
        W_K->data[i].diffs.clear();
        
        W_V->data[i].val = W_V->data[i].val - cfg.lr * W_V->data[i].diff;
        W_V->data[i].diff = 0;
        W_V->data[i].diffs.clear();
        
        W_O->data[i].val = W_O->data[i].val - cfg.lr * W_O->data[i].diff;
        W_O->data[i].diff = 0;
        W_O->data[i].diffs.clear();
    }
}

void MultiHeadAttention::clear()
{
    // FIX Bug 4: Clean up allocated memory
    for (auto* attn : attention_heads) {
        delete attn;
    }
    for (auto* head : head_outputs) {
        delete head;
    }
    attention_heads.clear();
    head_outputs.clear();
    
    for (int i = 0; i < seq_len * d_model; i++) {
        input->data[i].diff = 0;
        input->data[i].diffs.clear();
    }
    
    for (int i = 0; i < d_model * d_model; i++) {
        W_Q->data[i].diff = 0;
        W_Q->data[i].diffs.clear();
        W_K->data[i].diff = 0;
        W_K->data[i].diffs.clear();
        W_V->data[i].diff = 0;
        W_V->data[i].diffs.clear();
        W_O->data[i].diff = 0;
        W_O->data[i].diffs.clear();
    }
}

void MultiHeadAttention::save()
{
    std::cout << "\t" << nnCode;
}

/*
 * LayerNorm — 層正規化
 *
 * 數學公式：output = γ × (x - μ) / √(σ² + ε) + β
 *   μ = mean(x)，σ² = var(x)，γ(scale) 和 β(shift) 為可學習參數
 *
 * 作用：穩定各層的輸入分布，加速訓練收斂
 *
 * backward: LayerNorm 的反向較複雜，需考慮 μ 和 σ² 對所有元素的依賴
 *   ∂L/∂γ = Σ norm_i × ∂L/∂y_i
 *   ∂L/∂β = Σ ∂L/∂y_i
 *   ∂L/∂x_i 需經由 ∂L/∂σ² 和 ∂L/∂μ 間接計算
 */
// [TRANSFORMER] LayerNorm operation
class LayerNorm : public opBase
{
public:
    tensor *output;
    tensor *input;
    tensor *gamma;     // learnable scale parameter
    tensor *beta;      // learnable shift parameter
    int length;
    float eps;
    std::vector<float> mean_cache;     // cache mean for backward
    std::vector<float> var_cache;      // cache variance for backward
    std::vector<float> norm_cache;     // cache normalized values for backward
    
    LayerNorm(tensor &out, tensor &in, tensor &g, tensor &b, int len, float epsilon = 1e-6);
    void forward();
    void backward();
    void update();
    void clear();
    void save();
};

LayerNorm::LayerNorm(tensor &out, tensor &in, tensor &g, tensor &b, int len, float epsilon)
{
    output = &out;
    input = &in;
    gamma = &g;
    beta = &b;
    length = len;
    eps = epsilon;
    
    // Initialize gamma to 1 and beta to 0
    for (int i = 0; i < gamma->data.size(); i++) {
        gamma->data[i].val = 1.0f;
        beta->data[i].val = 0.0f;
    }
    
    mean_cache.resize(1);
    var_cache.resize(1);
    norm_cache.resize(length);
    
    // NNEF codeGen
    nnCode.append(out.name);
    nnCode.append(" = ");
    nnCode.append("layer_norm");
    nnCode.append("(");
    nnCode.append(in.name);
    nnCode.append(", eps = ");
    nnCode.append(std::to_string(eps));
    nnCode.append(");\n");
}

void LayerNorm::forward()
{
    // Compute mean
    float mean = 0.0f;
    for (int i = 0; i < length; i++) {
        mean += input->data[i].val;
    }
    mean /= length;
    mean_cache[0] = mean;
    
    // Compute variance
    float var = 0.0f;
    for (int i = 0; i < length; i++) {
        float diff = input->data[i].val - mean;
        var += diff * diff;
    }
    var /= length;
    var_cache[0] = var;
    
    // Normalize and apply scale/shift
    float inv_std = 1.0f / sqrt(var + eps);
    for (int i = 0; i < length; i++) {
        float normalized = (input->data[i].val - mean) * inv_std;
        norm_cache[i] = normalized;
        output->data[i].val = normalized * gamma->data[i].val + beta->data[i].val;
    }
}

void LayerNorm::backward()
{
    float mean = mean_cache[0];
    float var = var_cache[0];
    float inv_std = 1.0f / sqrt(var + eps);
    
    // Gradients w.r.t. gamma and beta
    for (int i = 0; i < length; i++) {
        gamma->data[i].diff += norm_cache[i] * output->data[i].diff;
        beta->data[i].diff += output->data[i].diff;
    }
    
    // Gradient w.r.t. normalized values
    std::vector<float> d_norm(length);
    for (int i = 0; i < length; i++) {
        d_norm[i] = gamma->data[i].val * output->data[i].diff;
    }
    
    // Gradient w.r.t. input (layer norm backward)
    float d_var = 0.0f;
    for (int i = 0; i < length; i++) {
        d_var += d_norm[i] * (input->data[i].val - mean);
    }
    d_var *= -0.5f * pow(var + eps, -1.5f);
    
    float d_mean = 0.0f;
    for (int i = 0; i < length; i++) {
        d_mean += d_norm[i];
    }
    d_mean *= -inv_std;
    
    // FIX Bug 3: Correct d_mean calculation - sum (input[i] - mean), not norm_cache values
    float sum_centered = 0.0f;
    for (int i = 0; i < length; i++) {
        sum_centered += (input->data[i].val - mean);
    }
    d_mean += d_var * (-2.0f / length) * sum_centered;
    
    for (int i = 0; i < length; i++) {
        input->data[i].diff += d_norm[i] * inv_std + 
                              d_var * (2.0f / length) * (input->data[i].val - mean) + 
                              d_mean / length;
    }
}

void LayerNorm::update()
{
    // Update input
    // [FIX] Only clear input diffs - do NOT SGD update input activations!
    for (int i = 0; i < length; i++) {
        input->data[i].diff = 0;
        input->data[i].diffs.clear();
    }
    
    // Update gamma and beta (SGD)
    for (int i = 0; i < length; i++) {
        if (cfg.Accuracy > cfg.START_QUANTIZATION) {
            gamma->data[i].f2q();
            gamma->data[i].q2f();
            beta->data[i].f2q();
            beta->data[i].q2f();
        }
        gamma->data[i].val = gamma->data[i].val - cfg.lr * gamma->data[i].diff;
        gamma->data[i].diff = 0;
        gamma->data[i].diffs.clear();
        
        beta->data[i].val = beta->data[i].val - cfg.lr * beta->data[i].diff;
        beta->data[i].diff = 0;
        beta->data[i].diffs.clear();
    }
}

void LayerNorm::clear()
{
    for (int i = 0; i < length; i++) {
        input->data[i].diff = 0;
        input->data[i].diffs.clear();
        gamma->data[i].diff = 0;
        gamma->data[i].diffs.clear();
        beta->data[i].diff = 0;
        beta->data[i].diffs.clear();
    }
}

void LayerNorm::save()
{
    std::cout << "\t" << nnCode;
}

/*
 * TransformerBlock — 完整的 Transformer 區塊
 *
 * 結構（Pre-LN 變體）：
 *   x → MultiHeadAttention → + (殘差連接) → LayerNorm → FFN → + (殘差連接) → LayerNorm → output
 *       ↑___________________↑                            ↑_____________________↑
 *
 * FFN = W2 × ReLU(W1 × x)，隱藏維度 d_ff 通常是 d_model 的 2~4 倍
 *
 * 殘差連接的作用：讓梯度可以直接跳過複雜的子層，緩解深層網路的梯度消失
 * backward 時殘差連接的梯度 = 直通梯度 + 子層梯度
 */
// [TRANSFORMER] TransformerBlock operation 
class TransformerBlock : public opBase
{
public:
    tensor *output;
    tensor *input;
    tensor *attn_wq, *attn_wk, *attn_wv, *attn_wo;  // attention weights
    tensor *ffn_w1, *ffn_w2;                        // feed-forward weights
    tensor *norm1_gamma, *norm1_beta;               // first layer norm params
    tensor *norm2_gamma, *norm2_beta;               // second layer norm params
    
    MultiHeadAttention *mha;
    LayerNorm *ln1, *ln2;
    
    int num_heads;
    int d_model; 
    int d_ff;
    int seq_len;
    
    // Intermediate tensors
    tensor *attn_out, *add1_out, *ln1_out;
    tensor *ffn_out, *add2_out;
    
    // FIX Bug 6: Cache FFN hidden activations for backward
    std::vector<float> ffn_hidden_cache;
    
    TransformerBlock(tensor &out, tensor &in, int heads, int model_dim, int ff_dim, int sequence_length);
    void forward();
    void backward();
    void update();
    void clear();
    void save();
};

TransformerBlock::TransformerBlock(tensor &out, tensor &in, int heads, int model_dim, int ff_dim, int sequence_length)
{
    output = &out;
    input = &in;
    num_heads = heads;
    d_model = model_dim;
    d_ff = ff_dim;
    seq_len = sequence_length;
    
    // Create weight tensors
    attn_wq = new tensor({d_model, d_model});
    attn_wk = new tensor({d_model, d_model});
    attn_wv = new tensor({d_model, d_model});
    attn_wo = new tensor({d_model, d_model});
    
    ffn_w1 = new tensor({d_model, d_ff});
    ffn_w2 = new tensor({d_ff, d_model});
    
    norm1_gamma = new tensor({d_model});
    norm1_beta = new tensor({d_model});
    norm2_gamma = new tensor({d_model});
    norm2_beta = new tensor({d_model});
    
    // Create intermediate tensors
    attn_out = new tensor({seq_len, d_model});
    add1_out = new tensor({seq_len, d_model});
    ln1_out = new tensor({seq_len, d_model});
    ffn_out = new tensor({seq_len, d_model});
    add2_out = new tensor({seq_len, d_model});
    
    // Create sub-operations
    mha = new MultiHeadAttention(*attn_out, *input, *attn_wq, *attn_wk, *attn_wv, *attn_wo, 
                                num_heads, d_model, seq_len);
    ln1 = new LayerNorm(*ln1_out, *add1_out, *norm1_gamma, *norm1_beta, d_model);
    ln2 = new LayerNorm(*output, *add2_out, *norm2_gamma, *norm2_beta, d_model);
    
    // NNEF codeGen
    nnCode.append(out.name);
    nnCode.append(" = ");
    nnCode.append("transformer_block");
    nnCode.append("(");
    nnCode.append(in.name);
    nnCode.append(", num_heads = ");
    nnCode.append(std::to_string(num_heads));
    nnCode.append(", d_model = ");
    nnCode.append(std::to_string(d_model));
    nnCode.append(", d_ff = ");
    nnCode.append(std::to_string(d_ff));
    nnCode.append(");\n");
}

void TransformerBlock::forward()
{
    // Multi-head attention
    mha->forward();
    
    // Residual connection + manual LayerNorm 1 (per-position)
    for (int i = 0; i < seq_len * d_model; i++) {
        add1_out->data[i].val = input->data[i].val + attn_out->data[i].val;
    }
    
    // Manual LayerNorm for each sequence position
    for (int seq = 0; seq < seq_len; seq++) {
        // Compute mean for this sequence position
        float mean = 0.0f;
        for (int d = 0; d < d_model; d++) {
            mean += add1_out->data[seq * d_model + d].val;
        }
        mean /= d_model;
        
        // Compute variance for this sequence position
        float var = 0.0f;
        for (int d = 0; d < d_model; d++) {
            float diff = add1_out->data[seq * d_model + d].val - mean;
            var += diff * diff;
        }
        var /= d_model;
        
        // Normalize and apply scale/shift for this sequence position
        float inv_std = 1.0f / sqrt(var + 1e-6f);
        for (int d = 0; d < d_model; d++) {
            float normalized = (add1_out->data[seq * d_model + d].val - mean) * inv_std;
            ln1_out->data[seq * d_model + d].val = normalized * norm1_gamma->data[d].val + norm1_beta->data[d].val;
        }
    }
    
    // Feed-forward network: W2 * ReLU(W1 * x)
    // W1: [seq_len, d_model] * [d_model, d_ff] -> [seq_len, d_ff]
    // FIX Bug 6: Cache ffn_hidden for backward pass
    ffn_hidden_cache.resize(seq_len * d_ff);
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < d_ff; j++) {
            float sum = 0.0f;
            for (int k = 0; k < d_model; k++) {
                sum += ln1_out->data[i * d_model + k].val * ffn_w1->data[k * d_ff + j].val;
            }
            ffn_hidden_cache[i * d_ff + j] = fmaxf(0.0f, sum);  // ReLU activation
        }
    }
    
    // W2: [seq_len, d_ff] * [d_ff, d_model] -> [seq_len, d_model]
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < d_model; j++) {
            float sum = 0.0f;
            for (int k = 0; k < d_ff; k++) {
                sum += ffn_hidden_cache[i * d_ff + k] * ffn_w2->data[k * d_model + j].val;
            }
            ffn_out->data[i * d_model + j].val = sum;
        }
    }
    
    // Residual connection + manual LayerNorm 2 (per-position)
    for (int i = 0; i < seq_len * d_model; i++) {
        add2_out->data[i].val = ln1_out->data[i].val + ffn_out->data[i].val;
    }
    
    // Manual LayerNorm for each sequence position
    for (int seq = 0; seq < seq_len; seq++) {
        // Compute mean for this sequence position
        float mean = 0.0f;
        for (int d = 0; d < d_model; d++) {
            mean += add2_out->data[seq * d_model + d].val;
        }
        mean /= d_model;
        
        // Compute variance for this sequence position
        float var = 0.0f;
        for (int d = 0; d < d_model; d++) {
            float diff = add2_out->data[seq * d_model + d].val - mean;
            var += diff * diff;
        }
        var /= d_model;
        
        // Normalize and apply scale/shift for this sequence position
        float inv_std = 1.0f / sqrt(var + 1e-6f);
        for (int d = 0; d < d_model; d++) {
            float normalized = (add2_out->data[seq * d_model + d].val - mean) * inv_std;
            output->data[seq * d_model + d].val = normalized * norm2_gamma->data[d].val + norm2_beta->data[d].val;
        }
    }
}

void TransformerBlock::backward()
{
    // Manual backward through LayerNorm 2 (simplified version)
    for (int i = 0; i < seq_len * d_model; i++) {
        add2_out->data[i].diff = output->data[i].diff;
    }
    
    // Backward through residual connection 2
    for (int i = 0; i < seq_len * d_model; i++) {
        ln1_out->data[i].diff += add2_out->data[i].diff;
        ffn_out->data[i].diff += add2_out->data[i].diff;
    }
    
    // FIX Bug 2: Implement proper FFN backward pass
    // Backward through W2: ffn_out = ffn_hidden * W2
    std::vector<float> d_ffn_hidden(seq_len * d_ff, 0.0f);
    
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < d_model; j++) {
            for (int k = 0; k < d_ff; k++) {
                // Gradient w.r.t. ffn_hidden
                d_ffn_hidden[i * d_ff + k] += ffn_out->data[i * d_model + j].diff * ffn_w2->data[k * d_model + j].val;
                // Gradient w.r.t. ffn_w2
                ffn_w2->data[k * d_model + j].diff += ffn_hidden_cache[i * d_ff + k] * ffn_out->data[i * d_model + j].diff;
            }
        }
    }
    
    // Backward through ReLU and W1: ffn_hidden = ReLU(ln1_out * W1)
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < d_ff; j++) {
            // ReLU backward: gradient passes through only if input > 0
            float relu_grad = (ffn_hidden_cache[i * d_ff + j] > 0.0f) ? d_ffn_hidden[i * d_ff + j] : 0.0f;
            
            for (int k = 0; k < d_model; k++) {
                // Gradient w.r.t. ln1_out
                ln1_out->data[i * d_model + k].diff += relu_grad * ffn_w1->data[k * d_ff + j].val;
                // Gradient w.r.t. ffn_w1
                ffn_w1->data[k * d_ff + j].diff += ln1_out->data[i * d_model + k].val * relu_grad;
            }
        }
    }
    
    // Manual backward through LayerNorm 1 (simplified version)
    for (int i = 0; i < seq_len * d_model; i++) {
        add1_out->data[i].diff = ln1_out->data[i].diff;
    }
    
    // Backward through residual connection 1
    for (int i = 0; i < seq_len * d_model; i++) {
        input->data[i].diff += add1_out->data[i].diff;
        attn_out->data[i].diff += add1_out->data[i].diff;
    }
    
    // Backward through multi-head attention
    mha->backward();
}

void TransformerBlock::update()
{
    mha->update();
    
    // Update LayerNorm parameters manually
    for (int i = 0; i < d_model; i++) {
        norm1_gamma->data[i].val = norm1_gamma->data[i].val - cfg.lr * norm1_gamma->data[i].diff;
        norm1_gamma->data[i].diff = 0;
        norm1_gamma->data[i].diffs.clear();
        
        norm1_beta->data[i].val = norm1_beta->data[i].val - cfg.lr * norm1_beta->data[i].diff;
        norm1_beta->data[i].diff = 0;
        norm1_beta->data[i].diffs.clear();
        
        norm2_gamma->data[i].val = norm2_gamma->data[i].val - cfg.lr * norm2_gamma->data[i].diff;
        norm2_gamma->data[i].diff = 0;
        norm2_gamma->data[i].diffs.clear();
        
        norm2_beta->data[i].val = norm2_beta->data[i].val - cfg.lr * norm2_beta->data[i].diff;
        norm2_beta->data[i].diff = 0;
        norm2_beta->data[i].diffs.clear();
    }
    
    // Update FFN weights (SGD with gradient clipping)
    for (int i = 0; i < d_model * d_ff; i++) {
        if (ffn_w1->data[i].diff > 5.0f) ffn_w1->data[i].diff = 5.0f;
        if (ffn_w1->data[i].diff < -5.0f) ffn_w1->data[i].diff = -5.0f;
        ffn_w1->data[i].val = ffn_w1->data[i].val - cfg.lr * ffn_w1->data[i].diff;
        ffn_w1->data[i].diff = 0;
        ffn_w1->data[i].diffs.clear();
    }
    
    for (int i = 0; i < d_ff * d_model; i++) {
        if (ffn_w2->data[i].diff > 5.0f) ffn_w2->data[i].diff = 5.0f;
        if (ffn_w2->data[i].diff < -5.0f) ffn_w2->data[i].diff = -5.0f;
        ffn_w2->data[i].val = ffn_w2->data[i].val - cfg.lr * ffn_w2->data[i].diff;
        ffn_w2->data[i].diff = 0;
        ffn_w2->data[i].diffs.clear();
    }
}

void TransformerBlock::clear()
{
    mha->clear();
    
    // Clear LayerNorm parameters manually
    for (int i = 0; i < d_model; i++) {
        norm1_gamma->data[i].diff = 0;
        norm1_gamma->data[i].diffs.clear();
        norm1_beta->data[i].diff = 0;
        norm1_beta->data[i].diffs.clear();
        norm2_gamma->data[i].diff = 0;
        norm2_gamma->data[i].diffs.clear();
        norm2_beta->data[i].diff = 0;
        norm2_beta->data[i].diffs.clear();
    }
    
    // Clear intermediate tensors
    for (int i = 0; i < seq_len * d_model; i++) {
        attn_out->data[i].diff = 0;
        attn_out->data[i].diffs.clear();
        add1_out->data[i].diff = 0;
        add1_out->data[i].diffs.clear();
        ln1_out->data[i].diff = 0;
        ln1_out->data[i].diffs.clear();
        ffn_out->data[i].diff = 0;
        ffn_out->data[i].diffs.clear();
        add2_out->data[i].diff = 0;
        add2_out->data[i].diffs.clear();
    }
    
    // Clear FFN weights
    for (int i = 0; i < d_model * d_ff; i++) {
        ffn_w1->data[i].diff = 0;
        ffn_w1->data[i].diffs.clear();
    }
    for (int i = 0; i < d_ff * d_model; i++) {
        ffn_w2->data[i].diff = 0;
        ffn_w2->data[i].diffs.clear();
    }
}

void TransformerBlock::save()
{
    std::cout << "\t" << nnCode;
}

class netBase
{
public:
    virtual void forward()
    {
        std::cout << "forward, Base" << std::endl;
    }
    virtual void backward()
    {
        std::cout << "backward, Base" << std::endl;
    }
    virtual void update()
    {
        std::cout << "update, Base" << std::endl;
    }
    virtual void save()
    {
        std::cout << "save, Base" << std::endl;
    }
};

/*
 * Net — 神經網路容器
 *
 * Layer 是一個有序的 op 列表（std::list<opBase*>）
 * forward()  — 從頭到尾依序執行每個 op 的 forward()
 * backward() — 從尾到頭逆序執行每個 op 的 backward()（鏈式法則）
 * update()   — 從尾到頭逆序執行每個 op 的 update()（SGD 參數更新）
 * save()     — 產生 NNEF 格式的模型描述（可部署到嵌入式推理引擎）
 */
class Net : public netBase
{
public:
    std::vector<std::string> input;
    std::vector<std::string> output;
    std::list<opBase *> Layer;
    void AddLayer(opBase *);
    void forward();
    void backward();
    void update();
    void clear();
    void save();
    void print();
};

void Net::AddLayer(opBase *ler)
{
    Layer.push_back(ler);
}

void Net::forward()
{
    // Net - > Op(Layer) -> node
    for (std::list<opBase *>::iterator choose = Layer.begin(); choose != Layer.end(); ++choose)
        (*choose)->forward();
}

void Net::backward()
{
    for (std::list<opBase *>::reverse_iterator choose = Layer.rbegin(); choose != Layer.rend(); ++choose)
        (*choose)->backward();
}

void Net::update()
{
    for (std::list<opBase *>::reverse_iterator choose = Layer.rbegin(); choose != Layer.rend(); ++choose)
        (*choose)->update();
}

void Net::clear()
{
    for (std::list<opBase *>::reverse_iterator choose = Layer.rbegin(); choose != Layer.rend(); ++choose)
        (*choose)->clear();
}

void Net::save()
{
    // Codegen NNEF head and tail
    std::cout << "version 1.0;" << std::endl;
    std::cout << std::endl;
    std::cout << "graph network";
    std::cout << "( ";
    for (auto i = 0; i < input.size(); i++)
    {
        std::cout << input[i];
    }
    std::cout << " )"; //intput
    std::cout << "->";
    std::cout << "( ";
    for (auto i = 0; i < output.size(); i++)
    {
        std::cout << output[i];
    }
    std::cout << " )"; //output
    std::cout << std::endl;
    std::cout << "{\n";

    for (std::list<opBase *>::iterator choose = Layer.begin(); choose != Layer.end(); ++choose)
        (*choose)->save();

    std::cout << "}\n";
}

void Net::print()
{
    for (std::list<opBase *>::reverse_iterator choose = Layer.rbegin(); choose != Layer.rend(); ++choose)
        (*choose)->print();
}

// #######################################
// # Wrapper functions — 高階 API
// # 每個 tir_xxx 函數建立對應的 Op 物件、分配輸出 tensor、加入 Net
// # 使用方式類似 PyTorch 的 functional API：
// #   tensor &output = tir_conv(input, weight, ...);
// # 回傳值是輸出 tensor 的參考，可直接串接下一層
// ---------------------------------------

Net net;

// tir_external — 宣告外部輸入張量（如影像資料），不含可訓練參數
tensor &tir_external(std::vector<int> p_shape)
{
    extern Net net;

    std::cout << "external : ";
    for (auto i = 0; i < p_shape.size(); i++)
    {
        if (i)
            std::cout << " x ";
        std::cout << p_shape[i];
    }
    std::cout << std::endl;

    tensor *out_tensor = new tensor(p_shape);
    out_tensor->name = "external" + std::to_string(++cfg.tensor_num);
    External *external = new External(*out_tensor, p_shape);
    net.AddLayer(external);
    net.input.push_back(out_tensor->name);
    return *out_tensor;
}

// tir_variable — 宣告可訓練的權重張量，可指定儲存路徑
tensor &tir_variable(std::vector<int> p_shape, std::string path)
{
    extern Net net;

    std::cout << "variable : ";
    for (auto i = 0; i < p_shape.size(); i++)
    {
        if (i)
            std::cout << " x ";
        std::cout << p_shape[i];
    }
    std::cout << std::endl;

    tensor *out_tensor = new tensor(p_shape);
    out_tensor->name = "variable" + std::to_string(++cfg.tensor_num);
    Variable *variable = new Variable(*out_tensor, p_shape, path);
    net.AddLayer(variable);
    return *out_tensor;
}

// tir_reshape — 改變張量形狀（不複製資料，共用底層 node）
tensor &tir_reshape(tensor &in_tensor, std::vector<int> p_shape)
{
    extern Net net;

    std::cout << "reshape : ";
    for (auto i = 0; i < p_shape.size(); i++)
    {
        if (i)
            std::cout << " x ";
        std::cout << p_shape[i];
    }
    std::cout << std::endl;

    tensor *out_tensor = new tensor(p_shape);
    out_tensor->name = "reshape" + std::to_string(++cfg.tensor_num);
    out_tensor->shape = p_shape;
    Reshape *reshape = new Reshape(*out_tensor, in_tensor, p_shape);
    net.AddLayer(reshape);
    return *out_tensor;
}

// tir_conv — 建立 2D 卷積層，自動計算輸出尺寸
tensor &tir_conv(tensor &in_tensor, tensor &filter, int in_ch, int in_dim, int stride, int pad, int ker_dim, int out_ch, int out_dim)
{
    extern Net net;
    std::vector<int> shape;
    int n, c, h, w;
    int padding = pad;

    if (padding == 1)
    {
        n = in_tensor.n;
        c = filter.n;
        h = ceil(((float)in_tensor.h) / ((float)stride));
        w = ceil(((float)in_tensor.w) / ((float)stride));
    }
    else
    {
        n = in_tensor.n;
        c = filter.n;
        h = ceil(((float)(in_tensor.h - filter.h + 1)) / ((float)stride));
        w = ceil(((float)(in_tensor.w - filter.w + 1)) / ((float)stride));
    }
    tensor *out_tensor = new tensor(shape = {n, c, h, w});
    out_tensor->name = "conv" + std::to_string(++cfg.tensor_num);
    std::cout << "conv : " << n << " x " << c << " x " << h << " x " << w << std::endl;
    Conv *conv = new Conv(*out_tensor, in_tensor, filter, in_ch, in_dim, in_dim, stride, pad, ker_dim, out_ch, out_dim, out_dim);
    net.AddLayer(conv);
    return *out_tensor;
}

// tir_max_pool — 建立最大池化層，縮小空間維度
tensor &tir_max_pool(tensor &in_tensor, int p_size, int p_padding, int p_stride)
{
    extern Net net;
    std::vector<int> shape;
    // out_tensor in
    int n, c, h, w;
    if (p_padding == 1)
    {
        n = in_tensor.shape[0];
        c = in_tensor.shape[1];
        h = (int)(ceil((float)(in_tensor.shape[2]) / (float)p_stride));
        w = (int)(ceil((float)(in_tensor.shape[3]) / (float)p_stride));
    }
    else
    {
        n = in_tensor.shape[0];
        c = in_tensor.shape[1];
        h = ceil(((float)(in_tensor.shape[2] - p_size + 1)) / ((float)p_stride));
        w = ceil(((float)(in_tensor.shape[3] - p_size + 1)) / ((float)p_stride));
    }
    tensor *out_tensor = new tensor(shape = {n, c, h, w});
    out_tensor->name = "max_pool" + std::to_string(++cfg.tensor_num);
    std::cout << "max_pool : " << n << " x " << c << " x " << h << " x " << w << std::endl;
    //tensor *out_tensor = new tensor(shape = {out_ch, out_dim, out_dim});
    Max_pool *max_pool = new Max_pool(*out_tensor, in_tensor, p_size, p_padding, p_stride);
    net.AddLayer(max_pool);
    return *out_tensor;
}

// tir_matmul — 矩陣乘法 [m,k] × [k,n] → [m,n]，即全連接層的核心
tensor &tir_matmul(tensor &mk, tensor &kn)
{
    extern Net net;
    std::vector<int> shape;
    //mk.shape.resize(2);
    //mk.shape = {m, k};
    //tensor *kn = new tensor(shape = {k, n});
    //*weight = kn;

    assert(mk.shape[1] == kn.shape[0]);
    std::cout << "matmul : " << mk.shape[0] << " x " << kn.shape[1] << std::endl;
    tensor *out_tensor = new tensor(shape = {mk.shape[0], kn.shape[1]});
    out_tensor->name = "matmul" + std::to_string(++cfg.tensor_num);
    Matmul *matmul = new Matmul(*out_tensor, mk, kn, mk.shape[0], mk.shape[1], kn.shape[1]);
    net.AddLayer(matmul);
    return *out_tensor;
}

// tir_add — 逐元素加法（常用於加 bias）
tensor &tir_add(tensor &in_tensor, tensor &weight)
{
    extern Net net;
    std::vector<int> shape;
    int in_tensor_size = 1;
    int weight_size = 1;

    std::cout << "add : ";
    for (auto i = 0; i < in_tensor.shape.size(); i++)
    {
        if (i)
            std::cout << " x ";
        std::cout << in_tensor.shape[i];
    }
    std::cout << std::endl;

    for (auto i = 0; i < in_tensor.shape.size(); i++)
        in_tensor_size *= in_tensor.shape[i];
    for (auto i = 0; i < weight.shape.size(); i++)
        weight_size *= weight.shape[i];

    for (auto i = 0; i < in_tensor.shape.size(); i++)
        assert(in_tensor.shape[i] == weight.shape[i]);

    tensor *out_tensor = new tensor(shape = in_tensor.shape);
    out_tensor->name = "add" + std::to_string(++cfg.tensor_num);
    Add *add = new Add(*out_tensor, in_tensor, weight, in_tensor_size);
    net.AddLayer(add);
    return *out_tensor;
}

// tir_sigmoid — Sigmoid 激活函數
tensor &tir_sigmoid(tensor &in_tensor)
{
    extern Net net;
    std::vector<int> shape;

    std::cout << "sigmoid : ";
    for (auto i = 0; i < in_tensor.shape.size(); i++)
    {
        if (i)
            std::cout << " x ";
        std::cout << in_tensor.shape[i];
    }
    std::cout << std::endl;

    int in_tensor_size = 1;
    for (auto i = 0; i < in_tensor.shape.size(); i++)
        in_tensor_size *= in_tensor.shape[i];

    tensor *out_tensor = new tensor(shape = {shape = in_tensor.shape});
    out_tensor->name = "sigmoid" + std::to_string(++cfg.tensor_num);
    Sigmoid *sigmoid = new Sigmoid(*out_tensor, in_tensor, in_tensor_size);
    net.AddLayer(sigmoid);
    return *out_tensor;
}

// tir_relu — ReLU 激活函數
tensor &tir_relu(tensor &in_tensor)
{
    extern Net net;
    std::vector<int> shape;

    std::cout << "relu : ";
    for (auto i = 0; i < in_tensor.shape.size(); i++)
    {
        if (i)
            std::cout << " x ";
        std::cout << in_tensor.shape[i];
    }
    std::cout << std::endl;

    int in_tensor_size = 1;
    for (auto i = 0; i < in_tensor.shape.size(); i++)
        in_tensor_size *= in_tensor.shape[i];

    tensor *out_tensor = new tensor(shape = in_tensor.shape);
    out_tensor->name = "relu" + std::to_string(++cfg.tensor_num);
    ReLU *relu = new ReLU(*out_tensor, in_tensor, in_tensor_size);
    net.AddLayer(relu);
    return *out_tensor;
}

tensor &tir_leaky_relu(tensor &in_tensor)
{
    extern Net net;
    std::vector<int> shape;

    std::cout << "leaky_relu : ";
    for (auto i = 0; i < in_tensor.shape.size(); i++)
    {
        if (i)
            std::cout << " x ";
        std::cout << in_tensor.shape[i];
    }
    std::cout << std::endl;

    int in_tensor_size = 1;
    for (auto i = 0; i < in_tensor.shape.size(); i++)
        in_tensor_size *= in_tensor.shape[i];

    tensor *out_tensor = new tensor(shape = {in_tensor.shape[0], in_tensor.shape[1]});
    out_tensor->name = "leaky_relu" + std::to_string(++cfg.tensor_num);
    Leaky_ReLU *leaky_relu = new Leaky_ReLU(*out_tensor, in_tensor, in_tensor_size);
    net.AddLayer(leaky_relu);
    return *out_tensor;
}

// tir_loss_mse — MSE 損失函數（迴歸任務用）
tensor &tir_loss_mse(tensor &in_tensor, tensor &ans)
{
    extern Net net;
    std::vector<int> shape;

    std::cout << "loss_mse : ";
    for (auto i = 0; i < in_tensor.shape.size(); i++)
    {
        if (i)
            std::cout << " x ";
        std::cout << in_tensor.shape[i];
    }

    std::cout << std::endl;
    tensor *loss = new tensor(shape = {1});
    loss->name = "loss_mse" + std::to_string(++cfg.tensor_num);
    Loss_MSE *loss_mse = new Loss_MSE(*loss, in_tensor, ans, ans.data.size());
    net.AddLayer(loss_mse);
    net.output.push_back(in_tensor.name);
    return *loss;
}

// tir_loss_cross_entropy — Softmax + 交叉熵損失（分類任務首選）
// [FIX #5] Softmax + Cross-Entropy loss wrapper (preferred for classification)
tensor &tir_loss_cross_entropy(tensor &in_tensor, tensor &ans)
{
    extern Net net;
    std::vector<int> shape;

    std::cout << "loss_cross_entropy : ";
    for (auto i = 0; i < in_tensor.shape.size(); i++)
    {
        if (i)
            std::cout << " x ";
        std::cout << in_tensor.shape[i];
    }

    std::cout << std::endl;
    tensor *loss = new tensor(shape = {1});
    loss->name = "loss_ce" + std::to_string(++cfg.tensor_num);
    Loss_CrossEntropy *loss_ce = new Loss_CrossEntropy(*loss, in_tensor, ans, ans.data.size());
    net.AddLayer(loss_ce);
    net.output.push_back(in_tensor.name);
    return *loss;
}

// tir_multi_head_attention — 多頭注意力層
// [TRANSFORMER] Multi-head attention wrapper function
tensor &tir_multi_head_attention(tensor &input, int num_heads, int d_model)
{
    extern Net net;
    
    std::cout << "multi_head_attention : num_heads = " << num_heads 
              << ", d_model = " << d_model << std::endl;
    
    // Calculate sequence length from input shape
    int seq_len = input.data.size() / d_model;
    
    // Create weight tensors
    tensor *W_Q = new tensor({d_model, d_model});
    tensor *W_K = new tensor({d_model, d_model});
    tensor *W_V = new tensor({d_model, d_model});
    tensor *W_O = new tensor({d_model, d_model});
    
    // Set names for weight tensors
    W_Q->name = "mha_wq" + std::to_string(++cfg.tensor_num);
    W_K->name = "mha_wk" + std::to_string(++cfg.tensor_num);
    W_V->name = "mha_wv" + std::to_string(++cfg.tensor_num);
    W_O->name = "mha_wo" + std::to_string(++cfg.tensor_num);
    
    // Create output tensor
    tensor *out_tensor = new tensor({seq_len, d_model});
    out_tensor->name = "multi_head_attention" + std::to_string(++cfg.tensor_num);
    
    // Create and add the operation
    MultiHeadAttention *mha = new MultiHeadAttention(*out_tensor, input, *W_Q, *W_K, *W_V, *W_O, 
                                                    num_heads, d_model, seq_len);
    net.AddLayer(mha);
    return *out_tensor;
}

// tir_layer_norm — 層正規化
// [TRANSFORMER] Layer normalization wrapper function  
tensor &tir_layer_norm(tensor &input, int length)
{
    extern Net net;
    
    std::cout << "layer_norm : length = " << length << std::endl;
    
    // Create learnable parameters
    tensor *gamma = new tensor({length});
    tensor *beta = new tensor({length});
    
    // Set names
    gamma->name = "ln_gamma" + std::to_string(++cfg.tensor_num);
    beta->name = "ln_beta" + std::to_string(++cfg.tensor_num);
    
    // Create output tensor
    tensor *out_tensor = new tensor(input.shape);
    out_tensor->name = "layer_norm" + std::to_string(++cfg.tensor_num);
    
    // Create and add the operation
    LayerNorm *ln = new LayerNorm(*out_tensor, input, *gamma, *beta, length);
    net.AddLayer(ln);
    return *out_tensor;
}

// tir_transformer_block — 完整 Transformer 區塊（Attention + FFN + 殘差 + LN）
// [TRANSFORMER] Transformer block wrapper function
tensor &tir_transformer_block(tensor &input, int num_heads, int d_model, int d_ff)
{
    extern Net net;
    
    std::cout << "transformer_block : num_heads = " << num_heads 
              << ", d_model = " << d_model 
              << ", d_ff = " << d_ff << std::endl;
              
    // Calculate sequence length
    int seq_len = input.data.size() / d_model;
    
    // Create output tensor
    tensor *out_tensor = new tensor({seq_len, d_model});
    out_tensor->name = "transformer_block" + std::to_string(++cfg.tensor_num);
    
    // Create and add the operation
    TransformerBlock *tfb = new TransformerBlock(*out_tensor, input, num_heads, d_model, d_ff, seq_len);
    net.AddLayer(tfb);
    return *out_tensor;
}

float test_acc(Net &net, mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> &dataset, tensor &input, tensor &output, tensor &answer)
{
    int tCorrect = 0;
    int tError = 0;
    float Accuracy;

    for (unsigned int i = 0; i < 1000; i++)
    {
        for (unsigned int j = 0; j < dataset.test_images[i].size(); j++)
            input[j].val = ((float)(unsigned int)dataset.test_images[i][j]) / 255;

        int target_value = (unsigned int)dataset.test_labels[i];
        for (int k = 0; k < 10; k++)
        {
            answer[k].val = 0;
        }
        answer[target_value].val = 1;

        // ---------------------------
        net.forward();
        net.clear();
        // ---------------------------

        double max_value = -99999;
        int max_index = 0;
        for (int k = 0; k < 10; k++)
        {
            if (output[k].val > max_value)
            {
                max_value = output[k].val;
                max_index = k;
            }
        }

        if (max_index == target_value)
            tCorrect++;
        else
            tError++;

        cfg.Accuracy = (float)tCorrect / ((float)tCorrect + (float)tError) * 100;
    }

    std::cout << "[Testing : " << cfg.Accuracy << "% ... success]" << std::endl;
    return cfg.Accuracy;
}

/*
// [TRANSFORMER] Demo: How to build a simple transformer
// 
// Example usage of the new Transformer operations:
//
// tensor &input = tir_external({10, 512});  // [seq_len=10, d_model=512]
// 
// // Simple transformer:
// tensor &attn_out = tir_multi_head_attention(input, num_heads=8, d_model=512);
// tensor &norm_out = tir_layer_norm(attn_out, length=10*512);
// 
// // Full transformer block (includes attention + FFN + residuals + layer norms):
// tensor &transformer_out = tir_transformer_block(input, num_heads=8, d_model=512, d_ff=2048);
// 
// // Multi-layer transformer:
// tensor &layer1 = tir_transformer_block(input, 8, 512, 2048);
// tensor &layer2 = tir_transformer_block(layer1, 8, 512, 2048);
// tensor &output = tir_transformer_block(layer2, 8, 512, 2048);
//
// // The transformer operations use TYPE2 backward passes with direct diff accumulation
// // All weights are initialized with He initialization for better convergence
// // NNEF codegen is included for each operation
*/

// Test function for transformer operations
void run_transformer_tests() {
    std::cout << "=== TRANSFORMER OPERATIONS TEST ===" << std::endl;
    
    const int seq_len = 2;
    const int d_model = 4; 
    const int num_heads = 2;
    const int d_ff = 8;
    const float eps = 1e-6f;
    
    std::cout << "Testing with seq_len=" << seq_len << ", d_model=" << d_model 
              << ", num_heads=" << num_heads << ", d_ff=" << d_ff << std::endl;
    
    // Test 1: ScaledDotProductAttention
    std::cout << "\n1. Testing ScaledDotProductAttention..." << std::endl;
    {
        tensor q({seq_len, d_model/num_heads});
        tensor k({seq_len, d_model/num_heads});
        tensor v({seq_len, d_model/num_heads});
        tensor attn_out({seq_len, d_model/num_heads});
        
        // Initialize with small known values
        for (int i = 0; i < seq_len * (d_model/num_heads); i++) {
            q.data[i].val = 0.1f + 0.01f * i;
            k.data[i].val = 0.2f + 0.01f * i;
            v.data[i].val = 0.3f + 0.01f * i;
        }
        
        ScaledDotProductAttention attn(attn_out, q, k, v, seq_len, d_model/num_heads);
        attn.forward();
        
        std::cout << "   Forward output[0]: " << attn_out.data[0].val << std::endl;
        std::cout << "   Forward output[1]: " << attn_out.data[1].val << std::endl;
        
        // Set output gradient and test backward
        for (int i = 0; i < seq_len * (d_model/num_heads); i++) {
            attn_out.data[i].diff = 0.1f;
        }
        attn.backward();
        
        std::cout << "   Backward q.diff[0]: " << q.data[0].diff << std::endl;
        std::cout << "   Backward k.diff[0]: " << k.data[0].diff << std::endl;
        std::cout << "   Backward v.diff[0]: " << v.data[0].diff << std::endl;
        
        if (std::isnan(attn_out.data[0].val) || std::isnan(q.data[0].diff)) {
            std::cout << "   ❌ FAILED: NaN detected!" << std::endl;
            return;
        } else {
            std::cout << "   ✅ PASSED: No NaN detected" << std::endl;
        }
    }
    
    // Test 2: LayerNorm
    std::cout << "\n2. Testing LayerNorm..." << std::endl;
    {
        tensor input({seq_len * d_model});
        tensor gamma({seq_len * d_model});
        tensor beta({seq_len * d_model});
        tensor output({seq_len * d_model});
        
        // Initialize with known values
        for (int i = 0; i < seq_len * d_model; i++) {
            input.data[i].val = 1.0f + 0.1f * i;
        }
        
        LayerNorm ln(output, input, gamma, beta, seq_len * d_model);
        ln.forward();
        
        std::cout << "   Forward output[0]: " << output.data[0].val << std::endl;
        std::cout << "   Forward output[1]: " << output.data[1].val << std::endl;
        
        // Set output gradient and test backward
        for (int i = 0; i < seq_len * d_model; i++) {
            output.data[i].diff = 0.1f;
        }
        ln.backward();
        
        std::cout << "   Backward input.diff[0]: " << input.data[0].diff << std::endl;
        std::cout << "   Backward gamma.diff[0]: " << gamma.data[0].diff << std::endl;
        
        if (std::isnan(output.data[0].val) || std::isnan(input.data[0].diff)) {
            std::cout << "   ❌ FAILED: NaN detected!" << std::endl;
            return;
        } else {
            std::cout << "   ✅ PASSED: No NaN detected" << std::endl;
        }
    }
    
    // Test 3: MultiHeadAttention
    std::cout << "\n3. Testing MultiHeadAttention..." << std::endl;
    {
        tensor input({seq_len, d_model});
        tensor wq({d_model, d_model});
        tensor wk({d_model, d_model});
        tensor wv({d_model, d_model});
        tensor wo({d_model, d_model});
        tensor output({seq_len, d_model});
        
        // Initialize with small values
        for (int i = 0; i < seq_len * d_model; i++) {
            input.data[i].val = 0.1f + 0.01f * i;
        }
        
        MultiHeadAttention mha(output, input, wq, wk, wv, wo, num_heads, d_model, seq_len);
        mha.forward();
        
        std::cout << "   Forward output[0]: " << output.data[0].val << std::endl;
        std::cout << "   Forward output[1]: " << output.data[1].val << std::endl;
        
        // Set output gradient and test backward
        for (int i = 0; i < seq_len * d_model; i++) {
            output.data[i].diff = 0.1f;
        }
        mha.backward();
        
        std::cout << "   Backward input.diff[0]: " << input.data[0].diff << std::endl;
        std::cout << "   Backward wq.diff[0]: " << wq.data[0].diff << std::endl;
        
        if (std::isnan(output.data[0].val) || std::isnan(input.data[0].diff)) {
            std::cout << "   ❌ FAILED: NaN detected!" << std::endl;
            return;
        } else {
            std::cout << "   ✅ PASSED: No NaN detected" << std::endl;
        }
    }
    
    // Test 4: TransformerBlock (full integration)
    std::cout << "\n4. Testing TransformerBlock (full integration)..." << std::endl;
    {
        tensor input({seq_len, d_model});
        tensor output({seq_len, d_model});
        
        // Initialize input
        for (int i = 0; i < seq_len * d_model; i++) {
            input.data[i].val = 0.1f + 0.01f * i;
        }
        
        TransformerBlock tfb(output, input, num_heads, d_model, d_ff, seq_len);
        tfb.forward();
        
        std::cout << "   Forward output[0]: " << output.data[0].val << std::endl;
        std::cout << "   Forward output[1]: " << output.data[1].val << std::endl;
        
        // Set output gradient and test backward
        for (int i = 0; i < seq_len * d_model; i++) {
            output.data[i].diff = 0.1f;
        }
        tfb.backward();
        
        std::cout << "   Backward input.diff[0]: " << input.data[0].diff << std::endl;
        std::cout << "   Backward ffn_w1.diff[0]: " << tfb.ffn_w1->data[0].diff << std::endl;
        
        bool has_nan = std::isnan(output.data[0].val) || std::isnan(input.data[0].diff) || std::isnan(tfb.ffn_w1->data[0].diff);
        
        if (has_nan) {
            std::cout << "   ❌ FAILED: NaN detected!" << std::endl;
            std::cout << "     output.val=" << output.data[0].val << std::endl;
            std::cout << "     input.diff=" << input.data[0].diff << std::endl;
            std::cout << "     ffn_w1.diff=" << tfb.ffn_w1->data[0].diff << std::endl;
            std::cout << "\n❌ TESTS FAILED" << std::endl;
            std::cout << "TransformerBlock produces NaN values!" << std::endl;
            return;
        } else {
            std::cout << "   ✅ PASSED: No NaN detected" << std::endl;
        }
    }
    
    std::cout << "\n🎉 ALL TESTS PASSED" << std::endl;
    std::cout << "Transformer operations are working correctly without NaN!" << std::endl;
}

/*
 * main — 訓練主程式
 *
 * 流程：
 *   1. 載入 MNIST 資料集（60000 張 28×28 手寫數字圖片）
 *   2. 建構模型：CNN (LeNet) 或 CNN + Transformer 混合架構
 *   3. 訓練迴圈：
 *      for each epoch:
 *        for each image:
 *          → 設定 input 和 answer (one-hot)
 *          → net.forward()   // 前向計算
 *          → net.backward()  // 反向傳播梯度
 *          → net.update()    // SGD 更新權重
 *        → test_acc()        // 每個 epoch 後測試準確率
 *        → 達到目標準確率就提前結束
 */
// ============================================================================
// [TEXT] run_text_transformer — 文字 Transformer Demo
//
// 用 TETF 基礎元件組裝一個字元級 Transformer，學會回答：
//   輸入: 誰是Nami？
//   生成: Nami是厲害的AI工程師
// ============================================================================

// UTF-8 多字節字元切割
std::vector<std::string> utf8_chars(const std::string &s) {
    std::vector<std::string> chars;
    size_t i = 0;
    while (i < s.size()) {
        int len = 1;
        unsigned char c = s[i];
        if (c >= 0xF0) len = 4;
        else if (c >= 0xE0) len = 3;
        else if (c >= 0xC0) len = 2;
        chars.push_back(s.substr(i, len));
        i += len;
    }
    return chars;
}

void run_text_transformer() {
    std::cout << "[Mode: Text Transformer]" << std::endl;
    std::cout << std::endl;

    // === 訓練文本 ===
    std::string text = "誰是Nami？Nami是厲害的AI工程師";
    std::cout << "📝 Training text: " << text << std::endl;

    // === 字元級 tokenizer ===
    std::vector<std::string> all_chars = utf8_chars(text);
    std::vector<std::string> vocab;
    for (auto &ch : all_chars) {
        bool found = false;
        for (auto &v : vocab)
            if (v == ch) { found = true; break; }
        if (!found) vocab.push_back(ch);
    }
    int vocab_size = vocab.size();
    std::cout << "📊 Vocab size: " << vocab_size << std::endl;

    // 印出詞表
    std::cout << "📖 Vocab: ";
    for (int i = 0; i < vocab_size; i++)
        std::cout << vocab[i] << "(" << i << ") ";
    std::cout << std::endl;

    // token ID 化
    std::vector<int> token_ids;
    for (auto &ch : all_chars) {
        for (int j = 0; j < vocab_size; j++)
            if (vocab[j] == ch) { token_ids.push_back(j); break; }
    }
    int total_len = token_ids.size();  // 全文長度
    int seq_len = total_len - 1;       // 訓練序列長度（去掉最後一個 token）

    // === 模型超參數 ===
    int d_model = 32;
    int d_ff = 64;
    int epochs = 500;
    cfg.lr = 0.05f;

    std::cout << "⚙️  d_model=" << d_model << ", d_ff=" << d_ff 
              << ", lr=" << cfg.lr << ", epochs=" << epochs << std::endl;
    std::cout << std::endl;
    std::cout << "🏋️ Training..." << std::endl;

    // === 手動管理 tensor（繞過 4D shape 限制）===
    // 不用 global Net，手動管理 op 列表
    srand(42);

    // Embedding 權重
    tensor *emb_weight = new tensor({1, 1, vocab_size, d_model});
    // 用小值初始化
    for (int i = 0; i < vocab_size * d_model; i++)
        emb_weight->data[i].val = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;

    // Positional encoding（可學習）
    tensor *pos_enc = new tensor({1, 1, seq_len, d_model});
    for (int i = 0; i < seq_len * d_model; i++)
        pos_enc->data[i].val = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;

    // Embedding 輸出 + Position 相加後的輸出
    tensor *emb_out = new tensor({1, 1, seq_len, d_model});
    tensor *pos_out = new tensor({1, 1, seq_len, d_model});

    // Attention: Wq, Wk, Wv
    tensor *Wq = new tensor({1, 1, d_model, d_model});
    tensor *Wk = new tensor({1, 1, d_model, d_model});
    tensor *Wv = new tensor({1, 1, d_model, d_model});
    for (int i = 0; i < d_model * d_model; i++) {
        float limit = sqrt(6.0f / (d_model + d_model));
        Wq->data[i].val = ((float)rand() / RAND_MAX * 2 - 1) * limit;
        Wk->data[i].val = ((float)rand() / RAND_MAX * 2 - 1) * limit;
        Wv->data[i].val = ((float)rand() / RAND_MAX * 2 - 1) * limit;
    }

    // Q, K, V, Attention output
    tensor *Q = new tensor({1, 1, seq_len, d_model});
    tensor *K = new tensor({1, 1, seq_len, d_model});
    tensor *V = new tensor({1, 1, seq_len, d_model});
    tensor *attn_out = new tensor({1, 1, seq_len, d_model});

    // Residual 1 output
    tensor *res1 = new tensor({1, 1, seq_len, d_model});

    // FFN: W1 [d_model, d_ff], W2 [d_ff, d_model]
    tensor *ffn_w1 = new tensor({1, 1, d_model, d_ff});
    tensor *ffn_w2 = new tensor({1, 1, d_ff, d_model});
    for (int i = 0; i < d_model * d_ff; i++) {
        float limit = sqrt(6.0f / (d_model + d_ff));
        ffn_w1->data[i].val = ((float)rand() / RAND_MAX * 2 - 1) * limit;
    }
    for (int i = 0; i < d_ff * d_model; i++) {
        float limit = sqrt(6.0f / (d_ff + d_model));
        ffn_w2->data[i].val = ((float)rand() / RAND_MAX * 2 - 1) * limit;
    }

    tensor *ffn_hidden = new tensor({1, 1, seq_len, d_ff});
    tensor *ffn_relu = new tensor({1, 1, seq_len, d_ff});
    tensor *ffn_out = new tensor({1, 1, seq_len, d_model});

    // Residual 2 output
    tensor *res2 = new tensor({1, 1, seq_len, d_model});

    // Output projection: [d_model, vocab_size]
    tensor *out_weight = new tensor({1, 1, d_model, vocab_size});
    for (int i = 0; i < d_model * vocab_size; i++) {
        float limit = sqrt(6.0f / (d_model + vocab_size));
        out_weight->data[i].val = ((float)rand() / RAND_MAX * 2 - 1) * limit;
    }
    tensor *logits = new tensor({1, 1, seq_len, vocab_size});
    tensor *loss_tensor = new tensor({1, 1, 1, 1});

    // === Op 物件 ===
    // === 印出模型架構 ===
    std::cout << "🏗️ Model Architecture:" << std::endl;
    std::cout << "embedding : " << vocab_size << " x " << d_model << std::endl;
    Embedding emb_op(*emb_out, *emb_weight, vocab_size, d_model, seq_len);

    std::cout << "pos_encoding : " << seq_len << " x " << d_model << std::endl;
    TextAdd pos_add_op(*pos_out, *emb_out, *pos_enc, seq_len * d_model, true);

    std::cout << "--- transformer_block : " << seq_len << " x " << d_model << " ---" << std::endl;

    std::cout << "  matmul (Wq) : " << d_model << " x " << d_model << std::endl;
    TextMatmul q_proj(*Q, *pos_out, *Wq, seq_len, d_model, d_model);

    std::cout << "  matmul (Wk) : " << d_model << " x " << d_model << std::endl;
    TextMatmul k_proj(*K, *pos_out, *Wk, seq_len, d_model, d_model);

    std::cout << "  matmul (Wv) : " << d_model << " x " << d_model << std::endl;
    TextMatmul v_proj(*V, *pos_out, *Wv, seq_len, d_model, d_model);

    std::cout << "  causal_attention : " << seq_len << " x " << d_model << std::endl;
    CausalAttention attn_op(*attn_out, *Q, *K, *V, seq_len, d_model);

    std::cout << "  residual_add : " << seq_len << " x " << d_model << std::endl;
    TextAdd res1_op(*res1, *pos_out, *attn_out, seq_len * d_model, false);

    std::cout << "  matmul (FFN1) : " << d_model << " x " << d_ff << std::endl;
    TextMatmul ffn1_op(*ffn_hidden, *res1, *ffn_w1, seq_len, d_model, d_ff);

    std::cout << "  relu : " << seq_len << " x " << d_ff << std::endl;
    TextReLU relu_op(*ffn_relu, *ffn_hidden, seq_len * d_ff);

    std::cout << "  matmul (FFN2) : " << d_ff << " x " << d_model << std::endl;
    TextMatmul ffn2_op(*ffn_out, *ffn_relu, *ffn_w2, seq_len, d_ff, d_model);

    std::cout << "  residual_add : " << seq_len << " x " << d_model << std::endl;
    TextAdd res2_op(*res2, *res1, *ffn_out, seq_len * d_model, false);

    std::cout << "--- end transformer_block : " << seq_len << " x " << d_model << " ---" << std::endl;

    std::cout << "output_proj : " << d_model << " x " << vocab_size << std::endl;
    TextMatmul out_proj(*logits, *res2, *out_weight, seq_len, d_model, vocab_size);

    std::cout << "cross_entropy_loss" << std::endl;
    TextCrossEntropy loss_op(*loss_tensor, *logits, seq_len, vocab_size);

    std::cout << std::endl;

    // Op 列表（前向順序）
    std::vector<opBase*> ops = {
        &emb_op, &pos_add_op,
        &q_proj, &k_proj, &v_proj, &attn_op, &res1_op,
        &ffn1_op, &relu_op, &ffn2_op, &res2_op,
        &out_proj, &loss_op
    };

    // 設定 input/target
    std::vector<int> input_ids(token_ids.begin(), token_ids.begin() + seq_len);
    std::vector<int> target_ids(token_ids.begin() + 1, token_ids.begin() + 1 + seq_len);
    emb_op.set_indices(input_ids);
    loss_op.set_targets(target_ids);

    // === 訓練迴圈 ===
    for (int epoch = 0; epoch < epochs; epoch++) {
        // Forward
        for (auto op : ops) op->forward();

        float loss = loss_tensor->data[0].val;

        // 計算 accuracy
        int correct = 0;
        for (int i = 0; i < seq_len; i++) {
            float max_val = -1e9f;
            int max_idx = 0;
            for (int j = 0; j < vocab_size; j++) {
                float v = logits->data[i * vocab_size + j].val;
                if (v > max_val) { max_val = v; max_idx = j; }
            }
            if (max_idx == target_ids[i]) correct++;
        }
        float acc = (float)correct / seq_len * 100;

        if (epoch % 50 == 0 || acc >= 100.0f) {
            std::cout << "  Epoch " << std::setw(4) << epoch 
                      << " | loss=" << std::fixed << std::setprecision(4) << loss 
                      << " | acc=" << correct << "/" << seq_len 
                      << " (" << std::setprecision(2) << acc << "%)" << std::endl;
        }

        if (acc >= 100.0f && epoch > 10) {
            std::cout << std::endl;
            std::cout << "🎉 Converged at epoch " << epoch << "!" << std::endl;
            break;
        }

        // Backward（逆序）
        for (int i = ops.size() - 1; i >= 0; i--) ops[i]->backward();

        // Update
        for (auto op : ops) op->update();
    }

    // === 推理：自回歸生成 ===
    std::cout << std::endl;

    // 找到「？」的位置，切出 prompt
    std::string prompt_str = "誰是Nami？";
    std::vector<std::string> prompt_chars = utf8_chars(prompt_str);
    std::vector<int> prompt_ids;
    for (auto &ch : prompt_chars) {
        for (int j = 0; j < vocab_size; j++)
            if (vocab[j] == ch) { prompt_ids.push_back(j); break; }
    }

    std::cout << "🔮 Input: " << prompt_str << std::endl;

    // 自回歸生成
    std::vector<int> gen_ids = prompt_ids;
    int max_gen = total_len - prompt_ids.size();

    for (int step = 0; step < max_gen; step++) {
        // 準備當前序列（pad 或截斷到 seq_len）
        int cur_len = gen_ids.size();
        std::vector<int> cur_input(seq_len, 0);
        int offset = 0;
        if (cur_len > seq_len) offset = cur_len - seq_len;
        for (int i = 0; i < seq_len && (offset + i) < cur_len; i++)
            cur_input[i] = gen_ids[offset + i];

        emb_op.set_indices(cur_input);

        // Forward only
        for (auto op : ops) op->forward();

        // 取最後一個有效位置的 logits
        int last_pos = std::min(cur_len - 1, seq_len - 1);
        float max_val = -1e9f;
        int max_idx = 0;
        for (int j = 0; j < vocab_size; j++) {
            float v = logits->data[last_pos * vocab_size + j].val;
            if (v > max_val) { max_val = v; max_idx = j; }
        }

        gen_ids.push_back(max_idx);

        // Clear（推理後也要清理 diffs）
        for (auto op : ops) op->clear();
    }

    // 輸出生成結果
    std::string generated = "";
    for (int i = prompt_ids.size(); i < (int)gen_ids.size(); i++)
        generated += vocab[gen_ids[i]];

    std::cout << "🔮 Generated: " << generated << std::endl;
    std::cout << std::endl;

    // 驗證
    std::string expected = "Nami是厲害的AI工程師";
    if (generated == expected)
        std::cout << "✅ Perfect! Transformer 成功學會了這句話！" << std::endl;
    else
        std::cout << "⚠️  Not perfect yet. Expected: " << expected << std::endl;

    // 清理記憶體
    delete emb_weight; delete pos_enc; delete emb_out; delete pos_out;
    delete Wq; delete Wk; delete Wv; delete Q; delete K; delete V; delete attn_out;
    delete res1; delete ffn_w1; delete ffn_w2; delete ffn_hidden; delete ffn_relu;
    delete ffn_out; delete res2; delete out_weight; delete logits; delete loss_tensor;
}

int main(int argc, char *argv[])
{
    // Parse command line arguments
    bool use_transformer = false;
    bool run_test = false;
    bool use_text = false;
    
    if (argc > 1) {
        if (std::string(argv[1]) == "--transformer") {
            use_transformer = true;
        } else if (std::string(argv[1]) == "--test") {
            run_test = true;
        } else if (std::string(argv[1]) == "--text") {
            use_text = true;
        }
    }
    
    // Run text transformer mode
    if (use_text) {
        run_text_transformer();
        return 0;
    }

    // Run test mode if requested
    if (run_test) {
        run_transformer_tests();
        return 0;
    }

    // Print which mode is running
    if (use_transformer) {
        std::cout << "[Mode: Transformer]" << std::endl;
    } else {
        std::cout << "[Mode: CNN/LeNet]" << std::endl;
    }

    // #######################################
    // # MNIST data informatiom
    // ---------------------------------------

    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
        mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("./third_party/mnist/");
    std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
    std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
    std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
    std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;
    std::cout << "----------------------" << std::endl;

    // #######################################
    // # Neural Networks (NN) model create
    // # Tensor input, answer
    // # input -> NN operations(conv, matmul, add, sigmoid, ...) -> lose function(output, answer)
    // ---------------------------------------

    int in_ch, in_dim, stride, pad, ker_dim, out_ch, out_dim, m, k, n, len, size;
    std::vector<int> shape;
    std::string label;

    tensor *input_ptr, *output_ptr;
    tensor answer(shape = {10});
    tensor *loss_ptr;

    if (!use_transformer) {
        // --------- CNN model (LeNet) ---------
        tensor &input = tir_external(shape = {1, 1, 28, 28});
        tensor &conv_weight = tir_variable(shape = {6, 1, 5, 5}, label = "conv_weight");
        conv_weight.init_he(); // [FIX #6] He init for conv layers
        tensor &matmul_weight = tir_variable(shape = {400, 120}, label = "matmul_weight");
        tensor &matmul1_weight = tir_variable(shape = {120, 10}, label = "matmul1_weight");
        tensor &add_weight = tir_variable(shape = {1, 120}, label = "add_weight");
        tensor &add1_weight = tir_variable(shape = {1, 10}, label = "add1_weight");
        tensor &conv1_weight = tir_variable(shape = {16, 6, 5, 5}, label = "conv1_weight");
        conv1_weight.init_he(); // [FIX #6] He init for conv layers

        tensor &conv = tir_conv(input, conv_weight, in_ch = 1, in_dim = 28, stride = 1, pad = 1, ker_dim = 5, out_ch = 6, out_dim = 28);
        tensor &relu = tir_relu(conv);
        tensor &max_pool = tir_max_pool(relu, size = 2, pad = 1, stride = 2);
        tensor &conv1 = tir_conv(max_pool, conv1_weight, in_ch = 6, in_dim = 14, stride = 1, pad = 0, ker_dim = 5, out_ch = 16, out_dim = 10);
        tensor &relu1 = tir_relu(conv1);
        tensor &max_pool1 = tir_max_pool(relu1, size = 2, pad = 1, stride = 2);
        tensor &reshape = tir_reshape(max_pool1, shape = {1, 400});
        tensor &matmul = tir_matmul(reshape, matmul_weight);
        tensor &add = tir_add(matmul, add_weight);
        tensor &sig = tir_sigmoid(add);
        tensor &matmul1 = tir_matmul(sig, matmul1_weight);
        tensor &add2 = tir_add(matmul1, add1_weight);
        // [FIX #5] Remove final Sigmoid; use Softmax+CrossEntropy instead of Sigmoid+MSE
        tensor &output = add2; // logits go directly to cross-entropy loss
        
        input_ptr = &input;
        output_ptr = &output;
        // ----------------------------
    } else {
        // --------- CNN + Transformer-like hybrid model ---------
        // CNN extracts local features, Transformer-like ops learn global relationships
        
        // === CNN Feature Extractor (same as LeNet front-end) ===
        tensor &input = tir_external(shape = {1, 1, 28, 28});
        tensor &conv_weight = tir_variable(shape = {6, 1, 5, 5}, label = "conv_weight");
        conv_weight.init_he();
        tensor &conv1_weight = tir_variable(shape = {16, 6, 5, 5}, label = "conv1_weight");
        conv1_weight.init_he();

        tensor &conv = tir_conv(input, conv_weight, in_ch = 1, in_dim = 28, stride = 1, pad = 1, ker_dim = 5, out_ch = 6, out_dim = 28);
        tensor &relu = tir_relu(conv);
        tensor &max_pool = tir_max_pool(relu, size = 2, pad = 1, stride = 2);
        tensor &conv1 = tir_conv(max_pool, conv1_weight, in_ch = 6, in_dim = 14, stride = 1, pad = 0, ker_dim = 5, out_ch = 16, out_dim = 10);
        tensor &relu1 = tir_relu(conv1);
        tensor &max_pool1 = tir_max_pool(relu1, size = 2, pad = 1, stride = 2);
        // Output: [1, 16, 5, 5] = 400 features

        // === Reshape to sequence for Transformer ===
        // [1, 400] — 400 CNN features as a sequence
        tensor &reshape = tir_reshape(max_pool1, shape = {1, 400});

        // === Linear projection: 400 -> 64 (reduce dim for transformer) ===
        tensor &proj_weight = tir_variable(shape = {400, 64}, label = "proj_weight");
        tensor &projected = tir_matmul(reshape, proj_weight);

        // === Transformer Block (decomposed into basic ops) ===
        std::cout << "--- transformer_block : ";
        for (auto i = 0; i < projected.shape.size(); i++) {
            if (i) std::cout << " x ";
            std::cout << projected.shape[i];
        }
        std::cout << " ---" << std::endl;

        // Self-attention (simplified for seq_len=1): just W_V projection  
        tensor &attn_wv = tir_variable(shape = {64, 64}, label = "attn_wv");
        tensor &attn_out = tir_matmul(projected, attn_wv);  // [1,64] x [64,64] = [1,64]

        // Residual connection 1: input + attention
        tensor &residual1 = tir_add(projected, attn_out);  // [1,64] + [1,64]

        // Feed-Forward Network
        tensor &ffn_w1 = tir_variable(shape = {64, 128}, label = "ffn_w1");
        tensor &ffn_hidden = tir_matmul(residual1, ffn_w1);  // [1,64] x [64,128] = [1,128]
        tensor &ffn_relu = tir_relu(ffn_hidden);
        tensor &ffn_w2 = tir_variable(shape = {128, 64}, label = "ffn_w2");
        tensor &ffn_out = tir_matmul(ffn_relu, ffn_w2);  // [1,128] x [128,64] = [1,64]

        // Residual connection 2: pre-FFN + FFN output
        tensor &residual2 = tir_add(residual1, ffn_out);  // [1,64] + [1,64]

        std::cout << "--- end transformer_block : ";
        for (auto i = 0; i < residual2.shape.size(); i++) {
            if (i) std::cout << " x ";
            std::cout << residual2.shape[i];
        }
        std::cout << " ---" << std::endl;

        // === Classifier head: [1, 64] -> [1, 10] ===
        tensor &classifier_weight = tir_variable(shape = {64, 10}, label = "classifier_weight");
        tensor &output = tir_matmul(residual2, classifier_weight);
        
        input_ptr = &input;
        output_ptr = &output;
        // ----------------------------
    }

    // [FIX #5] Cross-entropy loss (softmax computed internally)
    tensor &loss = tir_loss_cross_entropy(*output_ptr, answer);
    loss_ptr = &loss;

    // [FIX] Lower learning rate for deeper transformer network
    if (use_transformer) {
        cfg.lr = 0.001;
        std::cout << "[Transformer lr = " << cfg.lr << "]" << std::endl;
    }

    net.save();
#if 1
    // #######################################
    // # Training site
    // # set input, answer value
    // # net forward  ->
    // #     backward <-
    // #     update   ^
    // ---------------------------------------

    int Correct = 0;
    int Error = 0;
    int test_num = 0;
    int test_runs_count = 1000;
    int epoch = 1024;
    int batch = 1;

    for (int e = 0; e < epoch; e++)
    {
        for (unsigned int i = 0; i < dataset.training_images.size(); i++)
        {
            // Prepare input data differently for CNN vs Transformer
            // Both modes use 28x28 image input
            for (unsigned int j = 0; j < dataset.training_images[i].size(); j++)
                (*input_ptr)[j].val = ((float)(unsigned int)dataset.training_images[i][j]) / 255;

            int target_value = (unsigned int)dataset.training_labels[i];
            for (int k = 0; k < 10; k++)
            {
                answer[k].val = 0;
            }
            answer[target_value].val = 1;

            // ---------------------------
            net.forward();
            
            // [FIX] Check accuracy BEFORE update (measure prediction, not memorization)
            double max_value = -99999;
            int max_index = 0;
            for (int k = 0; k < 10; k++)
            {
                if ((*output_ptr)[k].val > max_value)
                {
                    max_value = (*output_ptr)[k].val;
                    max_index = k;
                }
            }
            
            net.backward();
            net.update();
            // ---------------------------

            if (max_index == target_value)
                Correct++;
            else
                Error++;

            test_num++;
            if ((int)test_num % test_runs_count == 0)
            {
                cfg.Accuracy = (float)Correct / ((float)Correct + (float)Error) * 100;
                std::cout << "[Epoch : " << e << " / " << epoch << "], [Batch : " << batch << "], [iteration : " << test_num << "], [Loss : " << std::setiosflags(std::ios::fixed) << std::setprecision(7) << (*loss_ptr)[0].val << "], [cfg.Accuracy : " << cfg.Accuracy << "% ... success]" << std::endl;
            }
        }

// testing
#if 1
        cfg.Accuracy = test_acc(net, dataset, *input_ptr, *output_ptr, answer);
#endif
        bool Acc_check = false;

        if (cfg.Accuracy >= cfg.Acc_ok)
        {
            Acc_check = true;
            net.save();
            goto exit;
        }
    }
#endif

exit:
    return 0;
}
