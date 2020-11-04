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

// Weight 93-94%
//#include "weight.inc"
#include "ff90_weight.inc"

// Net class
float START_QUANTIZATION = 100.0;
float Accuracy;
float lr = 0.01;
float Acc_ok = 95.0;
int global_num = 0;
int tensor_num = 0;

typedef int8_t q7_t;
typedef uint8_t u8_t;
typedef int16_t q15_t;
typedef uint16_t u16_t;
typedef int32_t q31_t;
typedef int64_t q63_t;

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

        srand((int)time(0) + rand());
        for (int i = 0; i < shape_size; i++)
            data[i].val = rand() % 1000 * 0.001 - 0.5;
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

// Recursive function
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

// input1/input2
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
    void save(){};
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

    std::cout << nnCode << std::endl;
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
    void save(){};
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
    nnCode.append("\"" + pSave_path + "\"");

    nnCode.append(");\n");

    std::cout << nnCode << std::endl;
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
    void save(){};
};

Reshape::Reshape(tensor &out, tensor &a, std::vector<int> p_shape)
{
    output = &a;
    input1 = &a;
    shape = p_shape;
}

void Reshape::forward()
{
    tensor &out = *output;
    tensor &a = *input1;
    out.shape = shape;
}

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
    void save(){};
};

Max_pool::Max_pool(tensor &out, tensor &a, int size, int pad, int stride)
{
    output = &out;
    input1 = &a;
    m_size = size;
    m_pad = pad;
    m_stride = stride;
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
                    int index_;
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
    int size = m_size;
    int stride = m_stride;
    int padding = m_pad;
    tensor &x = *input1;

    for (int i = 0; i < x.data.size(); i++)
    {
        if (Accuracy > START_QUANTIZATION)
        {
            x[i].f2q();
            x[i].q2f();
        }
        x[i].val = x[i].val - lr * x[i].diff;
        x[i].diff = 0;
        x[i].diffs.clear();
    }
}

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
    int ret = TYPE4_FORWARD_conv_CHW(Im_out, Im_in, filter, bias = 0.0, pad, stride, group = 1);
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
    int ret = TYPE4_BACKWARD_conv_CHW(Im_out, Im_in, filter, bias = 0, pad, stride, group = 1);
#else
    int c = m_c;
    int m = m_m;
    int n = m_n;
    int ks = m_ks;
    tensor &x = *input1;
    tensor &w = *input2;

    assert(x.data.size() == (c * m * n));
    assert(w.data.size() == (m_out_c * ks * ks));

    for (int i = 0; i < c * m * n; i++)
    {
        for (int j = 0; j < x[i].diffs.size(); j++)
        {
            x[i].diff += x[i].diffs[j].first * x[i].diffs[j].second->diff;
        }
    }

    for (int i = 0; i < m_out_c * ks * ks; i++)
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
    assert(w.data.size() == (m_out_c * ks * ks));
    /*
    if (Accuracy > Acc_ok)
    {
        unsigned char *ptr;
        bool dump;
        std::vector<float> temp;
        for (auto i = 0; i < w.data.size(); i++)
            temp.push_back(w.data[i].val);
        float2uc(temp.data(), &ptr, w.data.size() * sizeof(float), dump = true, "CONV" + std::to_string(global_num));
        global_num++;
    }
*/
    for (int i = 0; i < c * m * n; i++)
    {
        if (Accuracy > START_QUANTIZATION)
        {
            x[i].f2q();
            x[i].q2f();
        }
        x[i].val = x[i].val - lr * x[i].diff;
        x[i].diff = 0;
        x[i].diffs.clear();
    }

    for (int i = 0; i < m_out_c * ks * ks; i++)
    {
        if (Accuracy > START_QUANTIZATION)
        {
            w[i].f2q();
            w[i].q2f();
        }
        w[i].val = w[i].val - lr * w[i].diff;
        w[i].diff = 0;
        w[i].diffs.clear();
    }
}

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

    nnCode.append(");\n");

    std::cout << nnCode << std::endl;
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
    if (Accuracy > Acc_ok)
    {
        unsigned char *ptr;
        bool dump;
        std::vector<float> temp;
        for (auto i = 0; i < w.data.size(); i++)
            temp.push_back(w.data[i].val);
        float2uc(temp.data(), &ptr, w.data.size() * sizeof(float), dump = true, "MATMUL" + std::to_string(global_num));
        global_num++
    }
*/
    for (int i = 0; i < m * k; i++)
    {
        if (Accuracy > START_QUANTIZATION)
        {
            x[i].f2q();
            x[i].q2f();
        }
        x[i].val = x[i].val - lr * x[i].diff;
        x[i].diff = 0;
    }

    for (int i = 0; i < k * n; i++)
    {
        if (Accuracy > START_QUANTIZATION)
        {
            w[i].f2q();
            w[i].q2f();
        }
        w[i].val = w[i].val - lr * w[i].diff;
        w[i].diff = 0;
    }
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

    std::cout << nnCode << std::endl;
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
    if (Accuracy > Acc_ok)
    {
        unsigned char *ptr;
        bool dump;
        std::vector<float> temp;
        for (auto i = 0; i < w.data.size(); i++)
            temp.push_back(w.data[i].val);
        float2uc(temp.data(), &ptr, w.data.size() * sizeof(float), dump = true, "ADD" + std::to_string(global_num));
        global_num++
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
        if (Accuracy > START_QUANTIZATION)
        {
            x[i].f2q();
            x[i].q2f();
        }
        x[i].val = x[i].val - lr * x[i].diff;
        x[i].diff = 0;
    }

    for (int i = 0; i < length; i++)
    {
        if (Accuracy > START_QUANTIZATION)
        {
            w[i].f2q();
            w[i].q2f();
        }
        w[i].val = w[i].val - lr * w[i].diff;
        w[i].diff = 0;
    }
}

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
};

ReLU::ReLU(tensor &out, tensor &a, int len)
{
    output = &out;
    input1 = &a;
    length = len;
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
        if (Accuracy > START_QUANTIZATION)
        {
            x[i].f2q();
            x[i].q2f();
        }
        x[i].val = x[i].val - lr * x[i].diff;
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
        if (Accuracy > START_QUANTIZATION)
        {
            x[i].f2q();
            x[i].q2f();
        }
        x[i].val = x[i].val - lr * x[i].diff;
        x[i].diff = 0;
        x[i].diffs.clear();
    }
}

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

    std::cout << nnCode << std::endl;
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
        if (Accuracy > START_QUANTIZATION)
        {
            x[i].f2q();
            x[i].q2f();
        }
        x[i].val = x[i].val - lr * x[i].diff;
        x[i].diff = 0;
    }
}

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
        if (Accuracy > START_QUANTIZATION)
        {
            x[i].f2q();
            x[i].q2f();
        }
        x[i].val = x[i].val - lr * x[i].diff;
        x[i].diff = 0;
    }

    for (int i = 0; i < length; i++)
    {
        if (Accuracy > START_QUANTIZATION)
        {
            w[i].f2q();
            w[i].q2f();
        }
        w[i].val = w[i].val - lr * w[i].diff;
        w[i].diff = 0;
    }
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
};

class Net : public netBase
{
public:
    std::list<opBase *> Layer;
    void AddLayer(opBase *);
    void forward();
    void backward();
    void update();
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

void Net::print()
{
    for (std::list<opBase *>::reverse_iterator choose = Layer.rbegin(); choose != Layer.rend(); ++choose)
        (*choose)->print();
}

// #######################################
// # Wrapper function
// ---------------------------------------

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
    out_tensor->name = "external" + std::to_string(++tensor_num);
    External *external = new External(*out_tensor, p_shape);
    net.AddLayer(external);
    return *out_tensor;
}

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
    out_tensor->name = "variable" + std::to_string(++tensor_num);
    Variable *variable = new Variable(*out_tensor, p_shape, path);
    net.AddLayer(variable);
    return *out_tensor;
}

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

    tensor *out_tensor = &in_tensor;
    out_tensor->name = "reshape" + std::to_string(++tensor_num);
    out_tensor->shape = p_shape;
    Reshape *reshape = new Reshape(*out_tensor, in_tensor, p_shape);
    net.AddLayer(reshape);
    return *out_tensor;
}

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
    out_tensor->name = "conv" + std::to_string(++tensor_num);
    std::cout << "conv : " << n << " x " << c << " x " << h << " x " << w << std::endl;
    Conv *conv = new Conv(*out_tensor, in_tensor, filter, in_ch, in_dim, in_dim, stride, pad, ker_dim, out_ch, out_dim, out_dim);
    net.AddLayer(conv);
    return *out_tensor;
}

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
    out_tensor->name = "max_pool" + std::to_string(++tensor_num);
    std::cout << "max_pool : " << n << " x " << c << " x " << h << " x " << w << std::endl;
    //tensor *out_tensor = new tensor(shape = {out_ch, out_dim, out_dim});
    Max_pool *max_pool = new Max_pool(*out_tensor, in_tensor, p_size, p_padding, p_stride);
    net.AddLayer(max_pool);
    return *out_tensor;
}

// [m, k] * [k, n] -> [m, n]
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
    out_tensor->name = "matmul" + std::to_string(++tensor_num);
    Matmul *matmul = new Matmul(*out_tensor, mk, kn, mk.shape[0], mk.shape[1], kn.shape[1]);
    net.AddLayer(matmul);
    return *out_tensor;
}

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
    out_tensor->name = "add" + std::to_string(++tensor_num);
    Add *add = new Add(*out_tensor, in_tensor, weight, in_tensor_size);
    net.AddLayer(add);
    return *out_tensor;
}

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

    tensor *out_tensor = new tensor(shape = {in_tensor.shape[0], in_tensor.shape[1]});
    out_tensor->name = "sigmoid" + std::to_string(++tensor_num);
    Sigmoid *sigmoid = new Sigmoid(*out_tensor, in_tensor, in_tensor_size);
    net.AddLayer(sigmoid);
    return *out_tensor;
}

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

    tensor *out_tensor = new tensor(shape = {in_tensor.shape[0], in_tensor.shape[1]});
    out_tensor->name = "relu" + std::to_string(++tensor_num);
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
    out_tensor->name = "leaky_relu" + std::to_string(++tensor_num);
    Leaky_ReLU *leaky_relu = new Leaky_ReLU(*out_tensor, in_tensor, in_tensor_size);
    net.AddLayer(leaky_relu);
    return *out_tensor;
}

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
    loss->name = "loss_mse" + std::to_string(++tensor_num);
    Loss_MSE *loss_mse = new Loss_MSE(*loss, in_tensor, ans, ans.data.size());
    net.AddLayer(loss_mse);
    return *loss;
}

Net net;

int main()
{
    // #######################################
    // # Tesing section
    // ---------------------------------------
    // ...
    //std::vector<int> shape;
    //tensor input(shape = {28, 28, 1});

#if 1
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
    //tensor input(shape = {28, 28, 1});

    //tensor *conv_weight;
    //tensor *matmul_weight;
    //tensor *add_weight;
    //tensor *conv1_weight;
    //tensor *matmul1_weight;
    //tensor *add1_weight;

    // --------- NN model ---------
    tensor &input = tir_external(shape = {1, 784});
    //tensor &conv_weight = tir_variable(shape = {10, 1, 3, 3}, label = "weights");
    tensor &matmul_weight = tir_variable(shape = {784, 100}, label = "matmul_weight");
    tensor &matmul1_weight = tir_variable(shape = {100, 10}, label = "matmul1_weight");
    tensor &add_weight = tir_variable(shape = {1, 100}, label = "add_weight");
    tensor &add1_weight = tir_variable(shape = {1, 10}, label = "add1_weight");
    //tensor &conv1_weight = tir_variable(shape = {1, 1, 3, 3}, label = "weights1");

    //tensor &x = tir_conv(input, conv_weight, in_ch = 1, in_dim = 28, stride = 1, pad = 1, ker_dim = 3, out_ch = 10, out_dim = 28);
    //tensor &max_pool = tir_max_pool(x, size = 2, pad = 1, stride = 2);
    //tensor &reshape = tir_reshape(max_pool, shape = {1, 1960});
    tensor &o1 = tir_matmul(input, matmul_weight);
    tensor &sig1 = tir_add(o1, add_weight);
    tensor &sig_out1 = tir_sigmoid(sig1);
    //tensor &x1 = tir_conv(sig_out1, conv1_weight, in_ch = 1, in_dim = 10, stride = 1, pad = 1, ker_dim = 3, out_ch = 1, out_dim = 10);
    tensor &o2 = tir_matmul(sig_out1, matmul1_weight);
    tensor &sig2 = tir_add(o2, add1_weight);
    tensor &output = tir_sigmoid(sig2);
    // ----------------------------

    // Mean square error
    tensor answer(shape = {10});
    tensor &loss = tir_loss_mse(output, answer);

    matmul_weight.load_uc2f(w_matmul_weight);
    add_weight.load_uc2f(w_add_weight);
    matmul1_weight.load_uc2f(w_matmul1_weight);
    add1_weight.load_uc2f(w_add1_weight);

    // #######################################
    // # Training site
    // # set input, answer value
    // # net forward  ->
    // #     backward <-
    // #     update
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
            for (unsigned int j = 0; j < dataset.training_images[i].size(); j++)
                input[j].val = ((float)(unsigned int)dataset.training_images[i][j]) / 255;

            int target_value = (unsigned int)dataset.training_labels[i];
            for (int k = 0; k < 10; k++)
            {
                answer[k].val = 0;
            }
            answer[target_value].val = 1;

            net.forward();

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
                Correct++;
            else
                Error++;

            net.backward();
            net.update();

            test_num++;
            if ((int)test_num % test_runs_count == 0)
            {
                Accuracy = (float)Correct / ((float)Correct + (float)Error) * 100;
                std::cout << "[Epoch : " << e << " / " << epoch << "], [Batch : " << batch << "], [iteration : " << test_num << "], [Loss : " << std::setiosflags(std::ios::fixed) << std::setprecision(7) << loss[0].val << "], [Accuracy : " << Accuracy << "% ... success]" << std::endl;
            }
        }

        bool Acc_check = false;

        if (Accuracy >= Acc_ok)
        {

            printf("In check ... ok\n");
            Acc_check = true;
            matmul_weight.save_f2uc("matmul_weight");
            add_weight.save_f2uc("add_weight");
            matmul1_weight.save_f2uc("matmul1_weight");
            add1_weight.save_f2uc("add1_weight");
            goto exit;
        }
    }
#endif
exit:
    return 0;
}
