#include <iostream>
#include <cmath>
#include <unistd.h>
#include <vector>
#include <string>
#include <list>
#include <mnist/mnist_reader.hpp>

// Net class
float lr = 0.1;

class node
{
public:
    static int nextID;
    std::string id;
    float val;
    float diff;
    std::vector<std::pair<float, node *>> diffs;

    node();
    void printDiff(void);
    void setDiff(float dfdx, node *dldf);
};

int node::nextID = 0;

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
    std::vector<int> strides;
    uint64_t byte_offset;

    tensor()
    {
        //data = 0;
    }

    tensor(std::vector<int> _shape)
    {
        shape = _shape;
        ndim = shape.size();
        int shape_size = 1;

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
    virtual void print()
    {
        std::cout << "print, Base" << std::endl;
    }
};

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

void Conv::forward()
{
    const uint16_t in_tensor_dim = m_m;
    const uint16_t in_tensor_ch = m_c;
    const uint16_t out_tensor_ch = m_out_c;
    const uint16_t ker_dim = m_ks;
    const uint16_t pad = m_pad;
    const uint16_t stride = m_stride;
    const uint16_t out_tensor_dim = m_out_x;
    uint16_t i, j, k, l, m, n;
    long in_row, in_col;
    //NCHW
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
}

void Conv::backward()
{
    int c = m_c;
    int m = m_m;
    int n = m_n;
    int ks = m_ks;
    tensor &x = *input1;
    tensor &w = *input2;

    for (int i = 0; i < c * m * n; i++)
    {
        for (int j = 0; j < x[i].diffs.size(); j++)
        {
            x[i].diff += x[i].diffs[j].first * x[i].diffs[j].second->diff;
        }
    }

    //printf("conv->weight : ");
    for (int i = 0; i < m_out_c * ks * ks; i++)
    {
        for (int j = 0; j < w[i].diffs.size(); j++)
        {
            w[i].diff += w[i].diffs[j].first * w[i].diffs[j].second->diff;
        }
        //if (w[i].diff != 0.0)
        //	printf("%f ", w[i].diff);
    }
    //printf("\n");
}

void Conv::update()
{
    int c = m_c;
    int m = m_m;
    int n = m_n;
    int ks = m_ks;
    tensor &x = *input1;
    tensor &w = *input2;

    for (int i = 0; i < c * m * n; i++)
    {
        x[i].val = x[i].val - lr * x[i].diff;
        x[i].diff = 0;
        x[i].diffs.clear();
    }

    for (int i = 0; i < m_out_c * ks * ks; i++)
    {
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
}

void Matmul::forward()
{
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
}

void Matmul::backward()
{
    int m = m_m;
    int n = m_n;
    int k = m_k;
    tensor &x = *input1;
    tensor &w = *input2;

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

    for (int i = 0; i < m * k; i++)
    {
        x[i].val = x[i].val - lr * x[i].diff;
        x[i].diff = 0;
    }

    for (int i = 0; i < k * n; i++)
    {
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

    for (int i = 0; i < length; i++)
    {
        x[i].val = x[i].val - lr * x[i].diff;
        x[i].diff = 0;
    }

    for (int i = 0; i < length; i++)
    {
        w[i].val = w[i].val - lr * w[i].diff;
        w[i].diff = 0;
    }
}

class ReLU : public opBase
{
public:
    tensor output;
    tensor input1;
    int length;
    ReLU(tensor &out, tensor &a, int length);
    void forward();
    void backward();
    void update();
};

ReLU::ReLU(tensor &out, tensor &a, int len)
{
    output = out;
    input1 = a;
    length = len;
}

void ReLU::forward()
{
    tensor &out = output;
    tensor &a = input1;

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
    tensor &x = input1;

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
    tensor &x = input1;

    for (int i = 0; i < length; i++)
    {
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

    for (int i = 0; i < length; i++)
    {
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
    for (int i = 0; i < length; i++)
    {
        x[i].val = x[i].val - lr * x[i].diff;
        x[i].diff = 0;
    }

    for (int i = 0; i < length; i++)
    {
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

class External : public opBase
{
public:
    tensor *output;
    tensor keep;
    std::vector<int> shape;
    int length;
    External(tensor &out, std::vector<int> shape_);
    void forward();
    void backward();
    void update();
};

External::External(tensor &out, std::vector<int> shape_)
{
    tensor tmp(shape_);
    shape = shape_;
    int ndim = tmp.shape.size();
    int shape_size = 1;

    for (int i = 0; i < ndim; i++)
        shape_size *= shape[i];
    for (int i = 0; i < shape_size; i++)
        tmp[i].val = rand() % 1000 * 0.001 - 0.5;

    out = tmp;
    keep = tmp;
    output = &out; //keep
}

void External::forward()
{
    //...
}

void External::backward()
{
    //...
}

void External::update()
{
    //...
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

tensor &W_CONV(Net &net, tensor &in_tensor, int in_ch, int in_dim, int stride, int pad, int ker_dim, int out_ch, int out_dim)
{
    std::vector<int> shape;
    tensor *w = new tensor(shape = {out_ch, ker_dim, ker_dim});
    tensor *out_tensor = new tensor(shape = {out_ch, out_dim, out_dim});
    Conv *conv = new Conv(*out_tensor, in_tensor, *w, in_ch, in_dim, in_dim, stride, pad, ker_dim, out_ch, out_dim, out_dim);
    net.AddLayer(conv);
    return *out_tensor;
}
// [m, k] * [k, n] -> [m, n]
tensor &W_MATMUL(Net &net, tensor &mk, int m, int k, int n)
{
    std::vector<int> shape;
    mk.shape.resize(2);
    mk.shape = {m, k};
    tensor *kn = new tensor(shape = {k, n});
    tensor *out_tensor = new tensor(shape = {m, n});
    Matmul *matmul = new Matmul(*out_tensor, mk, *kn, m, k, n);
    net.AddLayer(matmul);
    return *out_tensor;
}

tensor &W_ADD(Net &net, tensor &in_tensor, int length)
{
    std::vector<int> shape;
    tensor *b = new tensor(shape = {in_tensor.shape[0], in_tensor.shape[1]});
    tensor *out_tensor = new tensor(shape = {in_tensor.shape[0], in_tensor.shape[1]});
    Add *add = new Add(*out_tensor, in_tensor, *b, length);
    net.AddLayer(add);
    return *out_tensor;
}

tensor &W_SIGMOID(Net &net, tensor &in_tensor, int length)
{
    std::vector<int> shape;
    tensor *out_tensor = new tensor(shape = {in_tensor.shape[0], in_tensor.shape[1]});
    Sigmoid *sigmoid = new Sigmoid(*out_tensor, in_tensor, length);
    net.AddLayer(sigmoid);
    return *out_tensor;
}

void W_LOSE_MSE(Net &net, tensor &in_tensor, tensor &ans)
{
    std::vector<int> shape;
    tensor *lose = new tensor(shape = {1});
    Loss_MSE *lose_mse = new Loss_MSE(*lose, in_tensor, ans, ans.data.size());
    net.AddLayer(lose_mse);
}

int main()
{
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

    Net net;
    int in_ch, in_dim, stride, pad, ker_dim, out_ch, out_dim, m, k, n, len;

    std::vector<int> shape;
    tensor input(shape = {1, 28, 28});
    tensor answer(shape = {10});

    tensor &x = W_CONV(net, input, in_ch = 1, in_dim = 28, stride = 1, pad = 1, ker_dim = 3, out_ch = 1, out_dim = 28);
    tensor &o1 = W_MATMUL(net, x, m = 1, k = 768, n = 100);
    tensor &sig1 = W_ADD(net, o1, len = 100);
    tensor &sig_out1 = W_SIGMOID(net, sig1, len = 100);
    tensor &o2 = W_MATMUL(net, sig_out1, m = 1, k = 100, n = 10);
    tensor &sig2 = W_ADD(net, o2, len = 10);
    tensor &output = W_SIGMOID(net, sig2, len = 10);

    W_LOSE_MSE(net, output, answer);

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
    int epoch = 160;
    int batch = 1;
    float Acc_ok = 99.0;
    float Accuracy;

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
                std::cout << "[Epoch : " << epoch << "], [Batch : " << batch << "], [iteration : " << test_num << "], [Accuracy : " << Accuracy << "% ... success]" << std::endl;
            }
        }
        if (Accuracy >= Acc_ok)
            break;
    }

    return 0;
}