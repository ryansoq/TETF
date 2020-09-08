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
        /*
        data = (node *)malloc(sizeof(node) * shape_size);
*/
        data.resize(shape_size);
        /*
        data = new node[shape_size];

        if (data == nullptr)
        {
            printf("MEM error ... \n");
            return;
        }
*/
        srand((int)time(0) + rand());
        for (int i = 0; i < shape_size; i++)
            data[i].val = rand() % 1000 * 0.001 - 0.5;
    }

    ~tensor()
    {
        /*
        if (data != 0)
            delete [] data;
        */
    }

    node &operator[](std::size_t idx)
    {
        return data[idx];
    }
};

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

class Conv_t : public opBase
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
    Conv_t(tensor &out, tensor &a, tensor &b, int mc, int mm, int nn, int stride, int pad, int ks, int out_c, int out_x, int out_y);
    void forward();
    void backward();
    void update();
};

Conv_t::Conv_t(tensor &out, tensor &a, tensor &b, int mc, int mm, int nn, int stride, int pad, int ks, int out_c, int out_x, int out_y)
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

void Conv_t::forward()
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
    /*
    for (int i = 0; i < m_out_c; i++) //2
    {
        for (int j = 0; j < m_out_y; j++) //1
        {
            for (int k = 0; i < m_out_x; k++) //2
            {
                //--------------------
                conv_out = 0;
                for (int m = 0; m < m_ks; m++)
                {
                    for (int n = 0; n < m_ks; n++)
                    {
                        // if-for implementation
                        in_row = stride * j + m - pad;
                        in_col = stride * k + n - pad;
                        if (in_row >= 0 && in_col >= 0 && in_row < m_out_x && in_col < m_out_y)
                        {
                            for (l = 0; l < in_tensor_ch; l++)
                            {
                                mul_acc(&(*output)[i * m_out_y + j], &(*input1)[(in_col) * m_m + (in_row)], &(*input2)[m * m_ks + n]);
                            }
                        }
                    }
                }
            }
        }
    }
*/
}

void Conv_t::backward()
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

void Conv_t::update()
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

class Conv : public opBase
{
public:
    node *output;
    node *input1;
    node *input2;
    int m_stride;
    int m_ks;
    int m_m;
    int m_n;
    int m_out_x;
    int m_out_y;
    Conv(node *out, node *a, node *b, int m, int n, int stride, int ks, int out_x, int out_y);
    void forward();
    void backward();
    void update();
};

Conv::Conv(node *out, node *a, node *b, int m, int n, int stride, int ks, int out_x, int out_y)
{
    output = out;
    input1 = a;
    input2 = b;
    m_m = m;
    m_n = n;
    m_ks = ks;
    m_stride = stride;
    m_out_x = out_x;
    m_out_y = out_y;
}

void Conv::forward()
{
    for (int i = 0; i < m_out_x; i++) //2
    {
        for (int j = 0; j < m_out_y; j++) //1
        {
            //--------------------
            output[i * m_out_y + j].val = 0;
            int offsetY = i * m_stride;
            int offsetX = j * m_stride;

            for (int y = 0; y < m_ks; y++)
            {
                for (int x = 0; x < m_ks; x++)
                {
                    //output[i * m_out_y + j].val += mul_diff(&input1[(y + offsetY) * m_n + (x + offsetX)], &input2[y * m_ks + x], &output[i * m_out_y + j]);

                    output[i * m_out_y + j].val = output[i * m_out_y + j].val + input1[(y + offsetY) * m_n + (x + offsetX)].val * input2[y * m_ks + x].val;
                    input1[(y + offsetY) * m_n + (x + offsetX)].setDiff(input2[y * m_ks + x].val, &output[i * m_out_y + j]);
                    input2[y * m_ks + x].setDiff(input1[(y + offsetY) * m_n + (x + offsetX)].val, &output[i * m_out_y + j]);
                }
            }
        }
    }
}

void Conv::backward()
{
    int m = m_m;
    int n = m_n;
    int ks = m_ks;
    node *x = input1;
    node *w = input2;

    for (int i = 0; i < m * n; i++)
    {
        for (int j = 0; j < x[i].diffs.size(); j++)
        {
            x[i].diff += x[i].diffs[j].first * x[i].diffs[j].second->diff;
        }
    }

    for (int i = 0; i < ks * ks; i++)
    {
        for (int j = 0; j < w[i].diffs.size(); j++)
        {
            w[i].diff += w[i].diffs[j].first * w[i].diffs[j].second->diff;
        }
    }
}

void Conv::update()
{
    int m = m_m;
    int n = m_n;
    int ks = m_ks;
    node *x = input1;
    node *w = input2;

    for (int i = 0; i < m * n; i++)
    {
        x[i].val = x[i].val - lr * x[i].diff;
        x[i].diff = 0;
        x[i].diffs.clear();
    }

    for (int i = 0; i < ks * ks; i++)
    {
        w[i].val = w[i].val - lr * w[i].diff;
        w[i].diff = 0;
        w[i].diffs.clear();
    }
}

class Matmul_t : public opBase
{
public:
    tensor *output;
    tensor *input1;
    tensor *input2;
    int m_m;
    int m_k;
    int m_n;
    Matmul_t(tensor &out, tensor &a, tensor &b, int m, int k, int n);
    void forward();
    void backward();
    void update();
};

Matmul_t::Matmul_t(tensor &out, tensor &a, tensor &b, int m, int k, int n)
{
    output = &out;
    input1 = &a;
    input2 = &b;
    m_m = m;
    m_k = k;
    m_n = n;
}

void Matmul_t::forward()
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

void Matmul_t::backward()
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

void Matmul_t::update()
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

class Matmul : public opBase
{
public:
    node *output;
    node *input1;
    node *input2;
    int m_m;
    int m_k;
    int m_n;
    Matmul(node *out, node *a, node *b, int m, int k, int n);
    void forward();
    void backward();
    void update();
};

Matmul::Matmul(node *out, node *a, node *b, int m, int k, int n)
{
    output = out;
    input1 = a;
    input2 = b;
    m_m = m;
    m_k = k;
    m_n = n;
}

void Matmul::forward()
{
    int m = m_m;
    int n = m_n;
    int k = m_k;
    node *out = output;
    node *a = input1;
    node *b = input2;

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
    node *x = input1;
    node *w = input2;

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
}

void Matmul::update()
{
    int m = m_m;
    int n = m_n;
    int k = m_k;
    node *x = input1;
    node *w = input2;

    for (int i = 0; i < m * k; i++)
    {
        x[i].val = x[i].val - lr * x[i].diff;
        x[i].diff = 0;
        x[i].diffs.clear();
    }

    for (int i = 0; i < k * n; i++)
    {
        w[i].val = w[i].val - lr * w[i].diff;
        w[i].diff = 0;
        w[i].diffs.clear();
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

class Add_t : public opBase
{
public:
    tensor *output;
    tensor *input1;
    tensor *input2;
    int length;

    Add_t(tensor &out, tensor &a, tensor &b, int len);
    void forward();
    void backward();
    void update();
};

Add_t::Add_t(tensor &out, tensor &a, tensor &b, int len)
{
    output = &out;
    input1 = &a;
    input2 = &b;
    length = len;
}

void Add_t::forward()
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

void Add_t::backward()
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

void Add_t::update()
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

class Add : public opBase
{
public:
    node *output;
    node *input1;
    node *input2;
    int length;

    Add(node *out, node *a, node *b, int len);
    void forward();
    void backward();
    void update();
};

Add::Add(node *out, node *a, node *b, int len)
{
    output = out;
    input1 = a;
    input2 = b;
    length = len;
}

void Add::forward()
{
    node *out = output;
    node *a = input1;
    node *b = input2;

    for (int i = 0; i < length; i++)
    {
        //out[i].val = a[i].val + b[i].val;
        out[i].val = add_diff(&a[i], &b[i], &out[i]);
        /*
        a[i].setDiff(1, &out[i]);
        b[i].setDiff(1, &out[i]);
*/
    }
}

void Add::backward()
{
    node *x = input1;
    node *w = input2;

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
}

void Add::update()
{
    node *x = input1;
    node *w = input2;

    for (int i = 0; i < length; i++)
    {
        x[i].val = x[i].val - lr * x[i].diff;
        x[i].diff = 0;
        x[i].diffs.clear();
    }

    for (int i = 0; i < length; i++)
    {
        w[i].val = w[i].val - lr * w[i].diff;
        w[i].diff = 0;
        w[i].diffs.clear();
    }
}

class ReLU_t : public opBase
{
public:
    tensor output;
    tensor input1;
    int length;
    ReLU_t(tensor &out, tensor &a, int length);
    void forward();
    void backward();
    void update();
};

ReLU_t::ReLU_t(tensor &out, tensor &a, int len)
{
    output = out;
    input1 = a;
    length = len;
}

void ReLU_t::forward()
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

void ReLU_t::backward()
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

void ReLU_t::update()
{
    tensor &x = input1;

    for (int i = 0; i < length; i++)
    {
        x[i].val = x[i].val - lr * x[i].diff;
        x[i].diff = 0;
        x[i].diffs.clear();
    }
}

class ReLU : public opBase
{
public:
    node *output;
    node *input1;
    int length;
    ReLU(node *out, node *a, int length);
    void forward();
    void backward();
    void update();
};

ReLU::ReLU(node *out, node *a, int len)
{
    output = out;
    input1 = a;
    length = len;
}

void ReLU::forward()
{
    node *out = output;
    node *a = input1;

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
    node *x = input1;

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
    node *x = input1;

    for (int i = 0; i < length; i++)
    {
        x[i].val = x[i].val - lr * x[i].diff;
        x[i].diff = 0;
        x[i].diffs.clear();
    }
}

class Sigmoid_t : public opBase
{
public:
    tensor *output;
    tensor *input1;
    int length;
    Sigmoid_t(tensor &out, tensor &a, int length);
    void forward();
    void backward();
    void update();
};

Sigmoid_t::Sigmoid_t(tensor &out, tensor &a, int len)
{
    output = &out;
    input1 = &a;
    length = len;
}

void Sigmoid_t::forward()
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

void Sigmoid_t::backward()
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

void Sigmoid_t::update()
{
    tensor &x = *input1;

    for (int i = 0; i < length; i++)
    {
        x[i].val = x[i].val - lr * x[i].diff;
        x[i].diff = 0;
    }
}

class Sigmoid : public opBase
{
public:
    node *output;
    node *input1;
    int length;
    Sigmoid(node *out, node *a, int length);
    void forward();
    void backward();
    void update();
};

Sigmoid::Sigmoid(node *out, node *a, int len)
{
    output = out;
    input1 = a;
    length = len;
}

void Sigmoid::forward()
{
    node *out = output;
    node *a = input1;

    for (int i = 0; i < length; i++)
    {
        out[i].val = 1.0 / (1.0 + exp(-a[i].val));
        a[i].setDiff(out[i].val * (1 - out[i].val), &out[i]);
    }
}

void Sigmoid::backward()
{
    node *x = input1;

    for (int i = 0; i < length; i++)
    {
        for (int j = 0; j < x[i].diffs.size(); j++)
        {
            x[i].diff += x[i].diffs[j].first * x[i].diffs[j].second->diff;
        }
    }
}

void Sigmoid::update()
{
    node *x = input1;

    for (int i = 0; i < length; i++)
    {
        x[i].val = x[i].val - lr * x[i].diff;
        x[i].diff = 0;
        x[i].diffs.clear();
    }
}

class Loss_MSE_t : public opBase
{
public:
    tensor *output;
    tensor *input1;
    tensor *input2;
    int length;

    Loss_MSE_t(tensor &out, tensor &a, tensor &b, int len);
    void forward();
    void backward();
    void update();
};

Loss_MSE_t::Loss_MSE_t(tensor &out, tensor &a, tensor &b, int len)
{
    output = &out;
    input1 = &a;
    input2 = &b;
    length = len;
}

void Loss_MSE_t::forward()
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

void Loss_MSE_t::backward()
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

void Loss_MSE_t::update()
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

class Loss_MSE : public opBase
{
public:
    node *output;
    node *input1;
    node *input2;
    int length;

    Loss_MSE(node *out, node *a, node *b, int len);
    void forward();
    void backward();
    void update();
};

Loss_MSE::Loss_MSE(node *out, node *a, node *b, int len)
{
    output = out;
    input1 = a;
    input2 = b;
    length = len;
}

void Loss_MSE::forward()
{
    node *loss = output;
    node *src = input1;
    node *dest = input2;
    int m = length;

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

void Loss_MSE::backward()
{
    node *x = input1;
    node *w = input2;

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
}

void Loss_MSE::update()
{
    node *x = input1;
    node *w = input2;

    for (int i = 0; i < length; i++)
    {
        x[i].val = x[i].val - lr * x[i].diff;
        x[i].diff = 0;
        x[i].diffs.clear();
    }

    for (int i = 0; i < length; i++)
    {
        w[i].val = w[i].val - lr * w[i].diff;
        w[i].diff = 0;
        w[i].diffs.clear();
    }

    output->diff = 0;
    output->diffs.clear();
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

class External_t : public opBase
{
public:
    tensor *output;
    tensor keep;
    std::vector<int> shape;
    int length;
    External_t(tensor &out, std::vector<int> shape_);
    void forward();
    void backward();
    void update();
};

External_t::External_t(tensor &out, std::vector<int> shape_)
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

void External_t::forward()
{
    //tensor &x = *output;
    //*output = keep;
    /*
    tensor &x = keep;
    std::cout << "keep[0].val : " << keep[0].val << std::endl;
    std::cout << "keep[1].val : " << keep[1].val << std::endl;
    std::cout << "keep[2].val : " << keep[2].val << std::endl;
    std::cout << "keep[3].val : " << keep[3].val << std::endl;
    

    x[0].val = 0.4; x[1].val = 0.5; x[2].val = 0.6;
    x[3].val = 0.7; x[4].val = 0.8; x[5].val = 0.9;
    x[6].val = 1.0; x[7].val = 1.1; x[8].val = 1.2;
*/
    //...
}

void External_t::backward()
{
    //...
}

void External_t::update()
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

#if 0
float Testing(Net &net, int test_runs_count)
{
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
    mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("./third_party/mnist/");

    std::vector<float> Value;
    Value.resize(10);

    int test_num = 0;
    int test_runs= test_runs_count;
    int test_success_count=0;

    for (unsigned int i = 0; i < dataset.test_images.size(); i++)
    {
        for (unsigned int j = 0; j < dataset.test_images[i].size(); j++)
            x[i].val = ((float)(unsigned int)dataset.training_images[i][j]) / 255;

        //initialize the label
        int target_value = (unsigned int)dataset.test_labels[i];
        for (int k = 0; k < 10; k++)
        {
            ans[k].val = 0;
        }
        ans[target_value].val = 1;

        net.forward();

		//get the ouput and compare with the targe
		double max_value = -99999;
		int max_index = 0;
		for (int k = 0; k < 10; k++)
        {
			if (o2[k].val > max_value)
            {
				max_value = o2[k].val;
				max_index = k;
			}
		}

        //output == target
		if (ans[max_index].val == 1)
        {
			test_success_count ++;
		}

		test_num ++;

		if ((int)test_num % test_runs_count == 0)
            return (float)test_success_count / (float)test_runs_count * 100;

		if (test_num>=test_runs) break;
    }
    return (float)test_success_count / (float)test_runs_count * 100;
}
#endif

// Wrapper
// Conv_t(tensor &out, tensor &a, tensor &b, int mc, int mm, int nn, int stride, int pad, int ks, int out_c, int out_x, int out_y);
tensor &W_CONV(Net &net, tensor &in_tensor, int in_ch, int in_dim, int stride, int pad, int ker_dim, int out_ch, int out_dim)
{
    std::vector<int> shape;
    tensor *w = new tensor(shape = {out_ch, ker_dim, ker_dim});
    tensor *out_tensor = new tensor(shape = {out_ch, out_dim, out_dim});
    Conv_t *conv = new Conv_t(*out_tensor, in_tensor, *w, in_ch, in_dim, in_dim, stride, pad, ker_dim, out_ch, out_dim, out_dim);
    net.AddLayer(conv);
    return *out_tensor;
}
// m*k k*n -> m*n
tensor &W_MATMUL(Net &net, tensor &mk, int m, int k, int n)
{
    std::vector<int> shape;
    mk.shape.resize(2);
    mk.shape = {m, k};
    tensor *kn = new tensor(shape = {k, n});
    tensor *out_tensor = new tensor(shape = {m, n});
    Matmul_t *matmul = new Matmul_t(*out_tensor, mk, *kn, m, k, n);
    net.AddLayer(matmul);
    return *out_tensor;
}

tensor &W_ADD(Net &net, tensor &in_tensor, int length)
{
    std::vector<int> shape;
    tensor *b = new tensor(shape = {in_tensor.shape[0], in_tensor.shape[1]});
    tensor *out_tensor = new tensor(shape = {in_tensor.shape[0], in_tensor.shape[1]});
    Add_t *add = new Add_t(*out_tensor, in_tensor, *b, length);
    net.AddLayer(add);
    return *out_tensor;
}

tensor &W_SIGMOID(Net &net, tensor &in_tensor, int length)
{
    std::vector<int> shape;
    tensor *out_tensor = new tensor(shape = {in_tensor.shape[0], in_tensor.shape[1]});
    Sigmoid_t *sigmoid = new Sigmoid_t(*out_tensor, in_tensor, length);
    net.AddLayer(sigmoid);
    return *out_tensor;
}

tensor &W_LOSE_MSE(Net &net, tensor &in_tensor, int length)
{
    std::vector<int> shape;
    tensor *out_tensor = new tensor(shape = {length});
    tensor *lose = new tensor(shape = {1});
    Loss_MSE_t *lose_mse = new Loss_MSE_t(*lose, in_tensor, *out_tensor, length);
    net.AddLayer(lose_mse);
    return *out_tensor;
}

int main()
{

#if 1
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
        mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("./third_party/mnist/");
    std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
    std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
    std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
    std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;
    std::cout << "----------------------" << std::endl;

    std::vector<int> shape;
    /*
    tensor x(shape = {1, 784});
    tensor w1(shape = {784, 100});
    tensor o1(shape = {100});
    tensor b1(shape = {100});
    tensor sig1(shape = {100});
    tensor sig_out1(shape = {100});

    tensor w2(shape = {100, 10});
    tensor o2(shape = {10});    
    tensor b2(shape = {10});
    tensor sig2(shape = {10});
    tensor sig_out2(shape = {10});

    tensor ans(shape = {10});
    tensor loss(shape = {1});
 

    Matmul_t mat_1(o1, x, w1, 1, 784, 100);
    Add_t add_1(sig1, o1, b1, 100);
    Sigmoid_t sig_1(sig_out1, sig1, 100);

    Matmul_t mat_2(o2, sig_out1, w2, 1, 100, 10);
    Add_t add_2(sig2, o2, b2, 10);
    Sigmoid_t sig_2(sig_out2, sig2, 10);

    Loss_MSE_t lose_mse_t(loss, sig_out2, ans, 10);
    //-----------------------------------------------------
    Net net;
    net.AddLayer(&mat_1);
    net.AddLayer(&add_1);
    net.AddLayer(&sig_1);

    net.AddLayer(&mat_2);
    net.AddLayer(&add_2);
    net.AddLayer(&sig_2);

    net.AddLayer(&lose_mse_t);
*/
    tensor cx(shape = {1, 28, 28});
    //tensor cw1(shape = {3, 3, 3});

    //tensor x(shape = {1, 784});
    //tensor w1(shape = {784, 100});
    //tensor o1(shape = {100});
    //tensor b1(shape = {100});
    //tensor sig1(shape = {100});
    //tensor sig_out1(shape = {100});

    //tensor w2(shape = {100, 10});
    //tensor o2(shape = {10});
    //tensor b2(shape = {10});
    //tensor sig2(shape = {10});
    //tensor sig_out2(shape = {10});

    //tensor ans(shape = {10});
    //tensor loss(shape = {1});
    //Conv_t(tensor &out, tensor &a, tensor &b, int mc, int mm, int nn, int stride, int pad, int ks, int out_c, int out_x, int out_y);
    //Conv_t conv_1(x, cx, cw1, 1, 28, 28, 1, 1, 3, 3, 28, 28);
    //Matmul_t mat_1(o1, x, w1, 1, 784, 100);
    //Add_t add_1(sig1, o1, b1, 100);
    //Sigmoid_t sig_1(sig_out1, sig1, 100);

    //Matmul_t mat_2(o2, sig_out1, w2, 1, 100, 10);
    //Add_t add_2(sig2, o2, b2, 10);
    //Sigmoid_t sig_2(sig_out2, sig2, 10);

    //Loss_MSE_t lose_mse_t(loss, sig_out2, ans, 10);
    //net.AddLayer(&conv_1);
    //net.AddLayer(&mat_1);
    //net.AddLayer(&add_1);
    //net.AddLayer(&sig_1);
    //net.AddLayer(&mat_2);
    //net.AddLayer(&add_2);
    //net.AddLayer(&sig_2);
    //net.AddLayer(&lose_mse_t);

    //-----------------------------------------------------
    Net net;
    int in_ch, in_dim, stride, pad, ker_dim, out_ch, out_dim, m, k, n, len;

    tensor &x = W_CONV(net, cx, in_ch = 1, in_dim = 28, stride = 1, pad = 1, ker_dim = 1, out_ch = 3, out_dim = 28);
    tensor &o1 = W_MATMUL(net, x, m = 1, k = 768, n = 100);
    tensor &sig1 = W_ADD(net, o1, len = 100);
    tensor &sig_out1 = W_SIGMOID(net, sig1, len = 100);
    tensor &o2 = W_MATMUL(net, sig_out1, m = 1, k = 100, n = 10);
    tensor &sig2 = W_ADD(net, o2, len = 10);
    tensor &sig_out2 = W_SIGMOID(net, sig2, len = 10);
    tensor &ans = W_LOSE_MSE(net, sig_out2, len = 10);

    int Correct = 0;
    int Error = 0;
    int test_num = 0;
    int test_runs_count = 1000;
    int epoch = 160;
    int batch = 1;
    float Acc_ok = 95.0;
    float Accuracy;

    for (int e = 0; e < epoch; e++)
    {
        for (unsigned int i = 0; i < dataset.training_images.size(); i++)
        {
            for (unsigned int j = 0; j < dataset.training_images[i].size(); j++)
                cx[j].val = ((float)(unsigned int)dataset.training_images[i][j]) / 255;

            int target_value = (unsigned int)dataset.training_labels[i];
            for (int k = 0; k < 10; k++)
            {
                ans[k].val = 0;
            }
            ans[target_value].val = 1;

            net.forward();

            double max_value = -99999;
            int max_index = 0;
            for (int k = 0; k < 10; k++)
            {
                if (sig_out2[k].val > max_value)
                {
                    max_value = sig_out2[k].val;
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
#endif
}
