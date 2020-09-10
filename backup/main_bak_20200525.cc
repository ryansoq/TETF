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
        float diff = 0;
        std::vector< std::pair<float, node*> > diffs;
      
        node();
        void printDiff(void);
        void setDiff(float dfdx, node * dldf);
};

int node::nextID = 0;

node::node() 
{
   std::string str = std::to_string(++nextID);
   id = "∂x" + str;
}

void node::setDiff(float dfdx, node * dldf)
{
    std::pair<float, node*> val;
    val.first = dfdx;
    val.second = dldf;
    diffs.push_back(val);
};

typedef struct {
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
    node * data = 0;
    int ndim;
    DLDataType dtype;
    std::vector<int> shape;
    std::vector<int> strides;
    uint64_t byte_offset;

    tensor()
    {
        data = 0;
    }
    
    tensor(std::vector<int> _shape)
    {
        shape = _shape;
        ndim = shape.size();
        int shape_size = 1;

        for (int i = 0; i < ndim; i++)
            shape_size *= shape[i];
        
        data = (node *)malloc(sizeof(node) * shape_size);

        srand((int)time(0) + rand());
        for (int i = 0; i < shape_size; i++)
            data[i].val = rand() % 1000 * 0.001 - 0.5;
    }
  
    ~tensor()
    {
        //if (data != 0)
            //free(data);
    }

    node& operator[](std::size_t idx)       
    { 
        return data[idx]; 
    }

};

float get_diff(node * src, node * dst)
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

void mul(node * output, node * input1, node * input2)
{
    std::pair<float, node*> diff1;
    diff1.first = input2->val;
    diff1.second = output;
    input1->diffs.push_back(diff1);

    std::pair<float, node*> diff2;
    diff2.first = input1->val;
    diff2.second = output;
    input2->diffs.push_back(diff2);

    output->val = input1->val * input2->val;
}

void mul_acc(node * output, node * input1, node * input2)
{
    std::pair<float, node*> diff1;
    diff1.first = input2->val;
    diff1.second = output;
    input1->diffs.push_back(diff1);

    std::pair<float, node*> diff2;
    diff2.first = input1->val;
    diff2.second = output;
    input2->diffs.push_back(diff2);

    output->val += input1->val * input2->val;
}

float mul_diff(node * input1, node * input2, node * output)
{
    std::pair<float, node*> diff1;
    diff1.first = input2->val;
    diff1.second = output;
    input1->diffs.push_back(diff1);

    std::pair<float, node*> diff2;
    diff2.first = input1->val;
    diff2.second = output;
    input2->diffs.push_back(diff2);

    return input1->val * input2->val;
}

void add(node * output, node * input1, node * input2)
{
    std::pair<float, node*> diff1;
    diff1.first = 1;
    diff1.second = output;
    input1->diffs.push_back(diff1);

    std::pair<float, node*> diff2;
    diff2.first = 1;
    diff2.second = output;
    input2->diffs.push_back(diff2);

    output->val = input1->val + input2->val;
}

void add_acc(node * output, node * input1, node * input2)
{
    std::pair<float, node*> diff1;
    diff1.first = 1;
    diff1.second = output;
    input1->diffs.push_back(diff1);

    std::pair<float, node*> diff2;
    diff2.first = 1;
    diff2.second = output;
    input2->diffs.push_back(diff2);

    output->val += input1->val + input2->val;
}

float add_diff(node * input1, node * input2, node * output)
{
    std::pair<float, node*> diff1;
    diff1.first = 1;
    diff1.second = output;
    input1->diffs.push_back(diff1);

    std::pair<float, node*> diff2;
    diff2.first = 1;
    diff2.second = output;
    input2->diffs.push_back(diff2);

    return input1->val + input2->val;
}

void sub(node * output, node * input1, node * input2)
{
    std::pair<float, node*> diff1;
    diff1.first = 1;
    diff1.second = output;
    input1->diffs.push_back(diff1);

    std::pair<float, node*> diff2;
    diff2.first = -1;
    diff2.second = output;
    input2->diffs.push_back(diff2);

    output->val = input1->val - input2->val;
}

void sub_acc(node * output, node * input1, node * input2)
{
    std::pair<float, node*> diff1;
    diff1.first = 1;
    diff1.second = output;
    input1->diffs.push_back(diff1);

    std::pair<float, node*> diff2;
    diff2.first = -1;
    diff2.second = output;
    input2->diffs.push_back(diff2);

    output->val += input1->val - input2->val;
}

// input1 - input2
float sub_diff(node * input1, node * input2, node * output)
{
    std::pair<float, node*> diff1;
    diff1.first = 1;
    diff1.second = output;
    input1->diffs.push_back(diff1);

    std::pair<float, node*> diff2;
    diff2.first = -1;
    diff2.second = output;
    input2->diffs.push_back(diff2);

    return input1->val - input2->val;
}

// input1/input2
// https://zs.symbolab.com/solver/partial-derivative-calculator/%20%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%20y%7D%5Cleft(%5Cfrac%7Bx%7D%7By%7D%5Cright)
float div_diff(node * input1, node * input2, node * output)
{
    std::pair<float, node*> diff1;
    diff1.first = 1 / input2->val;
    diff1.second = output;
    input1->diffs.push_back(diff1);

    std::pair<float, node*> diff2;
    diff2.first = -1 * (input1->val / (input2->val * input2->val));
    diff2.second = output;
    input2->diffs.push_back(diff2);

    return input1->val / input2->val;
}

void div(node * output, node * input1, node * input2)
{
    std::pair<float, node*> diff1;
    diff1.first = 1 / input2->val;
    diff1.second = output;
    input1->diffs.push_back(diff1);

    std::pair<float, node*> diff2;
    diff2.first = -1 * (input1->val / (input2->val * input2->val));
    diff2.second = output;
    input2->diffs.push_back(diff2);

    output->val = input1->val / input2->val;
}

void div_acc(node * output, node * input1, node * input2)
{
    std::pair<float, node*> diff1;
    diff1.first = 1 / input2->val;
    diff1.second = output;
    input1->diffs.push_back(diff1);

    std::pair<float, node*> diff2;
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
    tensor output;
    tensor input1;
    tensor input2;
    int m_stride;
    int m_ks;
    int m_m;
    int m_n;
    int m_out_x;
    int m_out_y;
    Conv_t(tensor &out, tensor &a, tensor &b, int m, int n, int stride, int ks, int out_x, int out_y);
    void forward();
    void backward();
    void update();
};

Conv_t::Conv_t(tensor &out, tensor &a, tensor &b, int m, int n, int stride, int ks, int out_x, int out_y)
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

void Conv_t::forward()
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
                    mul_acc(&output[i * m_out_y + j], &input1[(y + offsetY) * m_n + (x + offsetX)], &input2[y * m_ks + x]);
                }
            }
        }
    }
}

void Conv_t::backward()
{ 
    int m = m_m;
    int n = m_n;
    int ks = m_ks;
    tensor &x = input1;
    tensor &w = input2;

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

void Conv_t::update()
{ 
    int m = m_m;
    int n = m_n;
    int ks = m_ks;
    tensor &x = input1;
    tensor &w = input2;

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
 
                    output[i * m_out_y + j].val =  output[i * m_out_y + j].val + input1[(y + offsetY) * m_n + (x + offsetX)].val 
                                                *  input2[y * m_ks + x].val;
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
    tensor output;
    tensor input1;
    tensor input2;
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
    output = out;
    input1 = a;
    input2 = b;
    m_m = m;
    m_k = k;
    m_n = n;
}

void Matmul_t::forward()
{
    int m = m_m;
    int n = m_n;
    int k = m_k;
    tensor &out = output;
    tensor &a = input1;
    tensor &b = input2;
    output.ndim = 2;
    std::vector<int> shape = {m, n};
    output.shape = shape;

    for(int i=0; i < m; i++)
    { 
        for(int j=0; j < n; j++)    
        {  
            out[i*n+j].val = 0;
            for(int q=0; q < k; q++)    
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
    tensor &x = input1;
    tensor &w = input2;

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

void Matmul_t::update()
{
    int m = m_m;
    int n = m_n;
    int k = m_k;
    tensor &x = input1;
    tensor &w = input2;

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

    for (int i = 0; i < m * n; i++)
    {
        output[i].diff = 0;
        output[i].diffs.clear();
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

    for(int i=0; i < m; i++)
    { 
        for(int j=0; j < n; j++)    
        {  
            out[i*n+j].val = 0;
            for(int q=0; q < k; q++)    
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

void node::printDiff(void) 
{
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
};

class Add_t : public opBase 
{
public:
    tensor output;
    tensor input1;
    tensor input2;
    int length;

    Add_t(tensor &out, tensor &a, tensor &b, int len);
    void forward();
    void backward();
    void update();
};

Add_t::Add_t(tensor &out, tensor &a, tensor &b, int len)
{
    output = out;
    input1 = a;
    input2 = b;
    length = len;
}

void Add_t::forward()
{
    tensor &out = output;
    tensor &a = input1;
    tensor &b = input2;

    for(int i=0; i < length; i++)
    { 
        //out[i].val = a[i].val + b[i].val;
        out[i].val = add_diff(&a[i], &b[i], &out[i]);

        //a[i].setDiff(1, &out[i]);
        //b[i].setDiff(1, &out[i]);

    }
}

void Add_t::backward()
{
    tensor &x = input1;
    tensor &w = input2;

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

void Add_t::update()
{
    tensor &x = input1;
    tensor &w = input2;

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

    for (int i = 0; i < length; i++)
    {
        output[i].diff = 0;
        output[i].diffs.clear();
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

    for(int i=0; i < length; i++)
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
    tensor output;
    tensor input1;
    int length;
    Sigmoid_t(tensor &out, tensor &a, int length);
    void forward();
    void backward();
    void update();
};

Sigmoid_t::Sigmoid_t(tensor &out, tensor &a, int len)
{
    output = out;
    input1 = a;
    length = len;
}

void Sigmoid_t::forward()
{
    tensor &out = output;
    tensor &a = input1;

    for (int i = 0; i < length; i++)
    {
        out[i].val = 1.0/(1.0+exp(-a[i].val));
        a[i].setDiff(out[i].val * (1 - out[i].val), &out[i]);
    }
}

void Sigmoid_t::backward()
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

void Sigmoid_t::update()
{
    tensor &x = input1;

    for (int i = 0; i < length; i++)
    {
        x[i].val = x[i].val - lr * x[i].diff;
        x[i].diff = 0;
        x[i].diffs.clear();
    }

    for (int i = 0; i < length; i++)
    {
        output[i].diff = 0;
        output[i].diffs.clear();
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
        out[i].val = 1.0/(1.0+exp(-a[i].val));
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
    tensor output;
    tensor input1;
    tensor input2;
    int length;

    Loss_MSE_t(tensor &out, tensor &a, tensor &b, int len);
    void forward();
    void backward();
    void update();
};


Loss_MSE_t::Loss_MSE_t(tensor &out, tensor &a, tensor &b, int len)
{
    output = out;
    input1 = a;
    input2 = b;
    length = len;
}

void Loss_MSE_t::forward()
{
    tensor &loss = output;
    tensor &src = input1;
    tensor &dest = input2;
    int m = length;
    
    float sum = 0;

    for(int i=0; i < m; i++)
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
    tensor &x = input1;
    tensor &w = input2;

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

void Loss_MSE_t::update()
{
    tensor &x = input1;
    tensor &w = input2;

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

    output[0].diff = 0;
    output[0].diffs.clear();
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

    for(int i=0; i < m; i++)
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
    for(int i=0; i < m; i++)
    { 
        for(int j=0; j < n; j++)    
        {  
            out[i*n+j].val = 0;
            for(int q=0; q < k; q++)    
            {
                out[i * n + j].val+=a[i * k + q].val * b[q * n + j].val;
 
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

    for(int i=0; i < m; i++)
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
    for(int i=0; i < length; i++)
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
    std::list<opBase*> Layer;
    void AddLayer(opBase*);
    void forward();
    void backward();
    void update();
    void print();
};

void Net::AddLayer(opBase* ler)
{
    Layer.push_back(ler);
}

void Net::forward()
{
    // Net - > Op(Layer) -> node
    for (std::list<opBase*>::iterator choose=Layer.begin(); choose!=Layer.end(); ++choose)
        (*choose)->forward();
}

void Net::backward()
{
    for (std::list<opBase*>::reverse_iterator choose=Layer.rbegin(); choose!=Layer.rend(); ++choose)
        (*choose)->backward();
}

void Net::update()
{
    for (std::list<opBase*>::reverse_iterator choose=Layer.rbegin(); choose!=Layer.rend(); ++choose)
        (*choose)->update(); 
}

void Net::print()
{
    for (std::list<opBase*>::reverse_iterator choose=Layer.rbegin(); choose!=Layer.rend(); ++choose)
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

    tensor x(shape = {1, 784});
    tensor w1(shape = {784, 10});
    tensor o1(shape = {10});
    tensor b1(shape = {10});
    tensor sig1(shape = {10});
    tensor sig_out1(shape = {10});
    /*
    tensor w2(shape = {100, 10});
    tensor o2(shape = {10});
    tensor b2(shape = {10});
    tensor sig2(shape = {10});
    tensor sig_out2(shape = {10});
    */
    tensor ans(shape = {10});
    tensor loss(shape = {1});

    Matmul_t mat_1(o1, x, w1, 1, 784, 10);
    Add_t add_1(sig1, o1, b1, 10);
    Sigmoid_t sig_1(sig_out1, sig1, 10);
    Loss_MSE_t lose_mse_t(loss, sig_out1, ans, 10);
    //-----------------------------------------------------
    Net net;
    net.AddLayer(&mat_1);
    net.AddLayer(&add_1);
    net.AddLayer(&sig_1);
    net.AddLayer(&lose_mse_t);

    int Correct = 0;
    int Error = 0;

    int epoch = 2;
    for (int e = 0; e < epoch; e++)
    {
        for (unsigned int i = 0; i < dataset.training_images.size(); i++)
        {
            for (unsigned int j = 0; j < dataset.training_images[i].size(); j++)
                x[j].val = ((float)(unsigned int)dataset.training_images[i][j]) / 255;

            int target_value = (unsigned int)dataset.training_labels[i];
            for (int k = 0; k < 10; k++){
                ans[k].val = 0;
            }
            ans[target_value].val = 1;

            net.forward();
            std::cout << "ans : " << target_value << std::endl;
            std::cout << "sig_out[0].val : " << sig_out1[0].val << std::endl;
            std::cout << "sig_out[1].val : " << sig_out1[1].val << std::endl;
            std::cout << "sig_out[2].val : " << sig_out1[2].val << std::endl;
            std::cout << "sig_out[3].val : " << sig_out1[3].val << std::endl;
            std::cout << "sig_out[4].val : " << sig_out1[4].val << std::endl;
            std::cout << "sig_out[5].val : " << sig_out1[5].val << std::endl;
            std::cout << "sig_out[6].val : " << sig_out1[6].val << std::endl;
            std::cout << "sig_out[7].val : " << sig_out1[7].val << std::endl;
            std::cout << "sig_out[8].val : " << sig_out1[8].val << std::endl;
            std::cout << "sig_out[9].val : " << sig_out1[9].val << std::endl;
            std::cout << "loss : " << loss[0].val << std::endl;

            // tesing
            double max_value = -99999;
            int max_index = 0;
            for (int k = 0; k < 10; k++)
            {
                if (sig_out1[k].val > max_value)
                {
                    max_value = sig_out1[k].val;
                    max_index = k;
                }
            }

            if (max_index == target_value)
            {
                std::cout << "Correct !" << std::endl;
                Correct++;
            }
            else
            {
                std::cout << "Error !" << std::endl;
                Error++;
            }

            std::cout << "Correct : " << Correct << " , " << "Error : " << Error << " , Rate : " << (float)Correct / ((float)Correct + (float)Error)<< std::endl;

            net.backward();
            net.update();
            //usleep(1000*1000);
        }
    }

    return 0;
#endif

#if 0
    // Load MNIST data
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
    mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("./third_party/mnist/");
    std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
    std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
    std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
    std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;
    std::cout << "----------------------" << std::endl;
    //-----------------------------------------------------
    // (1) iteration: represents 1 iteration (also called training step), and updates the parameters of the network structure once per iteration;
    // (2) batch-size: the sample size used in one iteration;
    // (3) epoch: 1 epoch indicates that all samples in the training set have been passed.

    std::vector<int> shape;

    tensor x(shape = {1, 784});
    tensor w1(shape = {784, 10});
    tensor o1(shape = {10});
    tensor b1(shape = {10});
    tensor rel1(shape = {10});
    tensor rel1_out(shape = {10});
    tensor sig1(shape = {10});
    /*
    tensor w2(shape = {100, 10});
    tensor b2(shape = {10});
    tensor o2(shape = {10});
    tensor sig2(shape = {10});
    tensor sig2_out(shape = {10});
    */
    tensor ans(shape = {10});
    tensor loss(shape = {1});

    Matmul_t mat_1(o1, x, w1, 1, 784, 10);
    Sigmoid_t sig_1(sig1, o1, 10);
    Loss_MSE_t lose_mse_t(loss, sig1, ans, 10);
    //-----------------------------------------------------
    Net net;
    net.AddLayer(&mat_1);
    net.AddLayer(&sig_1);
    net.AddLayer(&lose_mse_t);

    // Dataset 60000 times
    int epoch = 1;
    int batch = 1;
    // iteration = 60000

    // Training
    std::vector<float> Value;
    int cnt = 0;
    int train_runs = 150000;
    for (int e = 0; e < epoch; e++)
    {
        for (unsigned int i = 0; i < dataset.training_images.size(); i++)
        {
            for (unsigned int j = 0; j < dataset.training_images[i].size(); j++)
                x[i].val = ((float)(unsigned int)dataset.training_images[i][j]) / 255;

            //initialize the label
            int target_value = (unsigned int)dataset.training_labels[i];
            for (int k = 0; k < 10; k++){
                ans[k].val = 0;
            }
            ans[target_value].val = 1;

            for (int b = 0; b < batch; b++)
            {
                net.forward();
                net.backward();
            }
            net.update();

            cnt ++;

            if (cnt % 1000 == 0)
            {
                //std::cout << "[Epoch : " << epoch << "], [Batch : " << batch << "], [iteration : " << cnt << "], [Accuracy : " << Testing(net, 1000) << "% ... success]" << std::endl;
                mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
                mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("./third_party/mnist/");
                int test_runs_count = 1000;
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
                        if (sig1[k].val > max_value)
                        {
                            max_value = sig1[k].val;
                            max_index = k;
                        }
                    }

                    //output == target
                    if (ans[max_index].val > 0.9)
                    {
                        test_success_count ++;
                    }
                    net.update();
                    test_num ++;

                    if ((int)test_num % test_runs_count == 0)
                        std::cout << "[Epoch : " << epoch << "], [Batch : " << batch << "], [iteration : " << cnt << "], [Accuracy : " << (float)test_success_count / (float)test_runs_count * 100 << "% ... success]" << std::endl;
                    if (test_num>=test_runs) break;
                }
                std::cout << "[Epoch : " << epoch << "], [Batch : " << batch << "], [iteration : " << cnt << "], [Accuracy : " << (float)test_success_count / (float)test_runs_count * 100 << "% ... success]" << std::endl;
            }

            if (cnt >= train_runs ) break;
        }
    }

    return 0;
#endif
}


#if 0
    //-----------------------------------------------------
    tensor x(shape = {3, 3});
    //-----------------------------------------------------
    tensor k(shape = {2, 2});
/*
    k[0].val = 1.0; k[1].val = 1.0; 
    k[2].val = 1.0; k[3].val = 1.0;
*/
    //-----------------------------------------------------
    tensor o(shape = {4});
    tensor b(shape = {4});
    //-----------------------
/*
    b[0].val = 1.0; b[1].val = 1.0; 
    b[2].val = 1.0; b[3].val = 1.0;
*/
    //-----------------------
    tensor ad(shape = {4});
    tensor sig(shape = {4});
    tensor ans(shape = {4});
    tensor loss(shape = {1});

    Net net;

    Conv_t conv_t(o, x, k, 3, 3, 1, 2, 2, 2);
    Add_t add_t(ad, o, b, 4);
    Sigmoid_t sig_t(sig, ad, 4);
    Loss_MSE_t lose_mse_t(loss, sig, ans, 4);

    net.AddLayer(&conv_t);
    net.AddLayer(&add_t);
    net.AddLayer(&sig_t);
    net.AddLayer(&lose_mse_t);
    
    do
    {
        //set input

        x[0].val = 0.4; x[1].val = 0.5; x[2].val = 0.6;
        x[3].val = 0.7; x[4].val = 0.8; x[5].val = 0.9;
        x[6].val = 1.0; x[7].val = 1.1; x[8].val = 1.2;

        ans[0].val = 0.333; 
        ans[1].val = 0.555;
        ans[2].val = 0.777; 
        ans[3].val = 0.999;

        net.forward();
        std::cout << "x[0].val : " << x[0].val << std::endl;
        std::cout << "x[1].val : " << x[1].val << std::endl;
        std::cout << "x[2].val : " << x[2].val << std::endl;
        std::cout << "x[3].val : " << x[3].val << std::endl;
        std::cout << "k[0].val : " << k[0].val << std::endl;
        std::cout << "k[1].val : " << k[1].val << std::endl;
        std::cout << "k[2].val : " << k[2].val << std::endl;
        std::cout << "k[3].val : " << k[3].val << std::endl;
        std::cout << "ad[0].val : " << ad[0].val << std::endl;
        std::cout << "ad[1].val : " << ad[1].val << std::endl;
        std::cout << "ad[2].val : " << ad[2].val << std::endl;
        std::cout << "ad[3].val : " << ad[3].val << std::endl;
        std::cout << "sig[0].val : " << sig[0].val << std::endl;
        std::cout << "sig[1].val : " << sig[1].val << std::endl;
        std::cout << "sig[2].val : " << sig[2].val << std::endl;
        std::cout << "sig[3].val : " << sig[3].val << std::endl;
        std::cout << "loss : " << loss[0].val << std::endl;
        net.backward();
        net.update();
        usleep(1000*1000);

    }
    while (loss[0].val > 0.0001);

    return 0;
#endif 

#if 0
// y = x * z; //12  dy/dx(4)
// g = x * h; //21  dg/dx(7)
// f = y + g; //33  df/dy(1) df/dg(1)
// L = f * h; //231 dL/df(7) dL/dh(33)
// [dy/dx(4) * df/dy(1) * dL/df(7)] + [dg/dx(7) * df/dg(1) * dL/df(7)] = 28 + 49 = 77
// [dL/dh(33)] + [dg/dh(3) * df/dg(1) * dL/df(7)] = 54
node x[1];
x[0].val = 3;
node z[1];
z[0].val = 4;
node h[1];
h[0].val = 7;
node y[1];
node g[1];
node f[1];
node L[1];
mul(y, x, z);
mul(g, x, h);
add(f, y, g);
mul(L, f, h);
printf("L[0].val : %f\n",L[0].val);
float diff = get_diff(g, L); //dL/dy
printf("diff : %f\n", diff); 
#endif