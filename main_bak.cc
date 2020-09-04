#include <iostream>
#include <cmath>
#include <unistd.h>
#include <vector>
#include <string>
#include <list>

// Net class
float lr = 0.01;

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

    return input1->val + input2->val;
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

    return input1->val + input2->val;
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
};

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
        out[i] = 1.0/(1.0+exp(-(a[i])));
        a[i].setDiff(out[i] * (1 - out), &out[i]);
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
    void forward();
    void backward();
    void update();
    void AddLayer(opBase*);
};

void Net::AddLayer(opBase* ler)
{
    Layer.push_back(ler);
}

void Net::forward()
{
    //for(opBase* choose : Layer) 
    //    choose->forward();
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

int main()
{
    node loss;
    //float loss1;
    //float ans = 13.0;
    //float ans1 = 17.0;
    node ans[2];
    //ans[0].val = 777.0; 
    //ans[1].val = -7.0;
    ans[0].val = 777.0; 
    ans[1].val = 7.0;
    node sig[2];

    node x[3];
    x[0].val = 1; x[1].val = 2; x[2].val = 3;
    node w[6];
    w[0].val = 4; w[1].val = 5; 
    w[2].val = 6; w[3].val = 7; 
    w[4].val = 8; w[5].val = 9;
    node o[2];
    node b[2];
    b[0].val = 1;
    b[1].val = 1;
    node out[2];
    Net net;
    Matmul mat(o, x, w, 1, 3, 2);
    Add add(out ,o, b, 2);
    
    Sigmoid relu(sig, out, 2);
    
    Loss_MSE lose_mse(&loss, sig, ans, 2);
    net.AddLayer(&mat);
    net.AddLayer(&add);
    net.AddLayer(&relu);
    net.AddLayer(&lose_mse);
    do{
        //set input 
        x[0].val = 1; x[1].val = 2; x[2].val = 3;

        //ans[0].val = 777.0; 
        //ans[1].val = -7.0;
        ans[0].val = 777.0; 
        ans[1].val = 7.0;

        //forward
        //matmul(o, x, w, 1, 3, 2);
        //mat.forward();
        //add.forward();
        net.forward();
        net.backward();
        std::cout << "loss.printDiff : ";loss.printDiff();
        std::cout << "out[0].printDiff : ";out[0].printDiff();
        std::cout << "out[1].printDiff : ";out[1].printDiff();
        std::cout << "o[0].printDiff : ";o[0].printDiff();
        std::cout << "o[1].printDiff : ";o[1].printDiff();
        std::cout << "b[0].printDiff : ";b[0].printDiff();
        std::cout << "b[1].printDiff : ";b[1].printDiff();
        std::cout << "w[0].printDiff : ";w[0].printDiff();
        std::cout << "w[1].printDiff : ";w[1].printDiff();
        std::cout << "w[2].printDiff : ";w[2].printDiff();
        std::cout << "w[3].printDiff : ";w[3].printDiff();
        std::cout << "w[4].printDiff : ";w[4].printDiff();
        std::cout << "w[5].printDiff : ";w[5].printDiff();
        std::cout << "x[0].printDiff : ";x[0].printDiff();
        std::cout << "x[1].printDiff : ";x[1].printDiff();
        std::cout << "x[2].printDiff : ";x[2].printDiff();
        std::cout << "loss : " << loss.val << std::endl;
        std::cout << "out[0].val : " << out[0].val << std::endl;
        std::cout << "out[1].val : " << out[1].val << std::endl;
        net.update();
        usleep(1000*1000);
      }while (loss.val > 0.01);

/*
    do{
        //init
        w0.diff = 0;
        x0.diff = 0;

        //forward
        mul(o0, x0, w0);
        mul(o1, x1, w1);
        add(o2, o0, o1);
        loss = o2.val - ans;
        //loss1 = ans1 - o1.val;
        std::cout << "-------------------" << std::endl;
        std::cout << "o2.val : " << o2.val << std::endl;
        std::cout << "o0.val : " << o0.val << std::endl;
        std::cout << "o1.val : " << o1.val << std::endl;
        std::cout << "loss : " << loss << std::endl;
        //std::cout << "loss1 : " << loss1 << std::endl;

        //backward
        o2.diff = 1.0; //df/do0
        //o1.diff = 1.0; //df/do1
        o0.diff = o0.diff * o0.next->diff;
        o1.diff = o1.diff * o1.next->diff;
        w0.diff = w0.diff * w0.next->diff;
        w1.diff = w1.diff * w1.next->diff;
       
        //w0.diff = w0.diff * o0.diff; // do/w0 * df/do
        std::cout << "w0.val :" << w0.val << std::endl;
        std::cout << "w0.diff : " << w0.diff << std::endl;

        //update
        w0.val = w0.val - 0.01 * w0.diff * loss;
        w1.val = w1.val - 0.01 * w1.diff * loss;
        x0.diff = 0; x1.diff = 0;
        w0.diff = 0; w1.diff = 0;
        o0.diff = 0; o1.diff = 0;
        std::cout << "w0.val : " << w0.val << std::endl;
         
        usleep(1000*1000);

    }while (fabs(loss) > 0.1);
*/
/*
        //loss1 = ans1 - o1.val;
        std::cout << "o[0].val : " << o[0].val << std::endl;
        std::cout << "o[1].val : " << o[1].val << std::endl;
        std::cout << "out[0].val : " << out[0].val << std::endl;
        std::cout << "loss : " << loss.val << std::endl;
        //std::cout << "loss1 : " << loss1 << std::endl;

        //backward
        std::pair<float, node*> value_out;
        value_out.first = 1.0;
        value_out.second = NULL; 
        out[0].diffs.push_back(value_out);
        out[0].diff = 1.0;
        out[0].loss = out[0].val - ans;

        value_out.first = 1.0;
        value_out.second = NULL; 
        out[1].diffs.push_back(value_out);
        out[1].diff = 1.0;
        out[1].loss = out[1].val - ans1;

        for (int i = 0; i < o[0].diffs.size(); i++)
        {
            o[0].diff += o[0].diffs[i].first * o[0].diffs[i].second->diff;
            o[0].loss += o[0].diffs[i].second->loss;
            std::cout << "o[0].diffs[i].first : " << o[0].diffs[0].first << std::endl;
            std::cout << "o[0].diffs[i].second->diff : " << o[0].diffs[0].second->diff << std::endl;
            std::cout << "o[0].diffs[i].second->val : " << o[0].diffs[0].second->val << std::endl;
        }

        for (int i = 0; i < b[0].diffs.size(); i++)
        {
            b[0].diff += b[0].diffs[i].first * b[0].diffs[i].second->diff;
            b[0].loss += b[0].diffs[i].second->loss;
        }

        for (int i = 0; i < o[1].diffs.size(); i++)
        {
            o[1].diff += o[1].diffs[i].first * o[1].diffs[i].second->diff;
            o[1].loss += o[1].diffs[i].second->loss;
            std::cout << "o[1].diffs[i].first : " << o[1].diffs[0].first << std::endl;
            std::cout << "o[1].diffs[i].second->diff : " << o[1].diffs[0].second->diff << std::endl;
            std::cout << "o[1].diffs[i].second->diff : " << o[1].diffs[0].second->val << std::endl;
        }

        for (int i = 0; i < b[1].diffs.size(); i++)
        {
            b[1].diff += b[1].diffs[i].first * b[1].diffs[i].second->diff;
            b[1].loss += b[1].diffs[i].second->loss;
        }
        for (int i = 0; i < w[0].diffs.size(); i++)
        {
            w[0].diff += w[0].diffs[i].first * w[0].diffs[i].second->diff;
            w[0].loss +=w[0].diffs[i].second->loss;
        }
        for (int i = 0; i < w[1].diffs.size(); i++)
        {
            w[1].diff += w[1].diffs[i].first * w[1].diffs[i].second->diff;
            w[1].loss +=w[1].diffs[i].second->loss;
        }
        for (int i = 0; i < w[2].diffs.size(); i++)
        {
            w[2].diff += w[2].diffs[i].first * w[2].diffs[i].second->diff;
            w[2].loss +=w[2].diffs[i].second->loss;
        }
        for (int i = 0; i < w[3].diffs.size(); i++)
        {
            w[3].diff += w[3].diffs[i].first * w[3].diffs[i].second->diff;
            w[3].loss +=w[3].diffs[i].second->loss;
        }
        for (int i = 0; i < w[4].diffs.size(); i++)
        {
            w[4].diff += w[4].diffs[i].first * w[4].diffs[i].second->diff;
            w[4].loss +=w[4].diffs[i].second->loss;
        }
        for (int i = 0; i < w[5].diffs.size(); i++)
        {
            w[5].diff += w[5].diffs[i].first * w[5].diffs[i].second->diff;
            w[5].loss +=w[5].diffs[i].second->loss;
        }

        //w0.diff = w0.diff * o0.diff; // do/w0 * df/do
        std::cout << "o[0].val : " << o[0].val << std::endl;
        std::cout << "b[0].val : " << b[0].val << std::endl;
        std::cout << "o[1].val : " << o[1].val << std::endl;
        std::cout << "b[1].val : " << b[1].val << std::endl;
        std::cout << "w[0].val : " << w[0].val << std::endl;
        std::cout << "w[1].val : " << w[1].val << std::endl;
        std::cout << "w[2].val : " << w[2].val << std::endl;
        std::cout << "w[3].val : " << w[3].val << std::endl;
        std::cout << "w[4].val : " << w[4].val << std::endl;
        std::cout << "w[5].val : " << w[5].val << std::endl;

        std::cout << "o[0].diff : " << o[0].diff << std::endl;
        std::cout << "b[0].diff : " << b[0].diff << std::endl;
        std::cout << "o[1].diff : " << o[1].diff << std::endl;
        std::cout << "b[1].diff : " << b[1].diff << std::endl;
        std::cout << "w[0].diff : " << w[0].diff << std::endl;
        std::cout << "w[1].diff : " << w[1].diff << std::endl;
        std::cout << "w[2].diff : " << w[2].diff << std::endl;
        std::cout << "w[3].diff : " << w[3].diff << std::endl;
        std::cout << "w[4].diff : " << w[4].diff << std::endl;
        std::cout << "w[5].diff : " << w[5].diff << std::endl;

        std::cout << "o[0].loss : " << o[0].loss << std::endl;
        std::cout << "b[0].loss : " << b[0].loss << std::endl;
        std::cout << "o[1].loss : " << o[1].loss << std::endl;
        std::cout << "b[1].loss : " << b[1].loss << std::endl;
        std::cout << "w[0].loss : " << w[0].loss << std::endl;
        std::cout << "w[1].loss : " << w[1].loss << std::endl;
        std::cout << "w[2].loss : " << w[2].loss << std::endl;
        std::cout << "w[3].loss : " << w[3].loss << std::endl;
        std::cout << "w[4].loss : " << w[4].loss << std::endl;
        std::cout << "w[5].loss : " << w[5].loss << std::endl;



        std::cout << "o[0].printDiff : ";o[0].printDiff();
        std::cout << "b[0].printDiff : ";b[0].printDiff();
        std::cout << "o[1].printDiff : ";o[1].printDiff();
        std::cout << "b[1].printDiff : ";b[1].printDiff();
        std::cout << "w[0].printDiff : ";w[0].printDiff();
        std::cout << "w[1].printDiff : ";w[1].printDiff();
        std::cout << "w[2].printDiff : ";w[2].printDiff();
        std::cout << "w[3].printDiff : ";w[3].printDiff();
        std::cout << "w[4].printDiff : ";w[4].printDiff();
        std::cout << "w[5].printDiff : ";w[5].printDiff();
        std::cout << "x[0].printDiff : ";x[0].printDiff();
        std::cout << "x[1].printDiff : ";x[1].printDiff();
        std::cout << "x[2].printDiff : ";x[2].printDiff();
        

        float lr = 0.01;
        //update
        o[0].val = o[0].val - lr * o[0].diff * o[0].loss;
        b[0].val = b[0].val - lr * b[0].diff * b[0].loss;
        o[1].val = o[1].val - lr * o[1].diff * o[1].loss;
        b[1].val = b[1].val - lr * b[1].diff * b[1].loss;
        w[0].val = w[0].val - lr * w[0].diff * w[0].loss;
        w[1].val = w[1].val - lr * w[1].diff * w[1].loss;
        w[2].val = w[2].val - lr * w[2].diff * w[2].loss;
        w[3].val = w[3].val - lr * w[3].diff * w[3].loss;
        w[4].val = w[4].val - lr * w[4].diff * w[4].loss;
        w[5].val = w[5].val - lr * w[5].diff * w[5].loss;

        out[0].diff = 0;
        o[0].diff = 0;o[1].diff = 0;
        b[0].diff = 0;b[1].diff = 0;
        w[0].diff = 0;
        w[1].diff = 0;
        w[2].diff = 0;
        w[3].diff = 0;
        w[4].diff = 0;
        w[5].diff = 0;
        
        o[0].loss = 0;
        b[0].loss = 0; 
        o[1].loss = 0; 
        b[1].loss = 0; 
        w[0].loss = 0; 
        w[1].loss = 0; 
        w[2].loss = 0; 
        w[3].loss = 0; 
        w[4].loss = 0; 
        w[5].loss = 0; 
        
        out[0].diffs.clear();
        out[1].diffs.clear();
        b[0].diffs.clear();
        b[1].diffs.clear();
        o[0].diffs.clear(); o[1].diffs.clear();
        b[0].diffs.clear(); b[1].diffs.clear();
        w[0].diffs.clear();
        w[1].diffs.clear();
        w[2].diffs.clear();
        w[3].diffs.clear();
        w[4].diffs.clear();
        w[5].diffs.clear();
        x[0].diffs.clear();
        x[1].diffs.clear();
        x[2].diffs.clear();
        usleep(1000*1000);
        */
    return 0;		
}
