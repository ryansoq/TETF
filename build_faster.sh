rm TETF
g++ main.cc -DTYPE2_BACKWARD -DTYPE3_BACKWARD_CONV -std=c++11 -Ofast -o TETF -I ${PWD}/third_party/mnist/include -I ${PWD}/third_party/f2uc
#g++ main.cc -std=c++11 -Ofast -o TETF -I ${PWD}/third_party/mnist/include -I ${PWD}/third_party/f2uc
