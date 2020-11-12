rm TETF
g++ main.cc -DTYPE2_BACKWARD -DTYPE4_BACKWARD_CONV -DIM2COLxGEMM -std=c++11 -Ofast -o TETF -I ${PWD}/third_party/mnist/include -I ${PWD}/third_party/f2uc
#g++ main.cc -std=c++11 -Ofast -o TETF -I ${PWD}/third_party/mnist/include -I ${PWD}/third_party/f2uc
