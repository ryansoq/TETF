rm TETF
#g++ main.cc -std=c++11 -O0 -g3 -o TETF -I ${PWD}/third_party/mnist/include -I ${PWD}/third_party/f2uc
g++ main.cc -DTYPE2_BACKWARD -std=c++11 -O0 -g3 -o TETF -I ${PWD}/third_party/mnist/include -I ${PWD}/third_party/f2uc
