# NCE_CNNLM
Convolutional Neural Network Language Models

# Requirement

- torch, cunn, cutorch, nngraph
- cudnn (tested with CUDA 7.5 and cudnn v5) for fast Convolutional Modules
- dpnn (for NCE modules to handle large vocabulary) https://github.com/Element-Research/dpnn

# Usage

- create a data dir ./data which should contain train.txt, valid.txt and test.txt corresponding to your training, validation and test data
- for example, with penntreebank you can use the data from https://github.com/facebookresearch/torch-rnnlib/tree/master/examples/word-language-model/penn
- run the program to train a model (if your data is located at ./data/ptb): with the default configuration (in the paper)

  th train.lua -dset ptb -name spatial_lm -nlayers 1 -nkernels 2 -highway_mlp 1 -vsize 128 -kmax 2 -ksize 3 -dropout 0.3 

- more configurations can be found at options.lua
