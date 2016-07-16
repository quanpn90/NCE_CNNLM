-- This file loads a trained CONVNET and samples text from a seed.
require('torch')
require('nn')
require('nnx')
require('nngraph')
require 'graph'
require('options')
-- require 'utils.misc'

require('utils.batchloader')
require('utils.textsource')
model_utils = require('utils/model_utils')


require 'cutorch'
require 'cunn'
require 'cunnx'
require 'cudnn'
require 'rnn'
require 'gnuplot'

local pl = require('pl.import_into')()

-- Parse arguments
local cmd = RNNOption()
g_params = cmd:parse(arg)
torch.manualSeed(1990)

cmd:print_params(g_params)
g_params.dataset.name = 'wmt2015en'
g_params.dataset.batch_size = 128
g_params.trainer.batch_size = 128

-- GPU querying
local gpuid = g_params.trainer.gpuid
cutorch.setDevice(gpuid)
local GPU = cutorch.getDeviceProperties(gpuid).name
print("Using CUDA on " .. GPU)
cutorch.manualSeed(1990)

-- build the torch dataset
local g_dataset = TextSource(g_params.dataset)
vocab_size = g_dataset:get_vocab_size()

local loader = BatchLoader(g_params.dataset, g_dataset)
id2word = g_dataset.dict.index_to_symbol
word2id = g_dataset.dict.symbol_to_index

local sampling_file = "sampling_output.txt"

local distribution = torch.Tensor(vocab_size - 2):zero()

for line in io.lines(sampling_file) do

	local words = pl.utils.split(line, ' ')
	
	for _, word in ipairs(words) do
		local wordid = word2id[word]
		
		if wordid > 2 then
			distribution[wordid - 2] = distribution[wordid - 2] + 1
		end
	end
end


local x, y = torch.sort(distribution, true)



for i = 1, vocab_size - 2 do
	
	if x[i] >= 0 then
		print(i .. "\t" .. x[i])
	end
end


--~ print(distribution)

--~ print(torch.max(distribution))

--~ gnuplot.plot(subset)

--~ gnuplot.hist(distribution, 100)
