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


local pl = require('pl.import_into')()

-- Parse arguments
local cmd = RNNOption()
g_params = cmd:parse(arg)
torch.manualSeed(1990)

--~ cmd:print_params(g_params)
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
eos = word2id["<eos>"]
--~ print(eos)
model_path = 'output/wmt2015en/nce_cnnlm_bsz=128_memsize=16_nkernels=2_vsize=256_nlayers=1_kmax=-1_dropout=0.2/model_27'

load_state = torch.load(model_path)

--~ print(load_state)

print("LOADING ...")
model = load_state.model
criterion = load_state.criterion
nce = load_state.nce
ncecrit = load_state.ncecrit
print("LOADED")

function eval(split)

	model:evaluate()
	nce:evaluate()
	loader:reset_batch_pointer(split)

	local n_batches = loader.split_sizes[split]
	local total_loss = 0
	local total_samples = 0
	smt = false

	for i = 1, n_batches do
		xlua.progress(i, n_batches)

		
		local context, target, time = loader:next_batch(split)
		local batch_size = target:size(1)

		-- local embeddings = lookuptable:forward(context)
		local input = {context, time}
		local net_output = model:forward(input)
		local nce_output = nce:forward({net_output, target})

		-- Get loss through the decoder
		local loss, tree_output
		
		local loss = criterion:forward(nce_output, target)

		total_loss = total_loss + loss
		total_samples = total_samples + batch_size

	end

	total_loss = total_loss / total_samples

	local perplexity = torch.exp(total_loss)

	return total_loss

end

function sample(seed, n, temperature, output_file)
	
	model:evaluate()
	nce:evaluate()
	n = n or 10
	temperature = temperature or 1
	local context_size = g_params.model.memsize
	local context = torch.Tensor(1, context_size):fill(eos)
	local words = pl.utils.split(seed, ' ')
	local text = seed 
	
	local y = g_params.model.memsize
	for i = #words, 1, -1 do
		local word = words[i]
		local id = word2id[word]
		context[1][y] = id
		y = y - 1
	end
	
	local time = torch.Tensor(1, context_size):cuda()
	for t = 1, context_size do
		time:select(2, t):fill(t)
	end
	
	-- A dummy target for NCE (we don't need it)
	local target = torch.Tensor(1):cuda()
	target[1] = 1990
	
	--~ Start sampling
	local timer = torch.tic()
	for step = 1, n do
		
		--~ xlua.progress(step, n)
		local net_output = model:forward({context, time})
		local prediction = nce:forward({net_output, target})
		
		prediction:div(temperature)
		local probs = torch.exp(prediction):squeeze()
		probs:div(torch.sum(probs)) -- renormalize so probs sum to one
		local next_id = torch.multinomial(probs:float(), 1):resize(1):float()
		local next_word = id2word[next_id[1]]
		--~ print(next_word)
		
		-- create a new context (move 1 step behind)
		local new_context = torch.Tensor(1, context_size):cuda()
		new_context:sub(1, -1, 1, -2):copy(context:sub(1, -1, 2, -1))
		new_context[1][context_size] = next_id
		context:copy(new_context)
		
		text = text .. " " .. next_word
		
		if next_word == "<eos>" then
			local file = io.open("sampling_output.txt", "a")
			io.output(file)
			io.write(text .. "\n")
			io.flush()
			io.close(file)
			text = ""
			--~ 
		end
		
		if step % 10000 == 0 then
			local elapse = torch.toc(timer)
			local speed = math.floor(elapse / 10)
			io.stdout:write("Sampled " .. step .. " words / " .. n .. " total words, speed = " .. speed .. ' ms per word\n')
			io.stdout:flush()
			timer = torch.tic()
		end
	end
	
	return text
end

local seed = "the meaning of life is"

local text = sample(seed, 20e6, 1, "sampling_output.txt")

--~ local file = io.open("sampling_output.txt", "w")
--~ io.output(file)

--~ io.write(text)

--~ io.close(file)
