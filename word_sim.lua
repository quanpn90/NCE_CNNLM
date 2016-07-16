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

cmd:print_params(g_params)
g_params.dataset.name = 'wmt2015en'
g_params.dataset.batch_size = 10000
g_params.trainer.batch_size = 10000

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

local embeddings
for _,node in ipairs(model.forwardnodes) do
	
	local module = node.data.module
	
	if (torch.type(module) == 'nn.LookupTable') then
		if (node.data.module.weight:size(1) == vocab_size) then
			embeddings = node.data.module.weight
		end
	end
	
end

print(embeddings:size())


local simlex999 = "SimLex-999/SimLex-999.txt"
local count = 0

local scores = {}
 
for line in io.lines(simlex999) do
	
	--~ print(line)
	local rows = pl.utils.split(line, '\t')
	
	local word1 = rows[1]
	local word2 = rows[2]
	local score = rows[4]
	
	if word1 ~= 'word1' then
		
		local id1 = word2id[word1] 
		local id2 = word2id[word2]
		
		if id1 ~= nil and id2 ~= nil then
			count = count + 1
			--~ gold_scores[count] = score
			local v1 = embeddings[id1]
			local v2 = embeddings[id2]
			
			local w2v_score = v1:dot(v2)
			--~ w2v_scores[count] = w2v_score
			
			table.insert(scores, {score, w2v_score})
			print(count, word1, word2, score, w2v_score)
		end
		
	end
	
	
end


function spearman(a)
	local function aux_func(t) -- auxiliary function
		return (t == 1 and 0) or (t*t - 1) * t / 12
	end

	for _, v in pairs(a) do v.r = {} end
	local T, S = {}, {}
	-- compute the rank
	for k = 1, 2 do
		table.sort(a, function(u,v) return u[k]<v[k] end)
		local same = 1
		T[k] = 0
		for i = 2, #a + 1 do
			if i <= #a and a[i-1][k] == a[i][k] then same = same + 1
			else
				local rank = (i-1) * 2 - same + 1
				for j = i - same, i - 1 do a[j].r[k] = rank end
				if same > 1 then T[k], same = T[k] + aux_func(same), 1 end
			end
		end
		S[k] = aux_func(#a) - T[k]
	end
	-- compute the coefficient
	local sum = 0
	for _, v in pairs(a) do -- TODO: use nested loops to reduce loss of precision
		local t = (v.r[1] - v.r[2]) / 2
		sum = sum + t * t
	end
	return (S[1] + S[2] - sum) / 2 / math.sqrt(S[1] * S[2])
end


print(scores)
print(spearman(scores))
