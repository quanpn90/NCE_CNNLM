-- This file trains and tests the CONVNET LM from a batch loader.
require('torch')
require('nn')
require('nngraph')
require 'graph'
require('options')
-- require 'utils.misc'

require('utils.batchloader')
require('utils.textsource')
model_utils = require('utils/model_utils')
require 'models.builder-spatial'
require 'dpnn'
require 'cutorch'
require 'cunn'
require 'cudnn'
cudnn.benchmark = true
cudnn.fastest = true




-- Parse arguments
local cmd = RNNOption()
g_params = cmd:parse(arg)
torch.manualSeed(1990)

local gpuid = g_params.trainer.gpuid
cutorch.setDevice(gpuid)
local GPU = cutorch.getDeviceProperties(gpuid).name
print("Using CUDA on " .. GPU)
cutorch.manualSeed(1990)

cmd:print_params(g_params)

-- build the torch dataset
local g_dataset = TextSource(g_params.dataset)
vocab_size = g_dataset:get_vocab_size()

local loader = BatchLoader(g_params.dataset, g_dataset)
local criterion = nn.ClassNLLCriterion(nil, false)
criterion:cuda()

local g_dictionary = g_dataset.dict

local unigrams = g_dataset.dict.index_to_freq:clone()
unigrams:div(unigrams:sum())

local k = 10
local Z = torch.exp(9)



--~ local model, output_size
if g_params.trainer.load == '' then
	model, output_size = ModelBuilder:make_net(g_params.model, vocab_size)
	params, grad_params = model:getParameters()

	
	nce = nn.NCEModule(output_size, vocab_size, k, unigrams, Z) 
	nce:cuda()
	nce.logsoftmax = true
	nce.normalized = true
	
	criterion = nn.ClassNLLCriterion(nil, false):cuda()
	ncecrit = nn.NCECriterion(false):cuda()
	ncecrit.sizeAverage = false
	nce.weight:normal(0, 0.05)
	nce.bias:zero()
else

	local load_state = torch.load(g_params.trainer.load)
	model = load_state.model
	criterion = load_state.criterion
	nce = load_state.nce
	ncecrit = load_state.ncecrit
	params, grad_params = model:getParameters()
	g_params.trainer.initial_learning_rate = load_state.learning_rate
end


local total_params = params:nElement()

print("Total parameters of model: " .. total_params)



function eval(split)

	model:evaluate()
	nce:evaluate()
	loader:reset_batch_pointer(split)

	local n_batches = loader.split_sizes[split]
	local total_loss = 0
	local total_samples = 0

	for i = 1, n_batches do
		xlua.progress(i, n_batches)

		
		local context, target, time = loader:next_batch(split)
		local batch_size = target:size(1)

		-- local embeddings = lookuptable:forward(context)
		local input = {context, time}
		local net_output = model:forward(input)
		

		local nce_output = nce:forward({net_output, target})
		local loss = criterion:forward(nce_output, target)	

		total_loss = total_loss + loss
		total_samples = total_samples + batch_size

	end

	total_loss = total_loss / total_samples

	local perplexity = torch.exp(total_loss)

	return total_loss

end

function train_epoch(learning_rate, gradient_clip)

	model:training()
	nce:training()
	loader:reset_batch_pointer(1)

	
	-- if hsm == true then hsm_grad_params:zero() end
	local speed
	local n_batches = loader.split_sizes[1]
	local total_loss = 0
	local total_samples = 0

	local timer = torch.tic()

	for i = 1, n_batches do
		
		model:zeroGradParameters()
		nce:zeroGradParameters()
		xlua.progress(i, n_batches)

		-- forward pass 
		local context, target, time = loader:next_batch(split)
		local batch_size = target:size(1)


		local input = {context, time}
		local net_output = model:forward(input)
		
		-- NCE pass
		total_samples = total_samples + batch_size
		local nce_output = nce:forward({net_output, target})
		
		local loss = ncecrit:forward(nce_output, target)

		

		total_loss = total_loss + loss
		total_samples = total_samples + batch_size

		-- backward pass
		local dloss = ncecrit:backward(nce_output, target)
		local nceloss = nce:backward({net_output, target}, dloss)
  		model:backward(input, nceloss[1])


		-- Control if gradient too big
		local norm = grad_params:norm()

		if norm > gradient_clip then
            grad_params:mul(gradient_clip / norm)
        end
        
        -- L2 regularisation
        local weight_decay = 3e-5
		grad_params:add(weight_decay, params)
        
		model:updateParameters(learning_rate)
		nce:updateParameters(learning_rate)
		

	end

	local elapse = torch.toc(timer)
	local speed = math.floor(total_samples / elapse)

	total_loss = total_loss / total_samples

	local perplexity = torch.exp(total_loss)

	return total_loss, speed
end

	

local function run(n_epochs)
	
	local val_loss = {}
	local l = eval(2)
	print(torch.exp(l))
	val_loss[0] = l

	local patience = 0

	local learning_rate = g_params.trainer.initial_learning_rate
	local gradient_clip = g_params.trainer.gradient_clip

	for epoch = 1, n_epochs do
		
		-- early stop when no improvement for a long time
		if patience >= g_params.trainer.max_patience then break end
		
		local train_loss, wps = train_epoch(learning_rate, gradient_clip)

		
		val_loss[epoch] = eval(2)
	
		
		--~ Control patience when no improvement
		if val_loss[epoch] >= val_loss[epoch - 1] * g_params.trainer.shrink_factor then
			patience = patience + 1
			learning_rate = learning_rate / g_params.trainer.learning_rate_shrink
		else
			patience = 0
		end
		

		--~ Display training information
		local stat = {train_nce_loss = torch.exp(train_loss) , epoch = epoch,
                valid_perplexity = torch.exp(val_loss[epoch]), LR = learning_rate, speed = wps, patience = patience}

        print(stat)
        
        -- save the trained model
		local save_dir = g_params.trainer.save_dir
		if save_dir ~= nil then
		  if paths.dirp(save_dir) == false then
			  os.execute('mkdir -p ' .. save_dir)	
		  end
		  local save_state = {}
		  save_state.model = model
		  save_state.criterion = criterion
		  save_state.learning_rate = learning_rate
		  save_state.nce = nce
		  save_state.ncecrit = ncecrit
		  torch.save(paths.concat(save_dir, 'model_' .. epoch), save_state)
		end

        -- early stop when learning rate too small
        if learning_rate <= 5e-4 then break end
        
		

	end

	print(torch.exp(eval(3)))
	
end

--~ ppl = torch.exp((eval(2)))
--~ print(ppl)

run(g_params.trainer.n_epochs)
