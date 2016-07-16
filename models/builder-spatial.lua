--~ th train.lua -dset ptb -name spatial_lm -nlayers 1 -nkernels 2 -highway_mlp 1 -vsize 128 -kmax 2 -ksize 3 -dropout 0.3


local ModelBuilder = torch.class('ModelBuilder')

function ModelBuilder:make_net(config, vocab_size)

	-- input: batch_size * context_size
	-- for example 128 * 32
	
	-- Time : batch * context_size
	--~ Value from 1 to context size - we want to make convolution less time invariant.
	
	
	local input = nn.Identity()()
	local time = nn.Identity()()
	
	--~ for experimental purposes ...
	local pooling = false
	local nin = config.nin
	local highway_conv = config.highway_conv
	local k = config.kmax
	if k == -1 then k = 2 end

	--~ selecting the non linear functions ...
	local nonlinear_func
	if config.non_linearity == 'tanh' then
		nonlinear_func = cudnn.Tanh
	elseif config.non_linearity == 'sigmoid' then
		nonlinear_func = cudnn.Sigmoid
	elseif config.non_linearity == 'relu' then
		nonlinear_func = cudnn.ReLU
	end
	
	hsm = hsm or false

	-- Projection layer
	local lookup = nn.LookupTable(vocab_size, config.vec_size)
	lookup.weight:uniform(-0.1, 0.1)
	
	local lookup_time = nn.LookupTable(config.memsize, config.vec_size)
	lookup_time.weight:uniform(-0.1, 0.1)
	
	local word_emb = nn.MulConstant(0.9, true)(lookup(input))
	local time_emb = nn.MulConstant(0.1, true)(lookup_time(time))
	
	local embeddings = nn.CAddTable()({word_emb, time_emb})


	local layer1 = {}
	local final_layer_size = 0
	local total_conv_weights = 0
	local kernel = config.kernel_size
	
	for i = 1, config.nkernels do
		
			
		local output_size = config.memsize
		local conv_output, activated_conv
		local conv_width = config.vec_size
		local conv_depth = config.vec_size
		local normalised_output, output

	
		kernel = config.kernel_size + 2 * (	i-1)
		
		if kernel > 9 then kernel = 1 end
		
		print("Convolution kernel " .. i .. " with size " .. kernel)
		
		local pad = (kernel - 1) / 2
		local stride = 1
		
		--~ convert embeddings to image
		local image = nn.Reshape(1, output_size, conv_width, true)(embeddings)
		image = nn.Dropout(config.cnndropout)(image)
		local conv = cudnn.SpatialConvolution(1, conv_depth, conv_width, kernel, 1, stride, 0, pad)
		local init_value = 2 / (conv_width * conv_depth * kernel)
		
		init_value = math.sqrt(init_value)
		
		init_value = math.max(init_value, 0.01)
		--~ print(init_value)
		conv.weight:normal(0, init_value)
		conv.bias:zero()
		total_conv_weights = total_conv_weights + conv.weight:nElement() + conv.bias:nElement()
		
		local conv_output = conv(image)
		--~ Add batch norm at "image" level
		--~ May help when using multiple layer cnn
		local conv_normalised = cudnn.SpatialBatchNormalization(conv_depth, 1e-3)(conv_output)
		conv_output = conv_normalised
		
		--~ Feature activation - Normally ReLU is the best
		local activated_conv = nonlinear_func()(conv_output)
		
		
		--~ Recomputing output sizes after convolution
		output_size = math.floor((output_size - kernel  + pad * 2)/stride) + 1
		conv_width = conv_depth --~ Width of new "image" = depth of convolution
		
		--~ Second conv layer - Network in network
		if nin == true then
			print("Using another layer in convolution for kernel " .. i)
			image = nn.Reshape(1, output_size, conv_width, true)(activated_conv)
			image = nn.Dropout(config.cnndropout)(image)
			local conv2 = cudnn.SpatialConvolution(1, conv_depth, conv_width, 1, 1, 1, 0, 0)
			init_value = 2 / (conv_width * conv_depth)
			init_value = math.sqrt(init_value)
			init_value = math.max(init_value, 0.01)
			conv2.weight:normal(0, init_value)
			conv2.bias:zero()
			total_conv_weights = total_conv_weights + conv2.weight:nElement() + conv2.bias:nElement()
			conv_output = conv2(image)
			conv_normalised = cudnn.SpatialBatchNormalization(conv_depth, 1e-3)(conv_output)
			activated_conv = nonlinear_func()(conv_normalised)
		end
		
		--~ output: 16 * 128
		local output = nn.Reshape(output_size, conv_width, true)(activated_conv):annotate{name='conv_'.. i}
		
		
		-- an optional parameter to combine the CNN output with input (residual ?)
		if highway_conv == true then
			print("Using highway convolution for kernel " .. i) 
			local bias = -2
			local temp = nn.Reshape(output_size, conv_width, true)(conv_normalised)
			local transform_gate = cudnn.Sigmoid()(nn.AddConstant(bias)(temp))
			local carry_gate = nn.AddConstant(1)(nn.MulConstant(-1)(transform_gate))
			
			output = nn.CAddTable()({
			  nn.CMulTable()({transform_gate, output}),
			  nn.CMulTable()({carry_gate, embeddings})})
		end
		
		output = cudnn.ReLU()(output)
		output:annotate{name='conv_'.. i}
		
		
		
		local reshaped_output = nn.Reshape(output_size * conv_width, true)(output)
		
		--~ Another batch norm here. This is very important for performance
		local normalised = cudnn.BatchNormalization(output_size * conv_width, 1e-3)(reshaped_output)
		
		-- Mapping features into k*word_vec dim using a linear combination
		-- More parameters but faster + work better than TemporalMaxPooling
		local feature
		if pooling == false then
			local mapping = nn.Linear(output_size * conv_width, k * conv_width)
			mapping.weight:uniform(-0.01, 0.01)
			mapping.bias:uniform(-0.01, 0.01)
			feature = mapping(nn.Dropout(config.dropout)(normalised))
			feature:annotate{name='mapping_' .. i}
			feature = cudnn.ReLU()(feature)
		else
			local reshaped = nn.Reshape(output_size, conv_width, true)(normalised)
			local pooling = nn.TemporalKMaxPooling(k)(reshaped)
			local max_time = nn.Reshape(k * conv_width, true)(pooling)
			--~ feature = cudnn.BatchNormalization(k * conv_width, 0.003)(max_time)
			feature = max_time
		end
			
    	
    	final_layer_size = final_layer_size + k * conv_width
		table.insert(layer1, feature)
		
		--~ all batch normalization modules are needed for the best performances ...
	end

	-- Concatenate output features
	local conv_layer_concat

	if #layer1 > 1 then
		conv_layer_concat = nn.JoinTable(2)(layer1)
	else
		conv_layer_concat = layer1[1]
	end

	local last_layer = conv_layer_concat
	

	local output, output_size
	
	
	if config.highway_mlp > 0 then
	-- use highway layers
		local hwdropout	 = nn.Dropout(config.dropout)(last_layer)
		local HighwayMLP = require 'models.HighwayMLP'
		local highway = HighwayMLP.mlp(final_layer_size, config.highway_mlp, config.dropout, -2, nonlinear_func())
		last_layer = highway(hwdropout)
	else
		--~ fill two layers with the same params as a highway layer
		--~ Same structure as Devlin et al (2014)
		
		local dropout = nn.Dropout(config.dropout)(last_layer)
		local linear = nn.Linear( final_layer_size, final_layer_size)
		linear.weight:uniform(-0.01, 0.01)
		linear.bias:uniform(-0.01, 0.01)
		
		local left = linear(dropout)
		left = cudnn.ReLU()(left)
		
		linear = nn.Linear( final_layer_size, final_layer_size)
		linear.weight:uniform(-0.01, 0.01)
		linear.bias:uniform(-0.01, 0.01)
		local right = linear(dropout)
		right = cudnn.ReLU()(right)
		
		last_layer = nn.CAddTable()({left, right})
	end
	
	local dropout = nn.Dropout(config.dropout)(last_layer)
	last_layer = dropout


	--~ Final step: Going to prediction
	--~ if hsm == false then
		--~ local linear = nn.Linear( final_layer_size, vocab_size)
		--~ output_size = vocab_size
		--~ linear.weight:uniform(-0.01, 0.01)
		--~ linear.bias:uniform(-0.01, 0.01)
		--~ local softmax = cudnn.LogSoftMax()

		--~ output = softmax(linear(last_layer))

	--~ else
		--~ Prediction with HSM will be done later (HSM has a different way to update parameters)
		--~ output = nn.Identity()(last_layer)
		--~ output_size = final_layer_size
	--~ end
	
	output = nn.Identity()(last_layer)
	output_size = final_layer_size

	local model = nn.gModule({input, time}, {output})
	--~ model.verbose = true
	--~ graph.dot(model.fg, 'LanguageModel', 'graph/LanguageModel')

	
	--~ Ship model to GPU
	model:cuda()
	
	print("Total number of weights for convolution: " .. total_conv_weights)

	return model, output_size


end

return ModelBuilder
