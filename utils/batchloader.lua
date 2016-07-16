
-- This file implements an intermediate class between the textsource and the trainer
-- The textsource outputs a text stream (with batch size)
-- And the batch loader will select the batch (with seq length) and give the trainer

local BatchLoader = torch.class('BatchLoader')

function BatchLoader:__init(config, text_source)

	self.text_source = text_source
	self.edim = ed
	self.seq_length = config.seq_length
	self.memsize = self.seq_length
	self.batch_size = config.batch_size
	self.vocab_size = self.text_source:get_vocab_size()
    self.eos = self.text_source.dict.symbol_to_index["<eos>"]
    self.offset = config.offset


	self.split_sizes = {}  
	self.indices = {}
	self.streams = {}

	self.streams[1] = text_source:get_set_from_name('train')
	self.streams[2] = text_source:get_set_from_name('valid')
	self.streams[3] = text_source:get_set_from_name('test')

	for i = 1, 3 do

		local stream = self.streams[i]
        -- because of patching
		local length = stream:size(1) - self.offset

        print(length)
		local indices
		if i == 1 then
			-- randomed permutation indexing (shuffling)
			indices = torch.randperm(length):long():split(self.batch_size)
		else
			-- in testing we don't care about shuffling
			indices = torch.Tensor(length)
			local k = 0
			indices:apply(function() k = k + 1 return k end)
			indices = indices:long():split(self.batch_size)
		end

		self.indices[i] = indices
		self.split_sizes[i] = #indices
	end

	self.batch_idx = {0, 0, 0}
	self.ntrain = self.split_sizes[1]
	print(string.format('data load done. Number of batches in train: %d, val: %d, test: %d', 
          self.split_sizes[1], self.split_sizes[2], self.split_sizes[3]))
    collectgarbage()

    self.y = torch.ones(1)

end

function BatchLoader:get_vocab_size()
	return self.vocab_size
end

function BatchLoader:reset_batch_pointer(split, batch_idx)
    batch_idx = batch_idx or 0
    self.batch_idx[split] = batch_idx
end

function BatchLoader:next_batch(split_idx)

	split_idx = split_idx or 1
    -- split_idx is integer: 1 = train, 2 = val, 3 = test
    self.batch_idx[split_idx] = self.batch_idx[split_idx] + 1
    if self.batch_idx[split_idx] > self.split_sizes[split_idx] then
        self.batch_idx[split_idx] = 1 -- cycle around to beginning
    end
    -- pull out the correct next batch
    local idx = self.batch_idx[split_idx]
    local stream = self.streams[split_idx]

    local indices = self.indices[split_idx][idx]
    local batch_size = indices:size(1)

    local target = torch.Tensor(batch_size)
    -- contexts are filled with <eos> as a default token to pad
    local context = torch.Tensor(batch_size, self.memsize):fill(self.eos)

    -- get target and context
    for b = 1, batch_size do

        -- note that the stream is patched with offset <eos> before the first word
        m = indices[b] + self.offset
        target[b] = stream[m]

        context[b]:copy(
                stream:narrow(1, m - self.memsize, self.memsize))
    end
	
	local time = torch.Tensor(batch_size, self.memsize)
	for t = 1, self.memsize do
		time:select(2, t):fill(t)
	end

    -- ship to GPU
    -- input = input:cuda()
    target = target:float():cuda()
    context = context:float():cuda()
    time = time:cuda()

    -- local x = {input, context, target, time}
    -- local y = self.y
    
    return context, target, time
     

end

-- function BatchLoader:get_lambada_streams()

-- 	return self.text_source:get_lambada_streams()
-- end
