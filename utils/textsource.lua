

-- This file loads a text dataset.
require 'torch'
require 'paths'
require 'math'
require 'xlua'

local TextSource = torch.class('TextSource')
local preprocessor = require 'utils.preprocessor'
local path = require 'pl.path'

-- config:
-- {
--   threshold : 0
--   shuff : false
--   task : "word"
--   name : "ptb"
--   batch_size : 32
--   nclusters : 0
-- }

function TextSource:__init(config)
    self.batch_size = config.batch_size
    self.root = paths.concat("./data", config.name)
    self.batch_size = config.batch_size
    local clean = config.clean or false

    self.train_txt = paths.concat(self.root, "train.txt")
    self.valid_txt = paths.concat(self.root, "valid.txt")
    self.test_txt = paths.concat(self.root, "test.txt")
    self.vocab_file = paths.concat(self.root, "vocab.txt")
    self.accuracy_valid_txt = paths.concat(self.root, "lambada.txt")
    self.files = {self.train_txt, self.valid_txt, self.test_txt}
    local output_tensors = {}

    if not path.exists(self.vocab_file) then 
        self.vocab_file = nil
    end

    local tensor_file = paths.concat(self.root, "tensors.t7")
    local dict_file = paths.concat(self.root, "dict.t7")
    local lambada_valid_file = paths.concat(self.root, "lambada.t7")

    if ( not path.exists(tensor_file) ) or ( not path.exists(dict_file) ) or clean then
        print("Generating tensor files...")
        self:_build_data(config, dict_file, tensor_file)
    end



    -- get the training, validation and test tensors
    self.sets = {}
    self.dict = torch.load(dict_file)
    local all_data = torch.load(tensor_file)

    -- for i, data in pairs(all_data) do
    self.sets['train'] = all_data[1]
    self.sets['valid'] = all_data[2]
    self.sets['test'] = all_data[3]
    print("Data loaded successfully")
    print("Vocab size: " .. #self.dict.index_to_symbol)

    if path.exists(self.accuracy_valid_txt) then
        if not path.exists(lambada_valid_file) then
            print("Making a list of lambada tensors...")
            local lambada_tensors = preprocessor.text_to_lambada_tensor(self.dict, self.accuracy_valid_txt, config)
            torch.save(lambada_valid_file, lambada_tensors)
        end


        self.lambada_tensors = torch.load(lambada_valid_file)

    end

    collectgarbage()
end

-- return a list of tensor streams
-- a stream is corresponding to one sentence in lambada dataset
--~ function TextSource:get_lambada_streams()

    --~ if self.lambada_tensors == nil then
        --~ return nil
    --~ end
    
    --~ local tensors = self.lambada_tensors

    --~ local reshaped_tensors = {}

    --~ for k, tensor in pairs(tensors) do

        --~ stream_length = tensor:size(1)
        --~ cur_stream = torch.LongTensor(stream_length, 1)
        --~ cur_stream[{{},1}]:copy(tensor[{{1, stream_length}}])

        --~ table.insert(reshaped_tensors, cur_stream)
    --~ end

    --~ return reshaped_tensors
--~ end


--~ This function groups words into clusters based on unigram frequency
function TextSource:create_frequency_clusters(config)

    local n_clusters = config.nclusters
    local vocab_size = self:get_vocab_size()

    -- Default: sqrt(N) (a heuristic though)
    if n_clusters == 0 then
        n_clusters = math.floor(math.sqrt(vocab_size))        
    end

    -- create the cluster index tensors
    self.dict.index_to_cluster = torch.LongTensor(vocab_size):fill(0)
    self.dict.index_to_index_within_cluster = torch.LongTensor(vocab_size):fill(0)

    -- sort the tokens by frequency
    local sorted_freqs, sorted_indx = torch.sort(self.dict.index_to_freq, true)
    local tot_nr_words = self.dict.index_to_freq:sum()
    sorted_freqs:div(math.max(1, tot_nr_words))
    
    local probab_mass = 1.0 / n_clusters
    local current_mass = 0
    local cluster_id = 1
    local within_cluster_index = 0

    for w = 1, vocab_size do
        if current_mass < probab_mass then
            current_mass = current_mass + sorted_freqs[w]
            within_cluster_index = within_cluster_index + 1
        else
            cluster_id = cluster_id + 1
            current_mass = sorted_freqs[w]
            within_cluster_index = 1
        end
        self.dict.index_to_cluster[sorted_indx[w]] = cluster_id
        self.dict.index_to_index_within_cluster[sorted_indx[w]] =
            within_cluster_index
    end
    print("[Created " .. cluster_id .. " clusters.]")

    -- Count how many words per cluster there are.
    local wordsPerCluster = torch.zeros(cluster_id)
    for w = 1, vocab_size do
        local curr_cluster = self.dict.index_to_cluster[w]
        wordsPerCluster[curr_cluster] = wordsPerCluster[curr_cluster] + 1
    end

    -- build reverse index from cluster id back to index
    -- also load the explicit mapping to be used by hsm
    self.dict.mapping = torch.LongTensor(vocab_size, 2)
    for c = 1, cluster_id do
        table.insert(self.dict.cluster_to_index,
                     torch.LongTensor(wordsPerCluster[c]))
    end

    for w = 1, vocab_size do
        local curr_cluster = self.dict.index_to_cluster[w]
        local curr_word = self.dict.index_to_index_within_cluster[w]
        self.dict.cluster_to_index[curr_cluster][curr_word] = w
        self.dict.mapping[w][1] = curr_cluster
        self.dict.mapping[w][2] = curr_word
    end


end

--~ This function assign words to clusters based on random distribution
function TextSource:create_clusters(config)

    local n_clusters = config.nclusters
    local vocab_size = self:get_vocab_size()

    -- Default: 50 words per clusters (a heuristic though)
    if n_clusters == 0 then
        n_clusters = math.floor(vocab_size / 50)        
    end

    -- Create a Tensor to indicate the clusters with size: vocab_size * 2
    -- Word at index i with have cluster id (clusters[i][1]) and in-cluster id (clusters[i][2])
    self.dict.clusters = torch.LongTensor(vocab_size, 2):zero()
    local n_in_each_cluster = math.ceil(vocab_size / n_clusters) 
    -- Randomly cluster words based on a normal distribution
    local _, idx = torch.sort(torch.randn(vocab_size), 1, true)

    local n_in_cluster = {} --number of tokens in each cluster
    local c = 1
    for i = 1, idx:size(1) do
        local word_idx = idx[i] 
        if n_in_cluster[c] == nil then
            n_in_cluster[c] = 1
        else
            n_in_cluster[c] = n_in_cluster[c] + 1
        end
        self.dict.clusters[word_idx][1] = c
        self.dict.clusters[word_idx][2] = n_in_cluster[c]        
        if n_in_cluster[c] >= n_in_each_cluster then
            c = c + 1
        end
        if c > n_clusters then --take care of some corner cases
            c = n_clusters
        end 
    end
    print(string.format('using hierarchical softmax with %d classes', c))


end

function TextSource:create_frequency_tree(config)

    local bin_size = config.bin_size

    if bin_size <= 0 then
        bin_size = 100
    end

    local sorted_freqs, sorted_indx = torch.sort(self.dict.index_to_freq, true)

    local indices = sorted_indx
    local tree = {}
    local id = indices:size(1)

    function recursive_tree(indices)
      if indices:size(1) < bin_size then
            id = id + 1
            tree[id] = indices
            return
          end
      local parents = {}

      for start = 1, indices:size(1), bin_size do
        local stop = math.min(indices:size(1), start+bin_size-1)
        local bin = indices:narrow(1, start, stop-start+1)
        -- print(bin)
        assert(bin:size(1) <= bin_size)

        id = id + 1
        table.insert(parents, id)

        tree[id] = bin
        -- print(id, bin)
      end
      recursive_tree(indices.new(parents))
    end

    recursive_tree(indices) 

    self.dict.tree = tree
    self.dict.root_id = id

    print('Created a frequency softmaxtree with ' .. self.dict.root_id - indices:size(1) .. ' nodes')

end

function TextSource:get_vocab_size()

    return #self.dict.index_to_symbol
end


-- Build vocab and tensor binary files
-- Only use train, valid and test files 
function TextSource:_build_data(config, dict_file, tensor_file)

    local output_tensors = {}
    local dict = preprocessor.build_dictionary(config, self.train_txt, self.vocab_file)

    for i, file in pairs(self.files) do
        
        output_tensors[i] = preprocessor.text_to_tensor(dict, file, config)

    end

    torch.save(dict_file, dict)
    torch.save(tensor_file, output_tensors)
end


-- returns the raw data for train|validation|test (given by set_name)
function TextSource:get_set_from_name(set_name)
    local out = self.sets[set_name]
    if out == nil then
        if set_name == 'nil' then
            error('Set name is nil')
        else
            error('Unknown set name: ' .. set_name)
        end
    end
    return out
end

