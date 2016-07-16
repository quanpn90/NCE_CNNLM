


require('math')
-- local ffivector = require('fb.ffivector')
local pl = require('pl.import_into')()

local Preprocessor = {}

function Preprocessor.build_dictionary(config, trainfname, vocabfname)
    local kMaxDictSize = 500000
    local dict = {}
    dict.symbol_to_index = {}   -- string -> id
    dict.index_to_symbol = {}   -- id -> string
    dict.index_to_freq = torch.Tensor(kMaxDictSize) -- id ->freq
    dict.index_to_cluster = nil -- id ->cluster_id
    dict.index_to_index_within_cluster = nil
    dict.cluster_to_index = {} -- reverse mapping from cluster to word id.
    dict.mapping = nil -- cluster_id to within_cluster_id mapping used by hsm

    local nr_clusters = config.nclusters
    local threshold = config.threshold

    
    local nr_words = 1 -- number of unique words
    local tot_nr_words = 0 -- total number of words in corpus
    -- Add by default an UNK token to be used for the rare entries
    local unk = "<unk>"
    dict.symbol_to_index[unk] = nr_words
    dict.index_to_symbol[nr_words] = unk
    dict.index_to_freq[nr_words] = 0

    -- Add an eos 
    nr_words = nr_words + 1
    dict.symbol_to_index["<eos>"] = nr_words
    dict.index_to_symbol[nr_words] = "<eos>"
    dict.index_to_freq[nr_words] = 0

    if vocabfname ~= nil then
      print("Loading vocab from " .. vocabfname .. " ....")
      local word
      for s in io.lines(vocabfname) do
        ts = s:gsub("\n", "")
        ts = ts:gsub("%s+", "")
        ts = ts:gsub("\t", "")
        -- Add this word to dict
        nr_words = nr_words + 1
        dict.symbol_to_index[ts] = nr_words
        dict.index_to_symbol[nr_words] = ts
        dict.index_to_freq[nr_words] = 0 

        
      end
      print("Loaded "  .. nr_words .. " words")
    else
      print("Vocab file not found. Loading vocab from train file " .. trainfname )
    end

    local cnt = 0

    print("Collecting numbers from " .. trainfname .. " ...")
    for s in io.lines(trainfname) do
        -- remove all the tabs in the string
        s = s:gsub("\t", "")
       -- convert multiple spaces into a single space: this is needed to
       -- make the following pl.utils.split() function return only words
       -- and not white spaes
       s = s:gsub("%s+", " ")
        local words = pl.utils.split(s, ' ')
        for i, word in pairs(words) do
            if word ~= "" then -- somehow the first token is always ""
                if dict.symbol_to_index[word] == nil and vocabfname == nil then
                    nr_words = nr_words + 1
                    dict.symbol_to_index[word] = nr_words
                    dict.index_to_symbol[nr_words] = word
                    dict.index_to_freq[nr_words] = 1
                else
                    local indx = dict.symbol_to_index[word]
                    if indx == nil then
                      -- print(word)
                      indx = dict.symbol_to_index["<unk>"]
                    end
                    dict.index_to_freq[indx] = dict.index_to_freq[indx] + 1
                end
                cnt = cnt + 1
            end
        end
        -- Add \n after every line
        if dict.symbol_to_index["<eos>"] == nil then
            nr_words = nr_words + 1
            dict.symbol_to_index["<eos>"] = nr_words
            dict.index_to_symbol[nr_words] = "<eos>"
            dict.index_to_freq[nr_words] = 1
        else
            local indx = dict.symbol_to_index["<eos>"]
            dict.index_to_freq[indx] = dict.index_to_freq[indx] + 1
        end

        cnt = cnt + 1
    end
    dict.index_to_freq:resize(nr_words)
    -- print(dict.index_to_freq)
    tot_nr_words = dict.index_to_freq:sum()
    print("Unknown tokens in training data: " .. dict.index_to_freq[1]) -- debugging: print the frequency of unknown word
    print("[Done making the dictionary. There are " .. nr_words ..
              " unique words and a total of " .. tot_nr_words ..
              " words in the training set.]")

    -- map rare words to special token and skip corresponding indices
    -- if the specified threshold is greater than 0
    local removed = 0
    local net_nwords = 1
    if threshold > 0 then
        for i = 2, dict.index_to_freq:size(1) do
            local word = dict.index_to_symbol[i]
            if dict.index_to_freq[i] < threshold then
                dict.index_to_freq[1] =
                    dict.index_to_freq[1] + dict.index_to_freq[i]
                dict.index_to_freq[i] = 0
                dict.symbol_to_index[word] = 1
                removed = removed + 1
            else
                -- re-adjust the indices to make them continuous
                net_nwords = net_nwords + 1
                dict.index_to_freq[net_nwords] = dict.index_to_freq[i]
                dict.symbol_to_index[word] = net_nwords
                dict.index_to_symbol[net_nwords] = word
            end
        end
        print('[Removed ' .. removed .. ' rare words. ' ..
                  'Effective number of words ' .. net_nwords .. ']')
        dict.index_to_freq:resize(net_nwords)
    else
        net_nwords = nr_words
    end

    print('There are effectively ' .. net_nwords .. ' words in the corpus.')

    collectgarbage()
    return dict
end


-- This function tokenizes the data (converts words to word_ids)
-- and stores the result in a 1D longTensor
-- Inputs:
--          dict: dictionary
--    filenameIn: full path of the input file
--        config: configuration parameters of the data
-- Outputs:
-- context set: matrix (n_words * context_size (memsize))
-- target set : vector (n_words)
function Preprocessor.text_to_tensor(dict, filenameIn, config)
   print("Processing file " .. filenameIn)
   local unk = "<unk>"
   local offset = config.offset
   local seq_length = config.seq_length
   -- first count how many words there are in the corpus
   -- local all_lines = ffivector.new_string()
   -- local all_lines = {}
   local tot_nr_words = 0 
   local tot_lines = 0
   for s in io.lines(filenameIn) do
       -- store the line
       tot_lines = tot_lines + 1
       -- all_lines[tot_lines] = s
       -- remove all the tabs in the string
       s = s:gsub("\t", "")
       -- remove leading and following white spaces
       s = s:gsub("^%s+", ""):gsub("%s+$", "")
       -- convert multiple spaces into a single space: this is needed to
       -- make the following pl.utils.split() function return only words
       -- and not white spaes
       s = s:gsub("%s+", " ")
       -- count the words
       local words = pl.utils.split(s, ' ')
       tot_nr_words = tot_nr_words + #words -- nr. words in the line
       tot_nr_words = tot_nr_words + 1 -- eos token 
   end
   print('-- total lines: ' .. tot_lines)

   -- now store the lines in the tensor
   local data = torch.Tensor(tot_nr_words + offset, 1):fill(dict.symbol_to_index["<eos>"]) -- Allocate memory for the data sequence (very long) 
   local contexts = torch.IntTensor(tot_nr_words, seq_length)
   local targets = torch.IntTensor(tot_nr_words)
   local id = 0
   local cnt = offset + 1
   local progress_count = 0
   -- for ln = 1, tot_lines do
  for s in io.lines(filenameIn) do
       progress_count = progress_count + 1
       xlua.progress(progress_count, tot_lines)
       -- remove all the tabs in the string
       s = s:gsub("\t", "")
       -- remove leading and following white spaces
       s = s:gsub("^%s+", ""):gsub("%s+$", "")
       -- convert multiple spaces into a single space: this is needed to
       -- make the following pl.utils.split() function return only words
       -- and not white spaces
       s = s:gsub("%s+", " ")
       local words = pl.utils.split(s, ' ')
       for i, word in pairs(words) do
           if word ~= "" then
               if dict.symbol_to_index[word] == nil  then
                   -- print('WARNING: ' .. word .. ' being replaced by ' .. unk)
                   id = dict.symbol_to_index[unk]
               else
                   id = dict.symbol_to_index[word]
               end
               data[cnt][1] = id

               cnt = cnt + 1
           end
       end
       -- Add newline at the end of the sentence

       id = dict.symbol_to_index["<eos>"]
       data[cnt][1] = id

       
       cnt = cnt + 1
       
      


      -- CG every 1000 words to save mem
      if progress_count % 1000 == 0 then
        collectgarbage()
      end
   end


   return data
end



-- This function converts a list of sentences to a list of tensors
-- Input: dictionary, text file and some parameters
-- Output: A table, each key holds one tensor for one sentence 

--~ function Preprocessor.text_to_lambada_tensor(dict, filenameIn, config)
   --~ print("Processing file " .. filenameIn)
   --~ local unk = "<unk>"
   --~ local threshold = config.threshold

   --~ local tensor_data = {}
   
   --~ for s in io.lines(filenameIn) do
       --~ -- remove all the tabs in the string
       --~ s = s:gsub("\t", "")
       --~ -- remove leading and following white spaces
       --~ s = s:gsub("^%s+", ""):gsub("%s+$", "")
       --~ -- convert multiple spaces into a single space: this is needed to
       --~ -- make the following pl.utils.split() function return only words
       --~ -- and not white spaes
       --~ s = s:gsub("%s+", " ")
       --~ -- count the words
       --~ local words = pl.utils.split(s, ' ')

       --~ local tensor = torch.Tensor(#words+1):zero() -- memory allocation
       --~ tensor[1] = dict.symbol_to_index["<eos>"] -- first token is an eos token 
       --~ local cnt = 2
       --~ for i, word in pairs(words) do
           --~ if word ~= "" then

              --~ -- check if word in dictionary
               --~ if dict.symbol_to_index[word] == nil or
               --~ dict.index_to_freq[dict.symbol_to_index[word]] < threshold then
                   --~ -- print('WARNING: ' .. word .. ' being replaced by ' .. unk)
                   --~ id = dict.symbol_to_index[unk] 
               --~ else
                   --~ id = dict.symbol_to_index[word]
               --~ end
               --~ tensor[cnt] = id
               --~ cnt = cnt + 1
           --~ end
       --~ end

       --~ -- put tensor into the table
       --~ tensor_data[#tensor_data + 1] = tensor
   --~ end
  
   --~ return tensor_data
--~ end


return Preprocessor
