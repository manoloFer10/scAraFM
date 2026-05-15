import math
from functools import reduce
import torch

MASK_PROB = 0.15 # args.mask_prob
REPLACE_PROB = 0.9 # args.replace_prob
RANDOM_TOKEN_PROB = 0.

from model.multi_file_dataset import N_CLASSES
MASK_TOKEN_ID = N_CLASSES
PAD_TOKEN_ID = N_CLASSES #TODO> come to an agreement with this. MANU: changed from N_CLASSES -1 to N_CLASSES, because N_CLASSES -1 is EOS token
MASK_IGNORE_TOKEN_IDS = [0]

# get the random prob matrix and True means smaller than prob threshold
def prob_mask_like(t, prob):
    return torch.zeros_like(t).float().uniform_(0, 1) < prob

# get the mask matrix which cannot be masked
def mask_with_tokens(t, token_ids):
    init_no_mask = torch.full_like(t, False, dtype=torch.bool)
    mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
    return mask

def get_mask_subset_with_prob(mask, prob):
    batch, seq_len, device = *mask.shape, mask.device
    max_masked = math.ceil(prob * seq_len)      # num of mask of a single sequence in average
    num_tokens = mask.sum(dim=-1, keepdim=True)     # num of pure tokens of each sequence except special tokens
    mask_excess = torch.cat((torch.zeros(0), torch.arange(mask.size(-1)).repeat(mask.size(0)))).reshape(mask.size(0),mask.size(-1)).to(device)
    mask_excess = (mask_excess >= (num_tokens * prob).ceil())        # only 15% of pure tokens can be masked
    mask_excess = mask_excess[:, :max_masked]       # get difference between 15% of pure tokens and 15% of all tokens
    rand = torch.rand((batch, seq_len), device=device).masked_fill(~mask, -1e9)     # rand (0-1) as prob, special token use -1e9
    _, sampled_indices = rand.topk(max_masked, dim=-1)      # get index of topk prob to mask
    sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)        # delete difference of mask not pure
    new_mask = torch.zeros((batch, seq_len + 1), device=device)     # get (batch, seq_len) shape zero matrix
    new_mask.scatter_(-1, sampled_indices, 1)       # set masks in zero matrix as 1
    return new_mask[:, 1:].bool()       # the final mask, True is mask

def data_mask(data,
       mask_prob = MASK_PROB,
       replace_prob = REPLACE_PROB,
       num_tokens = None,
       random_token_prob = RANDOM_TOKEN_PROB,
       mask_token_id = MASK_TOKEN_ID,
       pad_token_id = PAD_TOKEN_ID,
       mask_ignore_token_ids = MASK_IGNORE_TOKEN_IDS):
    
    mask_ignore_token_ids = set([*mask_ignore_token_ids, pad_token_id])
    
    # do not mask [pad] tokens, or any other tokens in the tokens designated to be excluded ([cls], [sep])
    # also do not include these special tokens in the tokens chosen at random
    
    # ignore_token as True, will not be masked later
    no_mask = mask_with_tokens(data, mask_ignore_token_ids)   
    
    # get the True/False mask matrix
    mask = get_mask_subset_with_prob(~no_mask, mask_prob)      
    
    # get mask indices
    ## mask_indices = torch.nonzero(mask, as_tuple=True)   # get the index of mask(nonzero value of mask matrix)
    # mask input with mask tokens with probability of `replace_prob` (keep tokens the same with probability 1 - replace_prob)
    masked_input = data.clone().detach()
    
    # if random token probability > 0 for mlm
    if random_token_prob > 0:
        
        assert num_tokens is not None, """
            num_tokens keyword must be supplied when instantiating MLM if using random token replacement
        """
        
        # get the mask matrix of random token replace
        random_token_prob = prob_mask_like(data, random_token_prob)

        # generate random token matrix with the same shape as input
        random_tokens = torch.randint(0, num_tokens, data.shape, device=data.device)
        
        # not masked matrix for the random token matrix
        random_no_mask = mask_with_tokens(random_tokens, mask_ignore_token_ids)        

        # get the pure mask matrix of random token replace
        random_token_prob = random_token_prob & (~random_no_mask)

        # index of random token replace
        random_indices = torch.nonzero(random_token_prob, as_tuple=True)        

        # replace some tokens by random token
        masked_input[random_indices] = random_tokens[random_indices]
    
    # [mask] input
    # get the mask matrix of token being masked
    replace_prob = prob_mask_like(data, replace_prob)     
    
    # get the data has been masked by mask_token
    masked_input = masked_input.masked_fill(mask * replace_prob, mask_token_id)        
    
    # mask out any tokens to padding tokens that were not originally going to be masked
    # the label of masked tokens
    labels = data.masked_fill(~mask, pad_token_id)        
    
    return masked_input, labels
