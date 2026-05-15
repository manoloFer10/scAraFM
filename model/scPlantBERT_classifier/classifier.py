import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader


class scPlantBERTClassifier(nn.Module):
    def __init__(
        self,
        dim,                                # dimension
        depth,                              # layers
        heads,                              # heads
        dim_head,                           # dim of head
        local_attn_heads = 0,               # num of local attention heads, (heads - local_attn_heads) is num of global performers
        local_window_size = 256,            # window size of local attention
        causal = False,                     # autoregressive or not
        ff_mult = 4,                        # dim of intermediate features after attention / dim of input features
        nb_features = None,                 # number of random features, if not set, will default to (d * log(d)), where d is the dimension of each head ?? what is random feature ??
        feature_redraw_interval = 1000,     # how frequently to redraw the projection matrix, the more frequent, the slower the training
        reversible = False,                 # reversible layers, from Reformer (save memory)
        ff_chunks = 1,                      # chunk feedforward layer, from Reformer
        generalized_attention = False,      # defaults to softmax approximation, but can be set to True for generalized attention ?? what is generalized attention ??
        kernel_fn = nn.ReLU(),              # the kernel function to be used, if generalized attention is turned on, defaults to Relu
        use_scalenorm = False,              # use scale norm, from 'Transformers without Tears' paper, a substitute for LayerNorm, priority: scalenorm.rezero.layernorm
        use_rezero = False,                 # use Rezero or not, from 'Rezero is all you need' paper, a substitute for LayerNorm, priority: scalenorm.rezero.layernorm
        ff_glu = False,                     # use GLU (Gated Linear Units) variant for feedforward
        ff_dropout = 0.,                    # feedforward dropout
        attn_dropout = 0.,                  # post-attention dropout
        cross_attend = False,               # ??
        no_projection = False,              # ??
        auto_check_redraw = True,           # ??
        qkv_bias = True,                    # ??
    ):
        super().__init__()
        layers = nn.ModuleList([])
        local_attn_heads = cast_tuple(local_attn_heads)
        local_attn_heads = local_attn_heads * depth if len(local_attn_heads) == 1 else local_attn_heads
        assert len(local_attn_heads) == depth, 'tuple specifying number of local attention heads per depth must be equal to the total depth'
        assert all(map(lambda n: n >= 0 and n <= heads, local_attn_heads)), 'local attention head value must be less than the total number of heads'

        if use_scalenorm:
            wrapper_fn = partial(PreScaleNorm, dim)
        elif use_rezero:
            wrapper_fn = ReZero
        else:
            wrapper_fn = partial(PreLayerNorm, dim)

        for _, local_heads in zip(range(depth), local_attn_heads):
            layers.append(nn.ModuleList([
                wrapper_fn(SelfAttention(dim, causal = causal, heads = heads, dim_head = dim_head, local_heads = local_heads, local_window_size = local_window_size, nb_features = nb_features, generalized_attention = generalized_attention, kernel_fn = kernel_fn, dropout = attn_dropout, no_projection = no_projection, qkv_bias = qkv_bias)),
                wrapper_fn(Chunk(ff_chunks, FeedForward(dim, mult = ff_mult, dropout = ff_dropout, glu = ff_glu), along_dim = 1))
            ]))
            # if no need cross_attend(decoder), begin next cycle
            if not cross_attend:
                continue
            layers.append(nn.ModuleList([
                wrapper_fn(SelfAttention(dim, heads = heads, dim_head = dim_head, nb_features = nb_features, generalized_attention = generalized_attention, kernel_fn = kernel_fn, dropout = attn_dropout, no_projection = no_projection)),
                wrapper_fn(Chunk(ff_chunks, FeedForward(dim, mult = ff_mult, dropout = ff_dropout, glu = ff_glu), along_dim = 1))
            ]))

        execute_type = ReversibleSequence if reversible else SequentialSequence

        route_attn = ((True, False),) * depth * (2 if cross_attend else 1) # ((True, False), (True, False), (True, False), (True, False), (True, False), (True, False))
        route_context = ((False, False), (True, False)) * depth
        attn_route_map = {'mask': route_attn, 'pos_emb': route_attn}
        context_route_map = {'context': route_context, 'context_mask': route_context} if cross_attend else {}
        self.net = execute_type(layers, args_route = {**attn_route_map, **context_route_map})

        # keeping track of when to redraw projections for all attention layers
        self.auto_check_redraw = auto_check_redraw
        self.feature_redraw_interval = feature_redraw_interval
        self.register_buffer('calls_since_last_redraw', torch.tensor(0))

    def fix_projection_matrices_(self):
        self.feature_redraw_interval = None

    def check_redraw_projections(self):
        if not self.training:
            return

        if exists(self.feature_redraw_interval) and self.calls_since_last_redraw >= self.feature_redraw_interval:
            device = get_module_device(self)

            fast_attentions = find_modules(self, FastAttention)
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix(device)

            self.calls_since_last_redraw.zero_()
            return

        self.calls_since_last_redraw += 1

    def forward(self, x, output_attentions = False, **kwargs):
        if self.auto_check_redraw:
            self.check_redraw_projections()
        return self.net(x, output_attentions = output_attentions, **kwargs)

class PerformerLM(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,                         # num of tokens
        max_seq_len,                        # max length of sequence
        dim,                                # dim of tokens
        depth,                              # layers
        heads,                              # num of heads
        dim_head = 64,                      # dim of heads
        local_attn_heads = 0,
        local_window_size = 256,
        causal = False,
        ff_mult = 4,
        nb_features = None,
        feature_redraw_interval = 1000,
        reversible = False,
        ff_chunks = 1,
        ff_glu = False,
        emb_dropout = 0.,
        ff_dropout = 0.,
        attn_dropout = 0.,
        generalized_attention = False,
        kernel_fn = nn.ReLU(),
        use_scalenorm = False,
        use_rezero = False,
        cross_attend = False,
        no_projection = False,
        tie_embed = False,                  # False: output is num of tokens, True: output is dim of tokens  //multiply final embeddings with token weights for logits, like gpt decoder//
        g2v_position_emb = None,            # priority: gene2vec, no embedding
        # g2v_position_emb = "./data/gene2vec_16906.npy",
        auto_check_redraw = True,
        qkv_bias = False
    ):
        super().__init__()
        local_attn_heads = cast_tuple(local_attn_heads)

        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(num_tokens, dim)

        if g2v_position_emb:
            self.pos_emb = Gene2VecPositionalEmbedding(dim, max_seq_len, path=None)
            self.layer_pos_emb = Always(None)
        else:
            self.pos_emb = torch.zeros_like
            self.layer_pos_emb = Always(None)

        self.dropout = nn.Dropout(emb_dropout)

        self.performer = Performer(dim, depth, heads, dim_head, local_attn_heads, local_window_size, causal, ff_mult, nb_features, feature_redraw_interval, reversible, ff_chunks, generalized_attention, kernel_fn, use_scalenorm, use_rezero, ff_glu, ff_dropout, attn_dropout, cross_attend, no_projection, auto_check_redraw, qkv_bias)
        self.norm = nn.LayerNorm(dim)
        self.to_out = nn.Linear(dim, num_tokens) if not tie_embed else None

    def check_redraw_projections(self):
        self.performer.check_redraw_projections()

    def fix_projection_matrices_(self):
        self.performer.fix_projection_matrices_()

    def forward(self, x, return_encodings = False, output_attentions = False, **kwargs):
        b, n, device = *x.shape, x.device
        assert n <= self.max_seq_len, f'sequence length {n} must be less than the max sequence length {self.max_seq_len}'

        # token and positional embedding
        x = self.token_emb(x)
        if output_attentions:
            x.requires_grad_()    # used for attn_map output
        x += self.pos_emb(x)
        x = self.dropout(x)

        # performer layers
        layer_pos_emb = self.layer_pos_emb(x)

        if output_attentions:
            x, attn_weights = self.performer(x, pos_emb = layer_pos_emb, output_attentions = output_attentions, **kwargs)
            # norm and to logits
            x = self.norm(x)
            if return_encodings:
                return x, attn_weights

            if exists(self.to_out):
                return self.to_out(x), attn_weights

            return (x @ self.token_emb.weight.t()), attn_weights
        else:
            x = self.performer(x, pos_emb = layer_pos_emb, output_attentions = output_attentions, **kwargs)

            # norm and to logits
            x = self.norm(x)
            if return_encodings:
                return x

            if exists(self.to_out):
                x = self.to_out(x)
                return x

            return x @ self.token_emb.weight.t()