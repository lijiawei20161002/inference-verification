"""A small, real GPT-style transformer in numpy.

Everything that costs FLOPs goes through `mm()` so we get an *exact* operation
count (2*m*k*n per (m,k)@(k,n) matmul), not an estimate. The model supports two
execution modes that matter for the verification protocol:

  * generate(...)  -- autoregressive greedy decoding with a KV cache (what a
                      provider does to serve a request). T sequential steps.
  * verify_logits  -- a single teacher-forced parallel pass over prompt+output
                      (what a client does to re-score a committed transcript).

We also expose a `cheat` mode that early-exits after half the layers, i.e. a
provider that runs a cheaper approximation to save compute.
"""
import numpy as np

# ----------------------------------------------------------------------------
# Exact FLOP accounting
# ----------------------------------------------------------------------------
class Flops:
    def __init__(self): self.n = 0
    def reset(self): self.n = 0

_COUNTER = Flops()

def mm(a, b):
    """2D matmul with exact multiply-add*2 FLOP accounting."""
    m, k = a.shape
    k2, n = b.shape
    assert k == k2, (a.shape, b.shape)
    _COUNTER.n += 2 * m * k * n
    return a @ b

def flops_reset(): _COUNTER.reset()
def flops_get():   return _COUNTER.n

# ----------------------------------------------------------------------------
# ops
# ----------------------------------------------------------------------------
def layernorm(x):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + 1e-5)

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.79788456 * (x + 0.044715 * x**3)))

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

# ----------------------------------------------------------------------------
# config + weights
# ----------------------------------------------------------------------------
class Config:
    def __init__(self, d_model=128, n_head=4, n_layer=6, d_ff=512,
                 vocab=256, max_ctx=256):
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.n_layer = n_layer
        self.d_ff = d_ff
        self.vocab = vocab
        self.max_ctx = max_ctx

class Model:
    def __init__(self, cfg: Config, seed=0):
        self.cfg = cfg
        rng = np.random.default_rng(seed)
        d, dff, V = cfg.d_model, cfg.d_ff, cfg.vocab
        s = 0.02
        self.Wemb = rng.normal(0, s, (V, d)).astype(np.float32)
        self.Wpos = rng.normal(0, s, (cfg.max_ctx, d)).astype(np.float32)
        self.layers = []
        for _ in range(cfg.n_layer):
            self.layers.append(dict(
                Wqkv=rng.normal(0, s, (d, 3 * d)).astype(np.float32),
                Wo=rng.normal(0, s, (d, d)).astype(np.float32),
                W1=rng.normal(0, s, (d, dff)).astype(np.float32),
                W2=rng.normal(0, s, (dff, d)).astype(np.float32),
            ))

    # ---- teacher-forced parallel pass over a full token sequence ----
    def forward_full(self, tokens, n_layers=None):
        """tokens: 1D int array length S. Returns logits (S, vocab).

        This is the client's *verification* pass: one parallel forward over the
        whole committed sequence, causal-masked. n_layers<n_layer => early exit.
        """
        cfg = self.cfg
        if n_layers is None: n_layers = cfg.n_layer
        S = len(tokens)
        x = self.Wemb[tokens] + self.Wpos[:S]           # (S, d)
        nh, dh = cfg.n_head, cfg.d_head
        mask = np.triu(np.full((S, S), -1e9, np.float32), 1)
        for li in range(n_layers):
            W = self.layers[li]
            h = layernorm(x)
            qkv = mm(h, W['Wqkv'])                        # (S, 3d)
            q, k, v = np.split(qkv, 3, axis=-1)
            q = q.reshape(S, nh, dh); k = k.reshape(S, nh, dh); v = v.reshape(S, nh, dh)
            out = np.empty((S, nh, dh), np.float32)
            for hh in range(nh):
                sc = mm(q[:, hh], k[:, hh].T) / np.sqrt(dh)   # (S,S)
                sc = sc + mask
                a = softmax(sc, -1)
                out[:, hh] = mm(a, v[:, hh])                   # (S,dh)
            attn = mm(out.reshape(S, cfg.d_model), W['Wo'])
            x = x + attn
            h2 = layernorm(x)
            x = x + mm(gelu(mm(h2, W['W1'])), W['W2'])
        x = layernorm(x)
        logits = mm(x, self.Wemb.T)                       # (S, vocab)
        return logits

    # ---- batched teacher-forced pass (a client re-scoring many audits) ----
    def forward_full_batch(self, tokens_batch, n_layers=None):
        """tokens_batch: (B, S) int. Returns logits (B, S, vocab).

        Same computation as forward_full but over a batch, which is how a real
        client re-scores its sampled audits in one shot. FLOPs scale as B*(one
        forward), so per-request cost is unchanged; wall-clock collapses.
        """
        cfg = self.cfg
        if n_layers is None: n_layers = cfg.n_layer
        B, S = tokens_batch.shape
        nh, dh, d = cfg.n_head, cfg.d_head, cfg.d_model
        x = self.Wemb[tokens_batch] + self.Wpos[:S][None]        # (B,S,d)
        mask = np.triu(np.full((S, S), -1e9, np.float32), 1)
        for li in range(n_layers):
            W = self.layers[li]
            h = layernorm(x)
            qkv = mm(h.reshape(B * S, d), W['Wqkv']).reshape(B, S, 3 * d)
            q, k, v = np.split(qkv, 3, axis=-1)
            q = q.reshape(B, S, nh, dh); k = k.reshape(B, S, nh, dh); v = v.reshape(B, S, nh, dh)
            out = np.empty((B, S, nh, dh), np.float32)
            for hh in range(nh):
                qh, kh, vh = q[:, :, hh], k[:, :, hh], v[:, :, hh]   # (B,S,dh)
                sc = np.einsum('bsd,btd->bst', qh, kh) / np.sqrt(dh)
                _COUNTER.n += 2 * B * S * S * dh
                sc = sc + mask
                a = softmax(sc, -1)
                out[:, :, hh] = np.einsum('bst,btd->bsd', a, vh)
                _COUNTER.n += 2 * B * S * S * dh
            attn = mm(out.reshape(B * S, d), W['Wo']).reshape(B, S, d)
            x = x + attn
            h2 = layernorm(x)
            ff = mm(gelu(mm(h2.reshape(B * S, d), W['W1'])), W['W2']).reshape(B, S, d)
            x = x + ff
        x = layernorm(x)
        logits = mm(x.reshape(B * S, d), self.Wemb.T).reshape(B, S, cfg.vocab)
        return logits

    # ---- autoregressive greedy generation with KV cache ----
    def generate(self, prompt, T, n_layers=None):
        """prompt: 1D int array. Generate T tokens greedily with a KV cache.

        Returns (out_tokens (T,), out_logits (T, vocab)). This mirrors real
        serving cost: L-token prefill + T sequential single-token decode steps.
        n_layers<n_layer models a cheating provider (early exit).
        """
        cfg = self.cfg
        if n_layers is None: n_layers = cfg.n_layer
        nh, dh, d = cfg.n_head, cfg.d_head, cfg.d_model
        L = len(prompt)
        # per-layer KV caches
        Kc = [np.zeros((0, nh, dh), np.float32) for _ in range(n_layers)]
        Vc = [np.zeros((0, nh, dh), np.float32) for _ in range(n_layers)]

        def step(tok_id, pos):
            x = (self.Wemb[tok_id] + self.Wpos[pos]).reshape(1, d)
            for li in range(n_layers):
                W = self.layers[li]
                h = layernorm(x)
                qkv = mm(h, W['Wqkv'])                    # (1,3d)
                q, k, v = np.split(qkv, 3, axis=-1)
                q = q.reshape(1, nh, dh); k = k.reshape(1, nh, dh); v = v.reshape(1, nh, dh)
                Kc[li] = np.concatenate([Kc[li], k], 0)   # (c, nh, dh)
                Vc[li] = np.concatenate([Vc[li], v], 0)
                c = Kc[li].shape[0]
                out = np.empty((1, nh, dh), np.float32)
                for hh in range(nh):
                    sc = mm(q[:, hh], Kc[li][:, hh].T) / np.sqrt(dh)   # (1,c)
                    a = softmax(sc, -1)
                    out[:, hh] = mm(a, Vc[li][:, hh])                  # (1,dh)
                attn = mm(out.reshape(1, d), W['Wo'])
                x = x + attn
                h2 = layernorm(x)
                x = x + mm(gelu(mm(h2, W['W1'])), W['W2'])
            x = layernorm(x)
            logits = mm(x, self.Wemb.T)                   # (1, vocab)
            return logits[0]

        # prefill: feed prompt tokens (positions 0..L-1)
        for p in range(L):
            last_logits = step(int(prompt[p]), p)
        out_tokens = np.empty(T, np.int64)
        out_logits = np.empty((T, cfg.vocab), np.float32)
        cur = int(np.argmax(last_logits))                 # first generated token
        for t in range(T):
            out_tokens[t] = cur
            # logits that *produced* this token were computed at previous step;
            # recompute cleanly: we store the logits used to pick token t
            # (i.e. the distribution over token t given context so far).
            lg = step(cur, L + t)
            out_logits[t] = last_logits                   # dist that chose out_tokens[t]
            last_logits = lg
            cur = int(np.argmax(lg))
        return out_tokens, out_logits
