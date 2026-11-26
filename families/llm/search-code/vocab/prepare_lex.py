"""Round w2: per-token string tables for the BPE-8192 tokenizer -> lex8192.npz (next to this file, so gpu.sh ships it).
Flags: ws (starts with whitespace), nl (contains newline), send (contains . ! ?), blen (byte-length bucket 0..7),
cls (first non-space char: lower/upper/digit/punct/space-only/other), b1 / bl (first / last byte), bag (bytes present),
and neighbour lists (V x 16, -1 pad) of tokens with the same string up to case+space, up to leading space, the same
first 3 / 2 letters, the same last 3 letters, and the top char-trigram Jaccard neighbours."""
import json, os, sys
import numpy as np
tokf = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "data", "tokenizer", "bpe8192.json")
vocab = json.load(open(tokf))["model"]["vocab"]
V, K = 8192, 16
bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
cs, n = bs[:], 0
for b in range(256):
    if b not in bs:
        bs.append(b); cs.append(256 + n); n += 1
dec = {chr(c): b for b, c in zip(bs, cs)}
raw = [b""] * V
for s, i in vocab.items():
    if i < V:
        raw[i] = bytes(dec.get(ch, 63) for ch in s)
ws = np.array([len(r) > 0 and r[:1] in (b" ", b"\n", b"\t", b"\r") for r in raw])
nl = np.array([b"\n" in r for r in raw])
send = np.array([any(c in r for c in (b".", b"!", b"?")) for r in raw])
blen = np.array([min(7, max(0, len(r) - 1)) for r in raw])
def cl(r):
    t = r.lstrip()
    if not t: return 4
    c = t[:1]
    if c.islower(): return 0
    if c.isupper(): return 1
    if c.isdigit(): return 2
    if t[0] < 128: return 3
    return 5
cls = np.array([cl(r) for r in raw])
b1 = np.array([r[0] if r else 0 for r in raw])
bl = np.array([r[-1] if r else 0 for r in raw])
bag = np.zeros((V, 256), np.uint8)
for i, r in enumerate(raw):
    bag[i, list(set(r))] = 1
txt = [r.decode("utf-8", "replace") for r in raw]
def groups(key):
    nb = -np.ones((V, K), np.int32)
    g = {}
    for i, t in enumerate(txt):
        k = key(t)
        if k: g.setdefault(k, []).append(i)
    for i, t in enumerate(txt):
        k = key(t)
        if not k: continue
        m = [j for j in g[k] if j != i][:K]
        nb[i, :len(m)] = m
    return nb
nbs = dict(nb_case=groups(lambda t: t.strip().lower() or None),
           nb_space=groups(lambda t: t.lstrip(" ") or None),
           nb_pref3=groups(lambda t: t.strip().lower()[:3] if len(t.strip()) >= 3 else None),
           nb_pref2=groups(lambda t: t.strip().lower()[:2] if len(t.strip()) >= 2 else None),
           nb_suf3=groups(lambda t: t.strip().lower()[-3:] if len(t.strip()) >= 4 else None))
# char-trigram Jaccard
tri = {}
rows, cols = [], []
for i, t in enumerate(txt):
    s = "^" + t.strip().lower() + "$"
    for k in set(s[j:j + 3] for j in range(len(s) - 2)):
        rows.append(i); cols.append(tri.setdefault(k, len(tri)))
import scipy.sparse as sp
M = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(V, len(tri)))
inter = (M @ M.T).toarray()
sz = np.asarray(M.sum(1)).ravel()
jac = inter / np.maximum(sz[:, None] + sz[None, :] - inter, 1)
np.fill_diagonal(jac, 0)
nbc = -np.ones((V, K), np.int32)
top = np.argsort(-jac, 1)[:, :8]
for i in range(V):
    m = [j for j in top[i] if jac[i, j] >= 0.3]
    nbc[i, :len(m)] = m
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lex8192.npz")
# round w3: net bracket depth change and quote-character count per token
brk = np.array([sum(r.count(c) for c in (b"(", b"[", b"{")) - sum(r.count(c) for c in (b")", b"]", b"}")) for r in raw])
quo = np.array([r.count(b'"') + r.count(b"'") for r in raw])
np.savez_compressed(out, ws=ws, nl=nl, send=send, blen=blen, cls=cls, b1=b1, bl=bl, bag=bag, nb_char=nbc, brk=brk, quo=quo, **nbs)
for k, v in nbs.items():
    print(k, "tokens with a neighbour:", int((v[:, 0] >= 0).sum()))
print("nb_char", int((nbc[:, 0] >= 0).sum()), "ws", int(ws.sum()), "nl", int(nl.sum()), "send", int(send.sum()))
i = vocab.get("Ġthe"); print("the:", [txt[j] for j in nbs["nb_case"][i] if j >= 0], [txt[j] for j in nbc[i] if j >= 0])
