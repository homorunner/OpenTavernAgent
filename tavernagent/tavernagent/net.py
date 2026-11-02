"""A small pre-norm residual network in numpy — no framework, no GPU.

The network keeps a shared card space and slot-specific input projection.

    numeric features + flattened shared card embeddings -> 192 -> 3 residual blocks
    -> {245 policy logits, 1 value}

Embedding dimension defaults to 8.
Latency is measured by bench.py.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import taverngym

N_ACTIONS = 245
N_FEATURES = taverngym.OBS_FLOATS
N_CARD_SLOTS = taverngym.OBS_CARDS
# Exclude historical opponent entities while retaining public summaries.
_LAYOUT = dict(taverngym.layout())
N_OPP_BOARD_SLOTS = _LAYOUT["N_OPP"] * _LAYOUT["N_BOARD"]
N_BUDGET_FEATURES = _LAYOUT["ACTION_BUDGET_F"]
BUDGET_FEATURE_START = N_FEATURES - N_BUDGET_FEATURES
OPP_BOARD_FEATURE_START = BUDGET_FEATURE_START - N_OPP_BOARD_SLOTS * _LAYOUT["MINION_F"]
# Hero selection and terminal rewards are handled outside the network.
# Shop freeze is encoded per item, at item-header offset 6.
INPUT_FEATURE_INDICES = np.array(
    [i for i in range(OPP_BOARD_FEATURE_START) if i not in (1, 2, 15)]
    + list(range(BUDGET_FEATURE_START, N_FEATURES))
)
N_INPUT_FEATURES = len(INPUT_FEATURE_INDICES)
MODEL_VERSION = 2


def card_schema():
    catalog = json.loads(taverngym.catalog_json())
    allowed = np.zeros(taverngym.N_CARDS, dtype=bool)
    kinds = {
        "minion",
        "spell",
        "tavern_spell",
        "hero",
        "hero_power",
        "trinket_lesser",
        "trinket_greater",
        "quest",
        "quest_reward",
    }
    for card in catalog:
        allowed[card["cid"] + 1] = card["kind"] in kinds
    fingerprint = hashlib.sha256(
        json.dumps(
            [(c["cid"], c["id"], c["kind"]) for c in catalog], separators=(",", ":")
        ).encode()
    ).hexdigest()
    return allowed, fingerprint


_GELU_C = 0.7978845608028654  # sqrt(2/pi)
_GELU_K = 0.044715


def gelu(x):
    inner = _GELU_C * (x + _GELU_K * x * x * x)
    return 0.5 * x * (1.0 + np.tanh(inner))


def gelu_grad(x):
    inner = _GELU_C * (x + _GELU_K * x * x * x)
    t = np.tanh(inner)
    d_inner = _GELU_C * (1.0 + 3.0 * _GELU_K * x * x)
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * d_inner


def activate(x, name):
    if name == "gelu":
        return gelu(x)
    if name == "silu":
        return x * (0.5 * (1.0 + np.tanh(x * 0.5)))
    if name == "relu":
        return np.maximum(x, 0)
    raise ValueError(f"unknown activation: {name}")


def activation_grad(x, name):
    if name == "gelu":
        return gelu_grad(x)
    if name == "silu":
        sigmoid = 0.5 * (1.0 + np.tanh(x * 0.5))
        return sigmoid + x * sigmoid * (1 - sigmoid)
    if name == "relu":
        return (x > 0).astype(np.float32)
    raise ValueError(f"unknown activation: {name}")


def layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    xhat = (x - mean) * inv
    return xhat * gamma + beta, (xhat, inv, gamma)


def layer_norm_backward(grad, cache):
    xhat, inv, gamma = cache
    dgamma = (grad * xhat).sum(axis=0)
    dbeta = grad.sum(axis=0)
    dxhat = grad * gamma
    dx = inv * (
        dxhat
        - dxhat.mean(axis=-1, keepdims=True)
        - xhat * (dxhat * xhat).mean(axis=-1, keepdims=True)
    )
    return dx, dgamma, dbeta


class PolicyValueNet:
    hidden = 192
    blocks = 3

    def __init__(
        self,
        seed: int = 0,
        embedding_dim: int = 8,
        activation="gelu",
        residual_scale=1.0,
        head_norm=False,
    ):
        if activation not in ("gelu", "silu", "relu") or not (0 < residual_scale <= 1):
            raise ValueError("unsupported activation or residual scale")
        self.activation = activation
        self.residual_scale = residual_scale
        self.head_norm = bool(head_norm)
        rng = np.random.default_rng(seed)
        h = self.hidden
        if N_FEATURES != 3237 or N_CARD_SLOTS != 124:
            raise ValueError("observation schema changed; rebuild taverngym and update the network")
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")
        self.embedding_dim = embedding_dim
        self.allowed_cards, self.card_fingerprint = card_schema()
        # Retain power and quest/reward slots; exclude Deity and secret identities.
        self.card_slots = np.ones(N_CARD_SLOTS, dtype=bool)
        self.card_slots[3] = False
        quest_start = 1 + _LAYOUT["N_POWERS"] + 2
        secret_start = quest_start + 3 * _LAYOUT["N_QUESTS"]
        self.card_slots[secret_start : secret_start + _LAYOUT["N_SECRETS"]] = False
        self.card_slots[-N_OPP_BOARD_SLOTS:] = False
        self.p = {
            "E": (
                rng.standard_normal((taverngym.N_CARDS, embedding_dim), dtype=np.float32)
                * np.float32(0.1)
            ),
            "W0": self._he(rng, N_INPUT_FEATURES + int(self.card_slots.sum()) * embedding_dim, h),
            "b0": np.zeros(h, np.float32),
            "g0": np.ones(h, np.float32),
            "k0": np.zeros(h, np.float32),
        }
        for i in range(self.blocks):
            self.p[f"W{i + 1}"] = (self._he(rng, h, h) * np.float32(0.6)).astype(np.float32)
            self.p[f"b{i + 1}"] = np.zeros(h, np.float32)
            self.p[f"g{i + 1}"] = np.ones(h, np.float32)
            self.p[f"k{i + 1}"] = np.zeros(h, np.float32)
        self.p["Wp"] = (self._he(rng, h, N_ACTIONS) * np.float32(0.1)).astype(np.float32)
        self.p["bp"] = np.zeros(N_ACTIONS, np.float32)
        self.p["Wv"] = (self._he(rng, h, 1) * np.float32(0.1)).astype(np.float32)
        self.p["bv"] = np.zeros(1, np.float32)
        self.p["E"][0] = 0.0
        self.cache = None

    @staticmethod
    def _he(rng, fan_in, fan_out, scale: float = 1.0):
        return (
            rng.standard_normal((fan_in, fan_out), dtype=np.float32)
            * np.float32(scale * np.sqrt(2.0 / fan_in))
        ).astype(np.float32)

    def n_params(self) -> int:
        return int(sum(v.size for v in self.p.values()))

    # ---- forward ---------------------------------------------------------

    def forward(self, x, cards):
        p = self.p
        if x.ndim == 1:
            x = x.reshape(1, -1)
        cards = np.asarray(cards)
        if cards.ndim == 1:
            cards = cards[None, :]
        if x.shape[1:] != (N_FEATURES,) or cards.shape != (len(x), N_CARD_SLOTS):
            raise ValueError("expected matching feature and card batches")
        if (
            not np.issubdtype(cards.dtype, np.integer)
            or np.any(cards < 0)
            or np.any(cards >= len(p["E"]))
        ):
            raise ValueError("card indices must be integers from the current card table")
        cards = cards[:, self.card_slots]
        cards = np.where(self.allowed_cards[cards], cards, 0)
        embedded = p["E"][cards] * (cards != 0)[..., None]
        x = np.concatenate((x[:, INPUT_FEATURE_INDICES], embedded.reshape(len(x), -1)), axis=1)
        z0 = x @ p["W0"] + p["b0"]
        n0, ln0 = layer_norm(z0, p["g0"], p["k0"])
        h = activate(n0, self.activation)
        layers = [(x, z0, n0, ln0)]
        for i in range(self.blocks):
            h_in = h
            z = h_in @ p[f"W{i + 1}"] + p[f"b{i + 1}"]
            n, ln = layer_norm(z, p[f"g{i + 1}"], p[f"k{i + 1}"])
            h = h_in + self.residual_scale * activate(n, self.activation)
            layers.append((h_in, z, n, ln))
        head_ln = None
        if self.head_norm:
            # Fixed affine terms: normalize the heads' input without adding parameters.
            h, head_ln = layer_norm(h, np.float32(1), np.float32(0))
        logits = h @ p["Wp"] + p["bp"]
        value = np.tanh(h @ p["Wv"] + p["bv"])[:, 0]
        self.cache = (layers, h, cards, head_ln)
        return logits, value

    def predict(self, x, cards, mask=None):
        logits, value = self.forward(x, cards)
        return masked_softmax(logits[0], mask), float(value[0])

    # ---- backward --------------------------------------------------------

    def backward(self, dlogits, dvalue) -> dict:
        p = self.p
        layers, h_final, cards, head_ln = self.cache
        g = {
            "Wp": h_final.T @ dlogits,
            "bp": dlogits.sum(axis=0),
            "Wv": h_final.T @ dvalue,
            "bv": dvalue.sum(axis=0),
        }
        dh = dlogits @ p["Wp"].T + dvalue @ p["Wv"].T
        if head_ln is not None:
            dh, _, _ = layer_norm_backward(dh, head_ln)

        for i in range(self.blocks, 0, -1):
            h_in, _z, n, ln = layers[i]
            dn = dh * self.residual_scale * activation_grad(n, self.activation)
            dz, dg, dk = layer_norm_backward(dn, ln)
            g[f"g{i}"] = dg
            g[f"k{i}"] = dk
            g[f"W{i}"] = h_in.T @ dz
            g[f"b{i}"] = dz.sum(axis=0)
            dh = dh + dz @ p[f"W{i}"].T  # residual: skip + block

        x0, _z0, n0, ln0 = layers[0]
        dn0 = dh * activation_grad(n0, self.activation)
        dz0, dg0, dk0 = layer_norm_backward(dn0, ln0)
        g["g0"], g["k0"] = dg0, dk0
        g["W0"] = x0.T @ dz0
        g["b0"] = dz0.sum(axis=0)
        de = (dz0 @ p["W0"][N_INPUT_FEATURES:].T).reshape(
            len(cards), int(self.card_slots.sum()), self.embedding_dim
        )
        g["E"] = np.zeros_like(p["E"])
        # Repeated cards share a row: scatter-add, not assignment.
        np.add.at(g["E"], cards, de * (cards != 0)[..., None])
        g["E"][0] = 0.0
        return g

    # ---- persistence -----------------------------------------------------

    def save(self, path):
        np.savez(
            path,
            **self.p,
            __hidden=self.hidden,
            __blocks=self.blocks,
            __version=MODEL_VERSION,
            __features=N_FEATURES,
            __input_features=N_INPUT_FEATURES,
            __embedding_dim=self.embedding_dim,
            __card_fingerprint=self.card_fingerprint,
            __activation=self.activation,
            __residual_scale=self.residual_scale,
            __head_norm=self.head_norm,
        )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            if "__version" not in data or int(data["__version"]) != MODEL_VERSION:
                raise ValueError("unsupported checkpoint format; expected model format v2")
            required = {
                "__hidden",
                "__blocks",
                "__features",
                "__input_features",
                "__embedding_dim",
                "__card_fingerprint",
                "__activation",
                "__residual_scale",
                "__head_norm",
            }
            if not required.issubset(data.files):
                raise ValueError("incomplete model format v2 metadata")
            net = cls(
                seed=0,
                embedding_dim=int(data["__embedding_dim"]),
                activation=str(data["__activation"]),
                residual_scale=float(data["__residual_scale"]),
                head_norm=bool(data["__head_norm"]),
            )
            if (
                int(data["__features"]) != N_FEATURES
                or int(data["__input_features"]) != N_INPUT_FEATURES
                or str(data["__card_fingerprint"]) != net.card_fingerprint
                or int(data["__hidden"]) != net.hidden
                or int(data["__blocks"]) != net.blocks
            ):
                raise ValueError(
                    "checkpoint does not match current observation/card/network schema"
                )
            if not net.p.keys() <= set(data.files):
                raise ValueError("incomplete model format v2 parameters")
            for k, expected in net.p.items():
                if data[k].shape != expected.shape:
                    raise ValueError(f"checkpoint parameter shape mismatch: {k}")
            net.p = {k: data[k].copy() for k in net.p}
        net.p["E"][0] = 0.0
        return net


def masked_softmax(logits, mask=None) -> np.ndarray:
    """Softmax over the legal actions only.

    With an empty mask this returns zeros, not a uniform distribution: a seat that has
    no legal action (hero already chosen, say) must not look like it can play anything.
    """
    z = logits.astype(np.float64)
    if mask is not None:
        z = np.where(mask, z, -np.inf)
    top = z.max()
    if not np.isfinite(top):
        return np.zeros(N_ACTIONS, np.float32)
    e = np.exp(z - top)
    total = e.sum()
    return (e / total).astype(np.float32) if total > 0 else np.zeros(N_ACTIONS, np.float32)
