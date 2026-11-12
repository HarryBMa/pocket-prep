"""Needle 2 as a TensorFlow graph that holds the PyTorch weights (BatchNorm folded), for TFLite and Vela.

    fn, shapes = build(model)        # model: needle2.Needle2 in eval mode
    out = fn(*bands)                 # bands: one NHWC tensor per band, (1, rows, frames, channels * stack)

The graph is what the NPU runs: the band tensors in (the Cortex-M85 fills them, see needle2.to_bands), the
filter taps on the band grid out, (1, rows, frames, K). Only operators Vela 5.2.0 puts on the Ethos-U55:

- convolutions with explicit zero padding (PAD + CONV_2D, like PyTorch's padding=1), stride (2, 2) or (2, 1);
- TDF: TRANSPOSE so that frequency becomes the channel axis, two 1x1 convolutions, TRANSPOSE back, ADD;
- GRU: written out step by step. The input half of the gates is one 1x1 convolution for all steps; each step
  then needs a 1x1 convolution on the state, LOGISTIC, TANH, MUL, ADD, SUB. The backward direction is the same
  wiring walked from the other end (no REVERSE operator, which Vela leaves on the CPU);
- upsampling: 2x2 is RESIZE_NEAREST_NEIGHBOR; frequency-only is CONCATENATION + RESHAPE (the U55 cannot
  resize by 2x1): reshape(concat([x, x], time axis)) repeats every row.
"""
import numpy as np
import tensorflow as tf
import torch

from needle2 import TDF, ConvBN, Needle2


def _np(t):
    return t.detach().cpu().numpy().astype(np.float32)


def _scale_shift(bn):
    """BatchNorm at inference is y = scale * x + shift per channel."""
    scale = bn.weight.detach() / torch.sqrt(bn.running_var + bn.eps)
    return scale, bn.bias.detach() - bn.running_mean * scale


def fold_conv(conv, bn=None):
    """Conv2d (+ the BatchNorm after it) -> kernel HWIO, bias."""
    w = conv.weight.detach()
    b = conv.bias.detach() if conv.bias is not None else torch.zeros(w.shape[0])
    if bn is not None:
        scale, shift = _scale_shift(bn)
        w, b = w * scale[:, None, None, None], b * scale + shift
    return _np(w.permute(2, 3, 1, 0)), _np(b)


def conv2d(x, w, b, stride=(1, 1), relu=True):
    if w.shape[0] == 3:
        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
    y = tf.nn.bias_add(tf.nn.conv2d(x, tf.constant(w), [1, stride[0], stride[1], 1], "VALID"), tf.constant(b))
    return tf.nn.relu(y) if relu else y


def fc(x, weight, bias, relu):
    """A fully connected layer on the channel axis = a 1x1 convolution. weight (out, in) as in PyTorch."""
    return conv2d(x, _np(weight).T[None, None], _np(bias), relu=relu)


def tdf(x, m: TDF):
    s1, o1 = _scale_shift(m.bn1)
    s2, o2 = _scale_shift(m.bn2)
    y = tf.transpose(x, [0, 3, 2, 1])                                   # (1, C, T, F): frequency is now the channel axis
    y = fc(y, m.fc1.weight.detach() * s1[:, None], o1, relu=True)
    y = fc(y, m.fc2.weight.detach() * s2[:, None], o2, relu=False)
    return x + tf.transpose(y, [0, 3, 2, 1])


def bigru(x, gru, bn, axis):
    """Bidirectional GRU along `axis` of x (1, F, T, C): axis 2 = along time, one sequence per row; axis 1 =
    along frequency, one sequence per frame. `bn` is the BatchNorm in front of the GRU (folded into the input
    weights). Returns (1, F, T, 2 * hidden), forward states then backward states, like PyTorch.

    PyTorch's GRU, per step:   r = sigmoid(Wir x + bir + Whr h + bhr)        z = sigmoid(Wiz x + biz + Whz h + bhz)
                               n = tanh(Win x + bin + r * (Whn h + bhn))      h = (1 - z) * n + z * h
    The x half (gx) is computed for all steps at once; bhr and bhz move into its bias, so the first step
    (h = 0) needs no state at all."""
    hid = gru.hidden_size
    scale, shift = _scale_shift(bn)
    steps_n = x.shape[axis]
    out = []
    for suffix in ("", "_reverse"):
        w_ih, w_hh = getattr(gru, "weight_ih_l0" + suffix).detach(), getattr(gru, "weight_hh_l0" + suffix).detach()
        b_ih, b_hh = getattr(gru, "bias_ih_l0" + suffix).detach(), getattr(gru, "bias_hh_l0" + suffix).detach()
        bx = b_ih + w_ih @ shift
        bx[: 2 * hid] += b_hh[: 2 * hid]
        gx = fc(x, w_ih * scale[None, :], bx, relu=False)                # (1, F, T, 3 hid) for every step at once
        bh = torch.cat([torch.zeros(2 * hid), b_hh[2 * hid:]])
        wh, bh = _np(w_hh).T[None, None], _np(bh)
        b_hn = tf.constant(_np(b_hh[2 * hid:]).reshape(1, 1, 1, hid))
        steps = tf.split(gx, steps_n, axis=axis)
        if suffix:
            steps = steps[::-1]
        h, states = None, []
        for g in steps:
            g_rz, g_n = tf.split(g, [2 * hid, hid], axis=3)
            if h is None:
                r, z = tf.split(tf.sigmoid(g_rz), 2, axis=3)
                n = tf.tanh(g_n + r * b_hn)
                h = n - z * n
            else:
                gh_rz, gh_n = tf.split(conv2d(h, wh, bh, relu=False), [2 * hid, hid], axis=3)
                r, z = tf.split(tf.sigmoid(g_rz + gh_rz), 2, axis=3)
                n = tf.tanh(g_n + r * gh_n)
                h = n + z * (h - n)
            states.append(h)
        out.append(tf.concat(states[::-1] if suffix else states, axis=axis))
    return tf.concat(out, axis=3)


def dual_path(x, m):
    x = x + conv2d(bigru(x, m.gru_t, m.bn_t, axis=2), *fold_conv(m.out_t, m.bno_t), relu=False)
    return x + conv2d(bigru(x, m.gru_f, m.bn_f, axis=1), *fold_conv(m.out_f, m.bno_f), relu=False)


def upsample(x, stride):
    h, w, c = x.shape[1:]
    if tuple(stride) == (2, 2):
        return tf.image.resize(x, (2 * h, 2 * w), method="nearest")
    assert tuple(stride) == (2, 1), stride
    return tf.reshape(tf.concat([x, x], axis=2), [1, 2 * h, w, c])    # rows 2i and 2i+1 are both row i


def block(x, blk):
    for layer in blk:
        if isinstance(layer, ConvBN):
            x = conv2d(x, *fold_conv(layer[0], layer[1]), stride=layer[0].stride)
        else:
            x = tdf(x, layer)
    return x


def build(model: Needle2):
    """-> (fn, shapes): fn(*bands) is the network as TensorFlow operations; shapes are the band inputs'
    (rows, frames, channels), without the batch dimension of 1."""
    assert not model.training, "call model.eval() first: the BatchNorms are folded with their running statistics"
    cfg = model.cfg
    shapes = [(b // s, cfg.frames, cfg.in_channels * s) for b, s in cfg.bands]

    def fn(*bands):
        x = tf.concat([conv2d(band, *fold_conv(stem[0], stem[1])) for stem, band in zip(model.stems, bands)], axis=1)
        skips = []
        for blk in model.enc:
            x = block(x, blk)
            skips.append(x)
        skips.pop()
        for blk in model.rnn:
            x = dual_path(x, blk)
        for i, blk in zip(reversed(range(len(model.dec))), model.dec):
            x = upsample(x, model.strides[i])
            x = block(x + skips.pop() if i == 0 else tf.concat([x, skips.pop()], axis=3), blk)
        x = conv2d(x, *fold_conv(model.head), relu=False)
        return tf.tanh(x) if cfg.taps else tf.sigmoid(x)

    return fn, shapes


def bands_nhwc(x, cfg):
    """Network input (C, bins, T) or (B, C, bins, T) as PyTorch has it -> the band tensors the TFLite model
    takes: a list of float32 arrays (B, rows, T, channels * stack)."""
    from needle2 import to_bands
    x = torch.as_tensor(x)
    if x.ndim == 3:
        x = x[None]
    return [b.permute(0, 2, 3, 1).contiguous().numpy().astype(np.float32) for b in to_bands(x, cfg)]
