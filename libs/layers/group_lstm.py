#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import theano
from theano import tensor as T

from ..constants import fX, profile
from .basic import _slice, _attention, dropout_layer
from ..utility.utils import _p, normal_weight, orthogonal_weight

__author__ = 'fyabc'


def param_init_group_lstm(O, params, prefix='group_lstm', nin=None, dim=None, **kwargs):
    if nin is None:
        nin = O['dim_proj']
    if dim is None:
        dim = O['dim_proj']

    layer_id = kwargs.pop('layer_id', 0)
    context_dim = kwargs.pop('context_dim', None)
    group_num = kwargs.pop('group_num', O.get('group_num', 2))

    params[_p(prefix, 'W', layer_id)] = np.stack([np.concatenate([
        normal_weight(nin/group_num, dim/group_num),
        normal_weight(nin/group_num, dim/group_num),
        normal_weight(nin/group_num, dim/group_num),
        normal_weight(nin/group_num, dim/group_num),
    ], axis=1) for _ in range(group_num)], axis=0)

    params[_p(prefix, 'U', layer_id)] = np.stack([np.concatenate([
            orthogonal_weight(dim/group_num),
            orthogonal_weight(dim/group_num),
            orthogonal_weight(dim/group_num),
            orthogonal_weight(dim/group_num),
    ], axis=1) for _ in range(group_num)], axis=0)

    params[_p(prefix, 'b', layer_id)] = np.stack([
        np.zeros((4 * dim/group_num,), dtype=fX) for _ in range(group_num)], axis=0)

    if context_dim is not None:
        params[_p(prefix, 'Wc', layer_id)] = np.stack([np.concatenate([
                normal_weight(context_dim, dim/group_num),
                normal_weight(context_dim, dim/group_num),
                normal_weight(context_dim, dim/group_num),
                normal_weight(context_dim, dim/group_num),
        ], axis=1) for _ in range(group_num)], axis=0)
    return params


def group_lstm_layer(P, state_below, O, prefix='group_lstm', mask=None, **kwargs):
    """LSTM layer

    inputs and outputs are same as GRU layer.

    outputs[1]: hidden memory
    """

    layer_id = kwargs.pop('layer_id', 0)
    dropout_params = kwargs.pop('dropout_params', None)
    context = kwargs.pop('context', None)
    one_step = kwargs.pop('one_step', False)
    init_state = kwargs.pop('init_state', None)
    init_memory = kwargs.pop('init_memory', None)
    group_num = kwargs.pop('group_num', O.get('group_num', 2))
    shuffle = kwargs.pop('shuffle_group', O.get('shuffle_group', True))
    print('group_num = {}'.format(group_num))
    print('shuffle = {}'.format(shuffle))

    kw_ret = {}

    n_steps = state_below.shape[0] if state_below.ndim == 3 else 1
    n_samples = state_below.shape[1] if state_below.ndim == 3 else state_below.shape[0]
    # dim = P[_p(prefix, 'U', layer_id)].shape[1] * group_num #// 4

    mask = T.alloc(1., n_steps, 1) if mask is None else mask

    # x1 = P[_p(prefix, 'W', layer_id)].shape[0]
    x2 = P[_p(prefix, 'W', layer_id)].shape[1]
    h_per_group = P[_p(prefix, 'U', layer_id)].shape[1]
    h4_per_group = P[_p(prefix, 'U', layer_id)].shape[2]
    dim = h_per_group * group_num
    # x3 = P[_p(prefix, 'W', layer_id)].shape[2]

    state_below = T.reshape(state_below, [n_steps, n_samples, group_num, x2])
    state_below = T.transpose(state_below, [2, 0, 1, 3])
    state_below = T.batched_dot(state_below, P[_p(prefix, 'W', layer_id)])
    # [n_steps, n_samples, group_num, 4*dim_per_group]
    state_below = T.transpose(state_below, [1, 2, 0, 3]) + P[_p(prefix, 'b', layer_id)]

    # prepare scan arguments
    init_states = [T.alloc(0., n_samples, group_num, h_per_group) 
                     if init_state is None 
                     else T.reshape(init_state, [n_samples, group_num, h_per_group]),
                   T.alloc(0., n_samples, group_num, h_per_group) 
                     if init_memory is None 
                     else T.reshape(init_memory, [n_samples, group_num, h_per_group]), ]

    def _group_lstm_step(mask_, x_, h_, c_, U):
        preact = T.transpose(h_, [1, 0, 2])
        preact = T.batched_dot(preact, U)
        preact = T.transpose(preact, [1, 0, 2])
        preact = preact + x_

        i = T.nnet.sigmoid(_slice(preact, 0, h_per_group))
        f = T.nnet.sigmoid(_slice(preact, 1, h_per_group))
        o = T.nnet.sigmoid(_slice(preact, 2, h_per_group))
        c = T.tanh(_slice(preact, 3, h_per_group))
       
        c = f * c_ + i * c
        h = o * T.tanh(c)
        c = mask_[:, None, None] * c + (1. - mask_)[:, None, None] * c_
        h = mask_[:, None, None] * h + (1. - mask_)[:, None, None] * h_
        return h, c

    def _group_lstm_step_attention(mask_, x_, context, h_, c_, U, Wc):
        preact = T.transpose(h_, [1, 0, 2])
        preact = T.batched_dot(preact, U)
        preact = T.transpose(preact, [1, 0, 2])
        # [n_samples, group_num, 4*dim_per_group]
        preact = preact + x_ + T.tensordot(context, Wc, [[1], [1]])

        i = T.nnet.sigmoid(_slice(preact, 0, h_per_group))
        f = T.nnet.sigmoid(_slice(preact, 1, h_per_group))
        o = T.nnet.sigmoid(_slice(preact, 2, h_per_group))
        c = T.tanh(_slice(preact, 3, h_per_group))
       
        c = f * c_ + i * c
        h = o * T.tanh(c)
        c = mask_[:, None, None] * c + (1. - mask_)[:, None, None] * c_
        h = mask_[:, None, None] * h + (1. - mask_)[:, None, None] * h_
        return h, c

    if context is None:
        seqs = [mask, state_below]
        shared_vars = [P[_p(prefix, 'U', layer_id)]]
        _step = _group_lstm_step
    else:
        seqs = [mask, state_below, context]
        shared_vars = [P[_p(prefix, 'U', layer_id)], 
                       P[_p(prefix, 'Wc', layer_id)]]
        _step = _group_lstm_step_attention

    if one_step:
        outputs = _step(*(seqs + init_states + shared_vars))
    else:
        outputs, _ = theano.scan(
            _step,
            sequences=seqs,
            outputs_info=init_states,
            non_sequences=shared_vars,
            name=_p(prefix, '_layers', layer_id),
            n_steps=n_steps,
            profile=profile,
            strict=True,
        )

    y_output = outputs[0]
    if shuffle:
        tmp = T.reshape(outputs[0], [n_steps, n_samples, group_num, -1])
        tmp = T.transpose(tmp, [0, 1, 3, 2])
        y_output = T.squeeze(T.reshape(tmp, [n_steps, n_samples, -1]))
    y_output = T.squeeze(T.reshape(y_output, [n_steps, n_samples, -1]))
    memory_output = T.squeeze(T.reshape(outputs[1], [n_steps, n_samples, -1]))

    kw_ret['hidden_without_dropout'] = y_output  # outputs[0]
    kw_ret['memory_output'] = memory_output  # outputs[1]

    outputs = [y_output, memory_output, kw_ret]

    if dropout_params:
        outputs[0] = dropout_layer(outputs[0], *dropout_params)

    return outputs


def param_init_group_lstm_cond(O, params, prefix='lstm_cond', nin=None, dim=None, dimctx=None, nin_nonlin=None,
                         dim_nonlin=None, **kwargs):
    if nin is None:
        nin = O['dim']
    if dim is None:
        dim = O['dim']
    if dimctx is None:
        dimctx = O['dim']
    if nin_nonlin is None:
        nin_nonlin = nin
    if dim_nonlin is None:
        dim_nonlin = dim
    layer_id = kwargs.pop('layer_id', 0)
    multi = 'multi' in O.get('unit', 'lstm_cond')
    unit_size = kwargs.pop('unit_size', O.get('cond_unit_size', 2))

    if not multi:
        params[_p(prefix, 'W', layer_id)] = np.concatenate([
            normal_weight(nin, dim),
            normal_weight(nin, dim),
            normal_weight(nin, dim),
            normal_weight(nin_nonlin, dim),
        ], axis=1)
        params[_p(prefix, 'b', layer_id)] = np.zeros((4 * dim,), dtype=fX)
        params[_p(prefix, 'U', layer_id)] = np.concatenate([
            orthogonal_weight(dim_nonlin),
            orthogonal_weight(dim_nonlin),
            orthogonal_weight(dim_nonlin),
            orthogonal_weight(dim_nonlin),
        ], axis=1)

        params[_p(prefix, 'U_nl', layer_id)] = np.concatenate([
            orthogonal_weight(dim_nonlin),
            orthogonal_weight(dim_nonlin),
            orthogonal_weight(dim_nonlin),
            orthogonal_weight(dim_nonlin),
        ], axis=1)
        params[_p(prefix, 'b_nl', layer_id)] = np.zeros((4 * dim_nonlin,), dtype=fX)

        # context to LSTM
        params[_p(prefix, 'Wc', layer_id)] = normal_weight(dimctx, dim * 4)
    else:
        params[_p(prefix, 'W', layer_id)] = np.stack([np.concatenate([
            normal_weight(nin, dim),
            normal_weight(nin, dim),
            normal_weight(nin, dim),
            normal_weight(nin_nonlin, dim),
        ], axis=1) for _ in xrange(unit_size)], axis=0)
        params[_p(prefix, 'b', layer_id)] = np.zeros((unit_size, 4 * dim,), dtype=fX)
        params[_p(prefix, 'U', layer_id)] = np.stack([np.concatenate([
            orthogonal_weight(dim_nonlin),
            orthogonal_weight(dim_nonlin),
            orthogonal_weight(dim_nonlin),
            orthogonal_weight(dim_nonlin),
        ], axis=1) for _ in xrange(unit_size)], axis=0)

        params[_p(prefix, 'U_nl', layer_id)] = np.stack([np.concatenate([
            orthogonal_weight(dim_nonlin),
            orthogonal_weight(dim_nonlin),
            orthogonal_weight(dim_nonlin),
            orthogonal_weight(dim_nonlin),
        ], axis=1) for _ in xrange(unit_size)], axis=0)
        params[_p(prefix, 'b_nl', layer_id)] = np.zeros((unit_size, 4 * dim_nonlin,), dtype=fX)

        # context to LSTM
        params[_p(prefix, 'Wc', layer_id)] = np.stack([
            normal_weight(dimctx, dim * 4) for _ in xrange(unit_size)], axis=0)

    # attention: combined -> hidden
    params[_p(prefix, 'W_comb_att', layer_id)] = normal_weight(dim, dimctx)

    # attention: context -> hidden
    params[_p(prefix, 'Wc_att', layer_id)] = normal_weight(dimctx)

    # attention: hidden bias
    params[_p(prefix, 'b_att', layer_id)] = np.zeros((dimctx,), dtype=fX)

    # attention:
    params[_p(prefix, 'U_att', layer_id)] = normal_weight(dimctx, 1)
    params[_p(prefix, 'c_tt', layer_id)] = np.zeros((1,), dtype=fX)

    return params


def _lstm_step_kernel(preact, mask_, h_, c_, _dim):
    i = T.nnet.sigmoid(_slice(preact, 0, _dim))
    f = T.nnet.sigmoid(_slice(preact, 1, _dim))
    o = T.nnet.sigmoid(_slice(preact, 2, _dim))
    c = T.tanh(_slice(preact, 3, _dim))

    c = f * c_ + i * c
    c = mask_[:, None] * c + (1. - mask_)[:, None] * c_

    h = o * T.tanh(c)
    h = mask_[:, None] * h + (1. - mask_)[:, None] * h_

    return i, f, o, c, h


def _lstm_step_slice(
        mask_, x_,
        h_, c_,
        U):
    _dim = U.shape[1] // 4
    preact = T.dot(h_, U) + x_

    i, f, o, c, h = _lstm_step_kernel(preact, mask_, h_, c_, _dim)
    return h, c


def _lstm_step_slice_gates(
        mask_, x_,
        h_, c_, i_, f_, o_,
        U):
    _dim = U.shape[1] // 4
    preact = T.dot(h_, U) + x_

    i, f, o, c, h = _lstm_step_kernel(preact, mask_, h_, c_, _dim)
    return h, c, i, f, o


def _lstm_step_slice_attention(
        mask_, x_, context,
        h_, c_,
        U, Wc):
    _dim = U.shape[1] // 4
    preact = T.dot(h_, U) + x_ + T.dot(context, Wc)

    i, f, o, c, h = _lstm_step_kernel(preact, mask_, h_, c_, _dim)
    return h, c


def _lstm_step_slice_attention_gates(
        mask_, x_, context,
        h_, c_, i_, f_, o_,
        U, Wc):
    _dim = U.shape[1] // 4
    preact = T.dot(h_, U) + x_ + T.dot(context, Wc)

    i, f, o, c, h = _lstm_step_kernel(preact, mask_, h_, c_, _dim)
    return h, c, i, f, o


def group_lstm_cond_layer(P, state_below, O, prefix='lstm', mask=None, context=None, one_step=False, init_memory=None,
                    init_state=None, context_mask=None, **kwargs):
    """Conditional LSTM layer with attention

    inputs and outputs are same as GRU cond layer.
    """

    layer_id = kwargs.pop('layer_id', 0)
    multi = 'multi' in O.get('unit', 'lstm_cond')
    unit_size = kwargs.pop('unit_size', O.get('cond_unit_size', 2))
    # FIXME: multi-gru/lstm do NOT support get_gates now
    get_gates = kwargs.pop('get_gates', False)

    kw_ret = {}

    assert context, 'Context must be provided'
    assert context.ndim == 3, 'Context must be 3-d: #annotation * #sample * dim'
    if one_step:
        assert init_state, 'previous state must be provided'

    # Dimensions
    n_steps = state_below.shape[0] if state_below.ndim == 3 else 1
    n_samples = state_below.shape[1] if state_below.ndim == 3 else state_below.shape[0]
    if multi:
        dim = P[_p(prefix, 'Wc', layer_id)][0].shape[1] // 4
    else:
        dim = P[_p(prefix, 'Wc', layer_id)].shape[1] // 4
    dropout_params = kwargs.pop('dropout_params', None)

    # Mask
    if mask is None:
        mask = T.alloc(1., n_steps, 1)

    # Initial/previous state
    if init_state is None:
        init_state = T.alloc(0., n_samples, dim)

    projected_context = T.dot(context, P[_p(prefix, 'Wc_att', layer_id)]) + P[_p(prefix, 'b_att', layer_id)]

    # Projected x
    if multi:
        state_below = T.concatenate([
            T.dot(state_below, P[_p(prefix, 'W', layer_id)][j]) + P[_p(prefix, 'b', layer_id)][j]
            for j in range(unit_size)
        ], axis=-1)
    else:
        state_below = T.dot(state_below, P[_p(prefix, 'W', layer_id)]) + P[_p(prefix, 'b', layer_id)]

    def _one_step_attention_slice(mask_, h1, c1, ctx_, Wc, U_nl, b_nl):
        preact2 = T.dot(h1, U_nl) + b_nl + T.dot(ctx_, Wc)

        i2 = T.nnet.sigmoid(_slice(preact2, 0, dim))
        f2 = T.nnet.sigmoid(_slice(preact2, 1, dim))
        o2 = T.nnet.sigmoid(_slice(preact2, 2, dim))
        c2 = T.tanh(_slice(preact2, 3, dim))

        c2 = f2 * c1 + i2 * c2
        c2 = mask_[:, None] * c2 + (1. - mask_)[:, None] * c1

        h2 = o2 * T.tanh(c2)
        h2 = mask_[:, None] * h2 + (1. - mask_)[:, None] * h1

        if get_gates:
            return h2, c2, i2, f2, o2

        return h2, c2

    def _step_slice(mask_, x_,
                    h_, c_, ctx_, alpha_,
                    projected_context_, context_,
                    U, Wc, W_comb_att, U_att, c_tt, U_nl, b_nl):
        # LSTM 1
        h1, c1 = _lstm_step_slice(mask_, x_, h_, c_, U)

        # Attention
        ctx_, alpha = _attention(h1, projected_context_, context_, W_comb_att, U_att, c_tt, context_mask=context_mask)

        # LSTM 2 (with attention)
        h2, c2 = _one_step_attention_slice(mask_, h1, c1, ctx_, Wc, U_nl, b_nl)

        return h2, c2, ctx_, alpha.T

    # todo: implement it
    def _step_slice_gates(mask_, x_,
                          h_, c_, ctx_, alpha_, i1_, f1_, o1_, i2_, f2_, o2_,
                          projected_context_, context_,
                          U, Wc, W_comb_att, U_att, c_tt, U_nl, b_nl):
        # LSTM 1
        h1, c1, i1, f1, o1 = _lstm_step_slice_gates(mask_, x_, h_, c_, i1_, f1_, o1_, U)

        # Attention
        ctx_, alpha = _attention(h1, projected_context_, context_, W_comb_att, U_att, c_tt, context_mask=context_mask)

        # LSTM 2 (with attention)
        h2, c2, i2, f2, o2 = _one_step_attention_slice(mask_, h1, c1, ctx_, Wc, U_nl, b_nl)

        return h2, c2, ctx_, alpha.T, i1, f1, o1, i2, f2, o2

    def _multi_step_slice(mask_, x_,
                          h_, c_, ctx_, alpha_,
                          projected_context_, context_,
                          U, Wc, W_comb_att, U_att, c_tt, U_nl, b_nl):
        # LSTM 1
        h_tmp = h_
        c_tmp = c_
        for j in range(unit_size):
            x = _slice(x_, j, 4 * dim)
            h1, c1 = _lstm_step_slice(mask_, x, h_tmp, c_tmp, U[j])
            h_tmp = h1
            c_tmp = c1

        # Attention
        ctx_, alpha = _attention(h1, projected_context_, context_, W_comb_att, U_att, c_tt, context_mask=context_mask)

        # LSTM 2 (with attention)
        h_tmp_att = h1
        c_tmp_att = c1
        for j in range(unit_size):
            h2, c2 = _one_step_attention_slice(mask_, h_tmp_att, c_tmp_att, ctx_, Wc[j], U_nl[j], b_nl[j])
            h_tmp_att = h2
            c_tmp_att = c2

        return h2, c2, ctx_, alpha.T

    # Prepare scan arguments
    seqs = [mask, state_below]
    if multi:
        _step = _multi_step_slice
    else:
        if get_gates:
            _step = _step_slice_gates
        else:
            _step = _step_slice
    init_states = [
        init_state,
        T.alloc(0., n_samples, dim) if init_memory is None else init_memory,
        T.alloc(0., n_samples, context.shape[2]),
        T.alloc(0., n_samples, context.shape[0]),
    ]
    if get_gates:
        init_states.extend([T.alloc(0., n_samples, dim) for _ in range(6)])

    shared_vars = [
        P[_p(prefix, 'U', layer_id)],
        P[_p(prefix, 'Wc', layer_id)],
        P[_p(prefix, 'W_comb_att', layer_id)],
        P[_p(prefix, 'U_att', layer_id)],
        P[_p(prefix, 'c_tt', layer_id)],
        P[_p(prefix, 'U_nl', layer_id)],
        P[_p(prefix, 'b_nl', layer_id)],
    ]

    if one_step:
        result = _step(*(seqs + init_states + [projected_context, context] + shared_vars))
    else:
        result, _ = theano.scan(
            _step,
            sequences=seqs,
            outputs_info=init_states,
            non_sequences=[projected_context, context] + shared_vars,
            name=_p(prefix, '_layers'),
            n_steps=n_steps,
            profile=profile,
            strict=True,
        )

    kw_ret['hidden_without_dropout'] = result[0]
    kw_ret['memory_output'] = result[1]

    if get_gates:
        kw_ret['input_gates'] = result[4]
        kw_ret['forget_gates'] = result[5]
        kw_ret['output_gates'] = result[6]
        kw_ret['input_gates_att'] = result[7]
        kw_ret['forget_gates_att'] = result[8]
        kw_ret['output_gates_att'] = result[9]

    result = list(result)

    if dropout_params:
        result[0] = dropout_layer(result[0], *dropout_params)

    # Return memory c at the last in kw_ret
    return result[0], result[2], result[3], kw_ret


__all__ = [
    'param_init_group_lstm',
    'group_lstm_layer',
    'param_init_group_lstm_cond',
    'group_lstm_cond_layer',
]
