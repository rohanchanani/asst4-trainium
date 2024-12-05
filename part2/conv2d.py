import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal


"""
A fused convolution - maxpool kernel that you need to implement for Part 2.

Parameters:
    X: the input tensor
    W: the weights of the convolution filters.
    bias: the biases of the convolution filters.
    pool_size: the size of the pool filter and pool stride.

expect: X.shape == [batch_size, in_channels, input_height, input_width]
expect: W.shape == [out_channels, in_channels, filter_height, filter_width]
expect: bias.shape == [out_channels]
expect: filter_height == filter_width
expect: pool_size == 1 || pool_size == 2
expect: input_channels % 128 == 0
expect: output_channels % 128 == 0

out_height = input_height - filter_height + 1
out_width = input_width - filter_width + 1

out_pool_height = out_height // pool_size
out_pool_width = out_width // pool_size

The shape of the output should be [batch_size, out_channels, out_pool_height, out_pool_width]

"""

@nki.jit
def fused_conv2d_maxpool(X, W, bias, pool_size=1):

    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]

    assert (
        in_channels_ == in_channels and out_channels_ == out_channels
    ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    out_pool_height = out_height // pool_size
    out_pool_width = out_width // pool_size
    
    # Can assume multiple of 128 to avoid using mask
    assert in_channels % 128 == 0

    # Can assume one PSUM bank can at least fit one row of the pixels
    assert nl.tile_size.gemm_moving_fmax >= out_width

    # Initialize output array
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    # Various tiling dimensions (You may want to define more of them)
    c_in_pmax = nl.tile_size.pmax
    n_tiles_c_in = in_channels // c_in_pmax
    c_out_pmax = nl.tile_size.pmax
    n_tiles_c_out = out_channels // c_out_pmax
    height_max = 128 - filter_height
    width_max = 128-filter_width
    n_tiles_height = (out_height+(height_max-1)) // height_max
    n_tiles_width = (out_width+(width_max-1)) // width_max
    actual_height = (out_height // n_tiles_height)
    while actual_height % pool_size != 0:
        actual_height+=1
    actual_width = (out_width // n_tiles_width)

    print(f"Actual: {actual_height}, Total: {out_height}, Num Tiles: {n_tiles_height}")



    weights = nl.ndarray(shape=(n_tiles_c_out, c_out_pmax, n_tiles_c_in, nl.par_dim(c_in_pmax), filter_height, filter_width), dtype=W.dtype, buffer=nl.sbuf)
    
    for i in nl.affine_range(n_tiles_c_in):
        for j in nl.affine_range(n_tiles_c_out):
            for k in nl.affine_range(c_out_pmax):
                weights[j, k, i, :, :, :] = nl.load(W[j*c_out_pmax+k, i*c_in_pmax:(i+1)*c_in_pmax, :,:])
    
    prepared_weights = nl.ndarray(shape=(filter_height, filter_width, n_tiles_c_out, n_tiles_c_in, nl.par_dim(c_in_pmax), c_out_pmax), dtype=W.dtype, buffer=nl.sbuf)
    for i in nl.affine_range(n_tiles_c_out):
        for j in nl.affine_range(n_tiles_c_in):
            for k in nl.affine_range(c_out_pmax):
                for m in nl.affine_range(filter_height):
                    for n in nl.affine_range(filter_width):
                        prepared_weights[m, n, i, j, :, k] = nl.copy(weights[i, k, j, :, m, n])

    
    # Process the images in batches
    for b in nl.affine_range(batch_size):
        for a in nl.affine_range(n_tiles_height):
            img = nl.ndarray(shape=(n_tiles_c_in, nl.par_dim(c_in_pmax), actual_height+filter_height-1, input_width), dtype=X.dtype, buffer=nl.sbuf)
            for i in nl.affine_range(n_tiles_c_in):
                i_p = a*actual_height + nl.arange(actual_height+filter_height-1)[None, None, :, None]
                img[i, :, :, :] = nl.load(X[b, i*c_in_pmax:(i+1)*c_in_pmax, a*actual_height:(a+1)*actual_height+filter_height-1, :], mask=i_p<input_height)

        
            for i in nl.affine_range(n_tiles_c_out):
                curr_output = nl.ndarray(shape=(nl.par_dim(c_out_pmax), actual_height, out_width), dtype=X.dtype, buffer=nl.sbuf)
                for curr_row in nl.affine_range(actual_height):
                    output_row = nl.zeros(shape=(c_out_pmax,out_width), dtype=X.dtype, buffer=nl.psum)
                    for y in nl.affine_range(filter_height):
                        for x in nl.affine_range(filter_width):
                            for j in nl.affine_range(n_tiles_c_in):
                                output_row[:, :]+=nl.matmul(prepared_weights[y, x, i, j, :, :], img[j, :, curr_row+y, x:x+out_width], transpose_x=True)
                    curr_output[:,curr_row, :] = nl.copy(output_row)
                nl.store(X_out[b, i*c_out_pmax:(i+1)*c_out_pmax, a*actual_height:(a+1)*actual_height, :], value=curr_output[...])

    return X_out

