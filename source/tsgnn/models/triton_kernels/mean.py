import torch

import triton
import triton.language as tl


@triton.jit
def relconv_aggregate_mean_kernel_forward(
        out_h,  # h_out output tensor of shape[num_nodes_dst,IN_CHAN] ; This stores places for the output
        adj_rowptr,  # graph adjacency matrix (rowptr) of shape[num_nodes_dst+1] ; a cumulative count in CSR for each row. This is the pointer pointing the start of the memory
        adj_indices,  # graph adjacency matrix (indices) of shape[num_edges]; the "column index" count, position for each nodes
        h_src,  # h_src tensor of shape[num_nodes_src, IN_CHAN] ; This is the input x
        IN_CHAN: tl.constexpr,  # number of features per node we are considering in this thread
        WG_SIZE: tl.constexpr,  # workgroup size
):
    # Pseudo-code (functional):
    #   for node_i in range(out_h.shape[0]):
    #     out_h[node_i,:] = 0
    #     col_start = adj_rowptr[node_i]
    #     col_count = adj_rowptr[node_i+1] – col_start
    #     for icol in range(col_count):
    #       node_j = adj_indices[col_start + icol]
    #       rel_j = rel_indices[col_start + icol]
    #       out_h[node_i,:] += h_src[node_j,:] * rels[rel_j, :]

    # (Triton implementation for float32)
    # get current node_index and features to process
    node_index_i = tl.program_id(0)  # Get which node we are operating one
    feat_offsets = tl.arange(0, WG_SIZE) + tl.program_id(
        1) * WG_SIZE  # tl.program_id(1) = which feature group we are working on. There might be multiple groups
    feat_valid_mask = feat_offsets < IN_CHAN
    feat_zeros = tl.zeros((WG_SIZE,), dtype=tl.float32)
    # identity list of node indices to aggregate. Note these are sorted in the CSR format
    col_start = tl.load(
        adj_rowptr + node_index_i)  # To extract the row[node_index_i], we want to get the index of starting column and ending column
    col_end = tl.load(adj_rowptr + node_index_i + 1)
    col_count = col_end - col_start  # The sliced indexes for the extracted row (indicated by the column index).
    # aggregate neighboring features
    aggr_sum = feat_zeros
    for index in range(col_count):
        node_index_j = tl.load(
            adj_indices + col_start + index)  # adj_indices points the stored total column index, then col_start is the start of the current considered row, index is the current edges in the considered row, which is what we want
        neighbor_feat_j = tl.load(h_src + node_index_j * IN_CHAN + feat_offsets, feat_valid_mask, feat_zeros)
        aggr_sum += neighbor_feat_j
    col_count = tl.where(col_count == 0, 1, col_count)  # MEAN
    aggr_mean = aggr_sum / col_count  # MEAN
    # store aggregator output
    tl.store(out_h + node_index_i * IN_CHAN + feat_offsets, aggr_mean, feat_valid_mask)

@triton.jit
def relconv_aggregate_mean_kernel_backward(
        dh_src,  # pre-initialized dh_src output tensor of shape[num_nodes_src,IN_CHAN]
        adj_rowptr,  # graph adjacency matrix (rowptr) of shape[num_nodes_dst+1]
        adj_indices,  # graph adjacency matrix (indices) of shape[num_edges]
        h_in,  # original input features of shape [num_nodes, IN_CHAN]
        dh_out,  # MAIN GRADIENT: dh_out tensor of shape[num_nodes_src,IN_CHAN]
        IN_CHAN: tl.constexpr,  # number of features per head
        WG_SIZE: tl.constexpr,  # workgroup size
):
    # Pseudo-code (functional):
    #   for node_i in range(dh_out.shape[0]):
    #     col_start = adj_rowptr[node_i]
    #     col_count = adj_rowptr[node_i+1] – col_start
    #     for icol in range(col_count):
    #       node_j = adj_indices[col_start + icol]
    #       rel_j = rel_indices[col_start + icol]
    #       nodej_feat = hin[node_j]
    #       relj_feat = rels[rel_j]
    #       dh_src[node_i,:] += dh * relj_feat
    #       drel_src[rel_j, :] += dh * nodej_feat

    # (Triton implementation for float32)
    # get current node_index and features to process
    node_index_i = tl.program_id(0)
    feat_offsets = tl.arange(0, WG_SIZE) + tl.program_id(1) * WG_SIZE
    feat_valid_mask = feat_offsets < IN_CHAN
    feat_zeros = tl.zeros((WG_SIZE,), dtype=tl.float32)
    # identify list of node indices to aggregate
    col_start = tl.load(adj_rowptr + node_index_i)
    col_end = tl.load(adj_rowptr + node_index_i + 1)
    col_count = col_end - col_start
    # load output gradient and scale
    h_out_grad = tl.load(dh_out + node_index_i * IN_CHAN + feat_offsets,
                         feat_valid_mask, feat_zeros)
    h_out_grad = h_out_grad / tl.where(col_count == 0, 1, col_count)  # MEAN
    # accumulate gradient to corresponding inputs
    for index in range(col_count):
        node_index_j = tl.load(adj_indices + col_start + index)
        tl.atomic_add(dh_src + node_index_j * IN_CHAN + feat_offsets, h_out_grad, feat_valid_mask)


class ConvMeanAggr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h_in, rowptr, indices, out_node_count, work_group_size=None):
        # need to set the current CUDA device to avoid the error
        # ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)
        # https://github.com/openai/triton/issues/2925
        torch.cuda.set_device(h_in.device)
        # get node feature count
        num_features_per_node = h_in.shape[1]
        # if work_group_size is not specified, pick a default value based on node feature count
        if not work_group_size:
            work_group_size = 32
            while work_group_size < num_features_per_node:
                work_group_size *= 2
        # calculate kernel configuration
        num_work_groups = (num_features_per_node + work_group_size - 1) // work_group_size
        num_nodes = out_node_count
        # invoke triton forward kernel
        h_out = torch.empty((out_node_count, h_in.shape[1]), dtype=h_in.dtype, layout=h_in.layout, device=h_in.device)
        relconv_aggregate_mean_kernel_forward[(num_nodes, num_work_groups)](
            h_out, rowptr, indices, h_in,
            num_features_per_node, work_group_size, num_warps=32)  # fixing num_warps to 32 as in the cuda rspmm kernel
        # save parameters for backward
        h_in_grad = torch.zeros_like(h_in, requires_grad=False)
        work_group_size_shaped_dummy = torch.empty(work_group_size, dtype=torch.int8)
        ctx.save_for_backward(rowptr, indices, h_in_grad, h_in, work_group_size_shaped_dummy)
        return h_out

    @staticmethod
    def backward(ctx, h_out_grad):
        # get saved variables from forward
        rowptr, indices, h_in_grad, h_in, work_group_size_shape_dummy = ctx.saved_tensors
        work_group_size = work_group_size_shape_dummy.shape[0]
        # calculate kernel configuration
        num_features_per_node = h_out_grad.shape[1]
        num_work_groups = (num_features_per_node + work_group_size - 1) // work_group_size
        num_nodes = h_out_grad.shape[0]
        # invoke triton backward kernel
        relconv_aggregate_mean_kernel_backward[(num_nodes, num_work_groups)](
            h_in_grad, rowptr, indices, h_in, h_out_grad, num_features_per_node, work_group_size, num_warps=32)
        return h_in_grad, None, None, None, None
