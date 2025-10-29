import triton
import triton.language as tl
import torch


class ConvGATAggr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h_in, adj_rowptr, adj_indices, att_src, att_dst, out_node_count, work_group_size=None):
        torch.cuda.set_device(h_in.device)
        num_features_per_node = h_in.shape[1]
        if not work_group_size:
            work_group_size = 32
            while work_group_size < num_features_per_node:
                work_group_size *= 2
        num_work_groups = (num_features_per_node + work_group_size - 1) // work_group_size
        num_nodes = out_node_count
        h_out = torch.empty((out_node_count, h_in.shape[1]), dtype=h_in.dtype, device=h_in.device)
        gat_forward[(num_nodes, num_work_groups)](
            h_out, adj_rowptr, adj_indices, h_in, att_src, att_dst,
            num_features_per_node, work_group_size, num_warps=32
        )
        ctx.save_for_backward(adj_rowptr, adj_indices, h_in, att_src, att_dst)
        return h_out

    @staticmethod
    def backward(ctx, dh_out):
        adj_rowptr, adj_indices, h_in, att_src, att_dst = ctx.saved_tensors
        dh_src = torch.zeros_like(h_in, requires_grad=False)
        datt_src_out = torch.zeros_like(att_src, requires_grad=False)
        datt_dst_out = torch.zeros_like(att_dst, requires_grad=False)
        num_features_per_node = dh_out.shape[1]
        work_group_size = 32
        while work_group_size < num_features_per_node:
            work_group_size *= 2
        num_work_groups = (num_features_per_node + work_group_size - 1) // work_group_size
        num_nodes = dh_out.shape[0]
        gat_backward[(num_nodes, num_work_groups)](
            dh_src, adj_rowptr, adj_indices, h_in, dh_out, att_src, att_dst, datt_src_out, datt_dst_out,
            num_features_per_node, work_group_size, num_warps=32
        )
        return dh_src, None, None, datt_src_out, datt_dst_out, None, None


@triton.jit
def gat_forward(
        out_h, adj_rowptr, adj_indices, h_src, att_src, att_dst,
        IN_CHAN: tl.constexpr, WG_SIZE: tl.constexpr
):
    node_index_i = tl.program_id(0)
    feat_offsets = tl.arange(0, WG_SIZE) + tl.program_id(1) * WG_SIZE
    feat_valid_mask = feat_offsets < IN_CHAN
    feat_zeros = tl.zeros((WG_SIZE,), dtype=tl.float32)

    col_start = tl.load(adj_rowptr + node_index_i)
    col_end = tl.load(adj_rowptr + node_index_i + 1)
    col_count = col_end - col_start

    aggr_sum = feat_zeros
    attn_sum = 0.0

    att_src_vec = tl.load(att_src + feat_offsets, feat_valid_mask, feat_zeros)
    att_dst_vec = tl.load(att_dst + feat_offsets, feat_valid_mask, feat_zeros)
    feat_i = tl.load(h_src + node_index_i * IN_CHAN + feat_offsets, feat_valid_mask, feat_zeros)
    max_attn = -1e9
    # First, compute the maximum attention score for numerical stability
    for index in range(col_count):
        node_index_j = tl.load(adj_indices + col_start + index)
        neighbor_feat_j = tl.load(h_src + node_index_j * IN_CHAN + feat_offsets, feat_valid_mask, feat_zeros)

        # Compute the attention coefficient as the sum of source and target attention coefficients
        attn_ij = tl.sum(att_dst_vec * feat_i + att_src_vec * neighbor_feat_j, axis=0)  # Sum of alpha_j and alpha_i
        max_attn = tl.maximum(max_attn, attn_ij)  # Track the maximum attention coefficient

    for index in range(col_count):
        node_index_j = tl.load(adj_indices + col_start + index)
        neighbor_feat_j = tl.load(h_src + node_index_j * IN_CHAN + feat_offsets, feat_valid_mask, feat_zeros)

        # Compute attention coefficient as the sum of source and target attention coefficients
        attn_ij = tl.sum(att_dst_vec * feat_i + att_src_vec * neighbor_feat_j, axis=0)  # Sum of alpha_j and alpha_i
        attn_ij = tl.where(attn_ij < 0, 0.2 * attn_ij, attn_ij)  # LeakyReLU
        attn_ij = tl.exp(attn_ij - max_attn)  # Subtract max for numerical stability

        aggr_sum += attn_ij * neighbor_feat_j  # Accumulate weighted sum
        attn_sum += attn_ij  # Accumulate attention sum

    # Normalize aggregated features
    attn_sum = tl.where(attn_sum == 0, 1, attn_sum)  # Avoid division by zero
    aggr_mean = aggr_sum / attn_sum  # Normalize using accumulated sum
    tl.store(out_h + node_index_i * IN_CHAN + feat_offsets, aggr_mean, feat_valid_mask)


@triton.jit
def gat_backward(
        dh_src, adj_rowptr, adj_indices, h_in, dh_out, att_src, att_dst, datt_src_out, datt_dst_out,
        IN_CHAN: tl.constexpr, WG_SIZE: tl.constexpr
):
    node_index_i = tl.program_id(0)
    feat_offsets = tl.arange(0, WG_SIZE) + tl.program_id(1) * WG_SIZE
    feat_valid_mask = feat_offsets < IN_CHAN
    feat_zeros = tl.zeros((WG_SIZE,), dtype=tl.float32)

    col_start = tl.load(adj_rowptr + node_index_i)
    col_end = tl.load(adj_rowptr + node_index_i + 1)
    col_count = col_end - col_start

    h_out_grad = tl.load(dh_out + node_index_i * IN_CHAN + feat_offsets, feat_valid_mask, feat_zeros)
    attn_sum = 0.0

    # Initialize gradients for att_src and att_dst
    datt_src = feat_zeros
    datt_dst = feat_zeros

    # Load attention weight vectors
    att_src_vec = tl.load(att_src + feat_offsets, feat_valid_mask, feat_zeros)
    att_dst_vec = tl.load(att_dst + feat_offsets, feat_valid_mask, feat_zeros)

    # Load the source node feature
    feat_i = tl.load(h_in + node_index_i * IN_CHAN + feat_offsets, feat_valid_mask, feat_zeros)

    # First, compute the maximum attention coefficient for numerical stability
    max_attn = -1e9
    for index in range(col_count):
        node_index_j = tl.load(adj_indices + col_start + index)
        neighbor_feat_j = tl.load(h_in + node_index_j * IN_CHAN + feat_offsets, feat_valid_mask, feat_zeros)

        # Compute attention score
        attn_ij = tl.sum(att_dst_vec * feat_i + att_src_vec * neighbor_feat_j, axis=0)
        max_attn = tl.maximum(max_attn, attn_ij)  # Track the max attention score

    # Now compute the attention scores with numerical stability (subtract max for stability)
    for index in range(col_count):
        node_index_j = tl.load(adj_indices + col_start + index)
        neighbor_feat_j = tl.load(h_in + node_index_j * IN_CHAN + feat_offsets, feat_valid_mask, feat_zeros)

        # Compute the attention score
        attn_ij = tl.sum(att_dst_vec * feat_i + att_src_vec * neighbor_feat_j, axis=0)
        attn_ij_leaky = tl.where(attn_ij < 0, 0.2, 1.0)  # LeakyReLU derivative
        attn_ij = attn_ij * attn_ij_leaky
        attn_ij = tl.exp(attn_ij - max_attn)  # Subtract max for numerical stability
        attn_sum += attn_ij

        # Compute gradients for attention weights
        datt_dst += attn_ij * h_out_grad * feat_i
        datt_src += attn_ij * h_out_grad * neighbor_feat_j

    # Normalize the gradient by the attention sum (avoiding division by zero)
    attn_sum = tl.where(attn_sum == 0, 1, attn_sum)  # Avoid division by zero
    h_out_grad /= attn_sum  # Normalize the gradient

    # Accumulate gradients for dh_src
    for index in range(col_count):
        node_index_j = tl.load(adj_indices + col_start + index)
        tl.atomic_add(dh_src + node_index_j * IN_CHAN + feat_offsets, h_out_grad, feat_valid_mask)

    # Store the gradients for att_src and att_dst
    tl.store(datt_src_out + feat_offsets, datt_src, feat_valid_mask)
    tl.store(datt_dst_out + feat_offsets, datt_dst, feat_valid_mask)
