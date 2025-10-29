import torch
import torch.nn.functional as F
import itertools
from torch_geometric.utils import erdos_renyi_graph

from models.triton_nn.mean_gnn import TritonMeanGNN
from helpers.gnn_type import GraphConvWrap
from helpers.utils import coo_to_csr, set_seed
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(42)


# Create a toy graph
def create_toy_graph():
    edge_index = erdos_renyi_graph(100, 0.2, directed=True).to(DEVICE)  # Example edges
    x = torch.randn(100, 33).to(DEVICE)  # 5 nodes, 4 features each
    return x, edge_index


# Function to compare forward and backward
def compare_layers():
    x, edge_index = create_toy_graph()
    channel = x.shape[1]

    # Define both models
    tri_layer1 = TritonMeanGNN(in_channels=channel, out_channels=channel).to(DEVICE)
    tri_layer2 = TritonMeanGNN(in_channels=channel, out_channels=channel).to(DEVICE)
    layer1 = GraphConvWrap(in_channels=channel, out_channels=channel, bias=False, aggr='mean').to(DEVICE)
    layer2 = GraphConvWrap(in_channels=channel, out_channels=channel, bias=False, aggr='mean').to(DEVICE)

    tri_layer1.aggr_lin.weight = layer1.lin_rel.weight
    tri_layer1.self_lin.weight = layer1.lin_root.weight

    tri_layer2.aggr_lin.weight = layer2.lin_rel.weight
    tri_layer2.self_lin.weight = layer2.lin_root.weight

    # optimizer
    tri_param = itertools.chain(tri_layer1.parameters(), tri_layer2.parameters())
    param = itertools.chain(layer1.parameters(), layer2.parameters())
    tri_optimizer = torch.optim.SGD(params=tri_param, lr=1e-3)
    optimizer = torch.optim.SGD(params=param, lr=1e-3)
    tri_optimizer.zero_grad()
    optimizer.zero_grad()

    # hand calculation
    adj_mat = torch.zeros(size=(x.shape[0], x.shape[0]))
    u, v = edge_index
    adj_mat[v, u] = 1
    deg_inv = 1 / adj_mat.sum(dim=1)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)
    deg_inv = torch.diag(deg_inv)
    norm_adj = (deg_inv @ adj_mat).to(DEVICE)
    mat_out = norm_adj @ x @ layer1.lin_rel.weight.T + x @ layer1.lin_root.weight.T
    mat_out = norm_adj @ mat_out @ layer2.lin_rel.weight.T + mat_out @ layer2.lin_root.weight.T

    # forward
    out = layer1(x, edge_index)
    out = layer2(out, edge_index)
    rowptr, indices = coo_to_csr(edge_index[1], edge_index[0], num_nodes=x.shape[0])
    tri_out = tri_layer1(x, edge_index=edge_index, rowptr=rowptr, indices=indices)
    tri_out = tri_layer2(tri_out, edge_index=edge_index, rowptr=rowptr, indices=indices)

    print("Forward Outputs:")
    print("Triton Output:", tri_out[:5])
    print("Regular Output:", out[:5])
    print("Matrix Output:", mat_out[:5])
    assert torch.allclose(tri_out, out, rtol=5e-3), f"Triton MeanGNN forward does not work with regular"
    print(f"Triton MeanGNN forward works!")

    # Compute losses
    target = torch.randn_like(out)
    tri_loss = F.mse_loss(tri_out, target)
    loss = F.mse_loss(out, target)
    tri_loss.backward()
    loss.backward()
    tri_optimizer.step()
    optimizer.step()

    # Forward pass
    out = layer1(x, edge_index)
    out = layer2(out, edge_index)
    tri_out = tri_layer1(x, edge_index=edge_index, rowptr=rowptr, indices=indices)
    tri_out = tri_layer2(tri_out, edge_index=edge_index, rowptr=rowptr, indices=indices)

    print("\nBackward Comparison:")
    print("Triton Output:", tri_out[:5])
    print("Regular Output:", out[:5])
    assert torch.allclose(tri_out, out, rtol=5e-3), f"Triton MeanGNN backward does not work"
    print(f"Triton MeanGNN backward works!\n")


# Run the comparison
compare_layers()
