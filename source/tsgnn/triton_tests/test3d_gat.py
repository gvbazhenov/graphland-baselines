import torch
import torch.nn.functional as F
import itertools
from torch_geometric.utils import softmax as sparse_softmax
from torch_scatter import scatter
from torch_geometric.utils import erdos_renyi_graph

from models.triton_nn.gat import TritonGAT
from triton_tests.gat import GATConv
from helpers.utils import coo_to_csr, set_seed
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(42)


# Create a toy graph
def create_toy_graph():
    edge_index = erdos_renyi_graph(100, 0.2, directed=True).to(DEVICE)  # Example edges
    x = torch.randn(100, 33, 15).to(DEVICE)  # 5 nodes, 4 features each
    return x, edge_index


# Function to compare forward and backward
def compare_layers():
    x, edge_index = create_toy_graph()
    channel = x.shape[2]

    # Define both layers
    tri_layer1 = TritonGAT(in_channels=channel, out_channels=channel, add_self_loops=False).to(DEVICE)
    tri_layer2 = TritonGAT(in_channels=channel, out_channels=channel, add_self_loops=False).to(DEVICE)
    layer1 = GATConv(in_channels=channel, out_channels=channel, heads=1, add_self_loops=False).to(DEVICE)
    layer2 = GATConv(in_channels=channel, out_channels=channel, heads=1, add_self_loops=False).to(DEVICE)

    tri_layer1.aggr_lin.weight = layer1.lin.weight
    tri_layer1.att_src.weight.data = layer1.att_src.data.squeeze(dim=1).squeeze(dim=1)
    tri_layer1.att_dst.weight.data = layer1.att_dst.data.squeeze(dim=1).squeeze(dim=1)

    tri_layer2.aggr_lin.weight = layer2.lin.weight
    tri_layer2.att_src.weight.data = layer2.att_src.data.squeeze(dim=1).squeeze(dim=1)
    tri_layer2.att_dst.weight.data = layer2.att_dst.data.squeeze(dim=1).squeeze(dim=1)

    # optimizer
    tri_param = itertools.chain(tri_layer1.parameters(), tri_layer2.parameters())
    param = itertools.chain(layer1.parameters(), layer2.parameters())
    tri_optimizer = torch.optim.SGD(params=tri_param, lr=1e-3)
    optimizer = torch.optim.SGD(params=param, lr=1e-3)
    tri_optimizer.zero_grad()
    optimizer.zero_grad()

    # hand calculation
    new_x = x @ layer1.lin.weight.T
    u, v = edge_index
    logits = new_x[u] @ layer1.att_src.data.squeeze(dim=1).squeeze(dim=1).T\
             + new_x[v] @ layer1.att_dst.data.squeeze(dim=1).squeeze(dim=1).T
    logits = logits.mean(dim=1, keepdim=True)
    logits = F.leaky_relu(logits, negative_slope=0.2)
    weight = sparse_softmax(logits, v)  # Normalize over incoming node
    mat_out = torch.zeros_like(new_x)
    mat_out = scatter(weight * new_x[u], v, dim=0, out=mat_out)  # Weight and sum
    new_x = mat_out @ layer2.lin.weight.T
    u, v = edge_index
    logits = new_x[u] @ layer2.att_src.data.squeeze(dim=1).squeeze(dim=1).T \
             + new_x[v] @ layer2.att_dst.data.squeeze(dim=1).squeeze(dim=1).T
    logits = logits.mean(dim=1, keepdim=True)
    logits = F.leaky_relu(logits, negative_slope=0.2)
    weight = sparse_softmax(logits, v)  # Normalize over incoming node
    mat_out = torch.zeros_like(new_x)
    mat_out = scatter(weight * new_x[u], v, dim=0, out=mat_out)  # Weight and sum

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
    assert torch.allclose(tri_out, out, rtol=5e-3), f"Triton GAT3D forward does not work with regular"
    print(f"Triton GAT3D forward works!")

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
    assert torch.allclose(tri_out, out, rtol=5e-3), f"Triton GAT3D backward does not work"
    print(f"Triton GAT3D backward works!\n")


# Run the comparison
compare_layers()
