import argparse
import time
import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import add_self_loop
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

import pandas as pd
output_data_list = []

class SAGE(nn.Module):
    def __init__(self, embed_size, hid_size, out_size, num_nodes):
        super().__init__()
        self.embed_layer = nn.Embedding(num_nodes, embed_size)
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(embed_size, hid_size, "gcn"))
        self.layers.append(dglnn.SAGEConv(hid_size, out_size, "gcn"))
        self.dropout = nn.Dropout(0.5)

    def forward(self, graph, node_ids):
        h = self.embed_layer(node_ids)
        h = self.dropout(h)
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h


def evaluate(g, labels, mask, model):
    model.eval()
    with torch.no_grad():
        logits = model(g, g.nodes())
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def train(g, labels, masks, model):
    # define train/val samples, loss function and optimizer
    train_mask, val_mask = masks
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    total_time = 0

    # training loop
    for epoch in range(400):
        start_time = time.time()
        
        model.train()
        logits = model(g, g.nodes())
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_time = time.time() - start_time
        total_time += epoch_time

        acc = evaluate(g, labels, val_mask, model)
        acc_test = evaluate(g, labels, test_mask, model)
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} | Accuracy_test {:.4f} | Epoch Time {:.4f}s | Total Time {:.4f}s".format(
                epoch, loss.item(), acc, acc_test, epoch_time, total_time
            )
        )

        epoch_data = {
            "Accuracy": acc,
            "Accuracy_test": acc_test,
            "Epoch Time": epoch_time,
            "Total Time": total_time
        }
        output_data_list.append(epoch_data)
    
    df = pd.DataFrame(output_data_list)
    df.to_excel("/home/ubuntu/workspace/logs/"+"full_products"+"_"+str(args.num_hidden)+".xlsx", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GraphSAGE")
    parser.add_argument(
        "--dataset",
        type=str,
        default="ogbn-products",
        help="Dataset name ('ogbn-products')",
    )
    parser.add_argument(
        "--dt",
        type=str,
        default="float",
        help="data type(float, bfloat16)",
    )
    parser.add_argument(
        "--num_hidden",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--emb_size",
        type=int,
        default=16,
    )
    args = parser.parse_args()
    print(f"Training with DGL built-in GraphSage module")

    # load and preprocess dataset
    if args.dataset == "ogbn-products":
        data = DglNodePropPredDataset(name="ogbn-products")
        split_idx = data.get_idx_split()
        g, labels = data[0]
        labels = labels[:, 0]  # Remove the extra dimension

        # Add self-loop
        g = add_self_loop(g)
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = g.int().to(device)
    labels = labels.to(device)
    train_mask = split_idx["train"].to(device)
    val_mask = split_idx["valid"].to(device)
    test_mask = split_idx["test"].to(device)
    masks = (train_mask, val_mask)

    # create GraphSAGE model
    num_nodes = g.num_nodes()
    embed_size = args.emb_size  # Set the embedding size
    out_size = data.num_classes
    model = SAGE(embed_size, args.num_hidden, out_size, num_nodes).to(device)

    # convert model and graph to bfloat16 if needed
    if args.dt == "bfloat16":
        g = dgl.to_bfloat16(g)
        model = model.to(dtype=torch.bfloat16)

    # model training
    print("Training...")
    train(g, labels, masks, model)

    # test the model
    print("Testing...")
    acc = evaluate(g, labels, test_mask, model)
    print("Test accuracy {:.4f}".format(acc))
