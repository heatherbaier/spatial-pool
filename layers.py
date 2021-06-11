class GraphConv(nn.Module):
    """
    GCN basic operation 
    BATCHING: This should be resistent to batching because it's just creating the node embeddings
    TO-DO: Double check the matmul stuff, this was taken from Ying's github so just confirm it's doing what you think
    """
    def __init__(self, input_dim, output_dim, bias = True):
        super(GraphConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.rand(input_dim, output_dim))#.cuda())
        if bias:
            self.bias = nn.Parameter(torch.rand(output_dim))#.cuda())
        else:
            self.bias = None

    def forward(self, x, adj):
        y = torch.matmul(adj, x)
        y = torch.matmul(y, self.weight)
        if self.bias is not None:
            y = y + self.bias
        return y


class GenAssign(nn.Module):
    """
    Generate Assignment Matrix S_l
    BATCHING: You could batch this or not. 
              1) Argument FOR batching: could pool features based on geography from other municiaplites.
              2) Argument AGAINST batching: In repsonse to point above, that could be okay because we are still taking features by                   muni in the next layer   
              - REGARDLESS: Make it clear which option you chose and why
              - FOR NOW: No batching    
              - YOU DECIDED AGAINST BECAUSE IT ACTUALLY SEEMS TO BE CLUSTERING BY MUNI WITHOUT BATCHING
    TO-DO: Why did you take out the linear layer?
    """
    def __init__(self, input_dim, num_clusters, bias=True):
        super(GenAssign, self).__init__()
        self.input_dim = input_dim
        self.num_clusters = num_clusters
        self.weight = nn.Parameter(torch.rand(input_dim, num_clusters))#.cuda())
        if bias:
            self.bias = nn.Parameter(torch.rand(num_clusters))#.cuda())
        else:
            self.bias = None
        # self.linear = torch.nn.Linear(input_dim, num_clusters)
        self.sm = torch.nn.Softmax(dim = 1)

    def forward(self, x, adj, batch):

        batch_ids = torch.unique(batch)
        a = 1

        for i in batch_ids:

            batch_indexes = torch.nonzero(batch == i, as_tuple = True)[0]
            min_index, max_index = torch.min(batch_indexes), torch.max(batch_indexes) + 1
            batch_x = torch.index_select(x, 0, batch_indexes)
            batch_adj = adj[min_index:max_index, min_index:max_index]

            y = torch.matmul(batch_adj, batch_x)
            y = torch.matmul(y, self.weight)
            if self.bias is not None:
                y = y + self.bias
            # y = self.linear(y)
            y = self.sm(y)
            _, i = torch.max(y, 1)

            if a == 1:
                out = i
            else:
                out = torch.cat((out, i))
            a += 1

        return out


class PoolClusters(nn.Module):
    """
    Pool based on clusters and assignment matrix (S_l) created in GenAssign
    BATCHING: This definitley needs to be batched
    """
    def __init__(self):
        super(PoolClusters, self).__init__()

    def forward(self, x, sl, batch):

        cluster_ids = torch.unique(sl)
        batch_ids = torch.unique(batch)
        a = 1

        for i in batch_ids:

            batch_indexes = torch.nonzero(batch == i, as_tuple = True)[0]
            batch_x = torch.index_select(x, 0, batch_indexes)

            for cluster in cluster_ids:

                indices = torch.nonzero(sl == cluster, as_tuple = True)[0]
                min_index, max_index = torch.min(batch_indexes), torch.max(batch_indexes)
                indices = torch.unique(torch.clip(indices, min = min_index, max = max_index))
                test_mean = torch.mean(torch.index_select(x, 0, indices), dim = 0).unsqueeze(0)
                
                if a == 1:
                    out = test_mean
                else:
                    out = torch.cat((out, test_mean))
                a += 1

        return out


def make_edge_list(neighbors_dict):
    edge_list = []
    for k,v in neighbors_dict.items():
        [edge_list.append([k, cur_v]) for cur_v in v]
    return edge_list

def make_adj_matrix(edge_list, dim):
    adj_matrix = np.zeros((dim, dim))
    for edge in edge_list:
        adj_matrix[edge[0]][edge[1]] = 1
    for i in range(dim):
        adj_matrix[i][i] = 1

    return adj_matrix


def fix_sl(sl):
    cluster_ids = torch.unique(sl)
    ref = dict(zip([i.item() for i in cluster_ids], [i for i in range(len(cluster_ids))]))
    for i in cluster_ids:
        sl[sl == i.item()] = ref[i.item()]
    return sl


# GCN basic operation
class GenAdj(nn.Module):
    def __init__(self):
        super(GenAdj, self).__init__()

    def forward(self, adj, sl, neighbors_dict):

        sl = fix_sl(sl)

        adj_map, new_neighbors_dict = {}, {}
        cluster_ids = torch.unique(sl)

        # Create a dictionary where each key is a node and the value is the cluster it belongs to
        for cluster in cluster_ids:
            indices = torch.nonzero(sl == cluster, as_tuple = True)[0]
            for i in indices:
                if i not in adj_map.keys():
                    adj_map[i.item()] = cluster.item()

        # Create a new neighbors dictionary for each of the clusters
        for cluster in cluster_ids:
            indices = torch.nonzero(sl == cluster, as_tuple = True)[0]

            for i in indices:
                temp = []
                [temp.append(adj_map[i]) for i in neighbors_dict[i.item()] if i not in temp]
                temp = list(set(temp)) # TO-DO: HERE CREATE SOME SORT OF SECONDARY DICTIONARY OR SOMETHING THAT COUNTS HOW MANY EDGES TO CREATE WEIGHTED ADJ MATRIX

                new_neighbors_dict[cluster.item()] = temp

        edge_list = make_edge_list(new_neighbors_dict)
        adj_matrix = make_adj_matrix(edge_list, len(cluster_ids))

        return torch.tensor(adj_matrix, dtype = torch.float32), new_neighbors_dict


class diffPool(nn.Module):
    def __init__(self, input_dim, output_dim, num_clusters, bias=True):
        super(diffPool, self).__init__()

        self.gc = GraphConv(input_dim = input_dim, output_dim = output_dim)
        self.bn = torch.nn.BatchNorm1d(output_dim)
        self.gen_assign = GenAssign(input_dim = output_dim, num_clusters = num_clusters)
        self.pool = PoolClusters()
        self.gen_adj = GenAdj()

    def forward(self, x, adj, neighbors_dict):

        # Generate node embeddings & normalize (no pooling yet)
        x = self.gc(x, torch.tensor(adj, dtype = torch.float32))
        if x.shape[0] > 1:
            x = self.bn(x)
        # Create the assignment matrix (i.e. assign each node to a cluster)
        sl = self.gen_assign(x, adj)
        # print("SL: ", sl)
        # Create the pooled feature matrix
        x = self.pool(x, sl)
        # Generate the new adjacency matrix
        adj, neighbors_dict = self.gen_adj(adj, sl, neighbors_dict)
        # print(neighbors_dict)

        return x, adj, neighbors_dict


class spatialPool(nn.Module):
    def __init__(self, input_dim, output_dim, num_clusters, resnet, bias=True):
        super(spatialPool, self).__init__()

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        self.dp1 = diffPool(input_dim = input_dim, output_dim = output_dim, num_clusters = 4)
        self.dp2 = diffPool(input_dim = output_dim, output_dim = 128, num_clusters = 2)
        self.linear = torch.nn.Linear(128, 1)

    def forward(self, x, adj, neighbors_dict):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(start_dim=1)

        x, adj, neighbors_dict = self.dp1(x, adj, neighbors_dict)
        x, adj, neighbors_dict = self.dp2(x, adj, neighbors_dict)
        x = self.linear(x)

        return x