
import os.path as osp
import math

import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.data import DenseDataLoader
from torch_geometric.nn import DenseSAGEConv, dense_mincut_pool, dense_diff_pool
from torch.nn import Parameter
EPS = 1e-15
max_nodes = 200
average_nodes = 32
import torch.nn as nn

class MyFilter(object):
    def __call__(self, data):
        return data.num_nodes <= max_nodes



def mincut4A(adj, s):

    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s

    k = s.size(-1)

    # s = torch.softmax(s, dim=-1)


    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

    # MinCUT regularization.
    mincut_num = _rank3_trace(out_adj)
    d_flat = torch.einsum('ijk->ij', adj)
    d = _rank3_diag(d_flat)
    mincut_den = _rank3_trace(
        torch.matmul(torch.matmul(s.transpose(1, 2), d), s))
    mincut_loss = -(mincut_num / mincut_den)
    mincut_loss = torch.mean(mincut_loss)

    # Orthogonality regularization.
    ss = torch.matmul(s.transpose(1, 2), s)
    i_s = torch.eye(k).type_as(ss)
    ortho_loss = torch.norm(
        ss / torch.norm(ss, dim=(-1, -2), keepdim=True) -
        i_s / torch.norm(i_s), dim=(-1, -2))
    ortho_loss = torch.mean(ortho_loss)

    # Fix and normalize coarsened adjacency matrix.
    ind = torch.arange(k, device=out_adj.device)
    out_adj[:, ind, ind] = 0
    d = torch.einsum('ijk->ij', out_adj)
    d = torch.sqrt(d)[:, None] + EPS
    out_adj = (out_adj / d) / d.transpose(1, 2)

    return out_adj, mincut_loss, ortho_loss


def _rank3_trace(x):
    return torch.einsum('ijj->i', x)


def _rank3_diag(x):
    eye = torch.eye(x.size(1)).type_as(x)
    out = eye * x.unsqueeze(2).expand(*x.size(), x.size(1))
    return out





def diff4A(adj, s):

    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s


    # s = torch.softmax(s, dim=-1)
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

    link_loss = adj - torch.matmul(s, s.transpose(1, 2))
    link_loss = torch.norm(link_loss, p=2)
    link_loss = link_loss / adj.numel()

    ent_loss = (-s * torch.log(s + EPS)).sum(dim=-1).mean()

    return out_adj, link_loss, ent_loss


path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'ENZYMES_d')
print(path)
# path = '/home/zyh/Documents/data/'
dataset = TUDataset(
    path,
    name='ENZYMES',
    transform=T.ToDense(max_nodes),
    pre_filter=MyFilter())

device = 'cpu'
torch.manual_seed(777)
if torch.cuda.is_available():
    torch.cuda.manual_seed(777)
    device = 'cuda:0'
dataset = dataset.shuffle()
n = (len(dataset) + 9) // 10
test_dataset = dataset[:n]
val_dataset = dataset[n:2 * n]
train_dataset = dataset[2 * n:]
test_loader = DenseDataLoader(test_dataset, batch_size=20)
val_loader = DenseDataLoader(val_dataset, batch_size=20)
train_loader = DenseDataLoader(train_dataset, batch_size=20)







class Prototype_Pooling(torch.nn.Module):
    def __init__(self, num_super_nodes, in_channels, hiden_channels, out_channels,improved=False, ln=True):
        super(Prototype_Pooling, self).__init__()

        self.in_channels = in_channels
        self.prototype_dim = 2*in_channels
        self.hiden_channels = hiden_channels
        self.out_channels = out_channels
        self.improved = improved
        self.num_super_nodes = num_super_nodes
        self.softmax4S = 2
        self.link_loss_type = 'mincut'
        self.ln = ln
        self.lsweight = 1.0
        print('softmax for S:', self.softmax4S)
        if self.link_loss_type=='mincut':
            print('mincut')
            self.link_loss_F = mincut4A
        else:
            print('diff')
            self.link_loss_F = diff4A

        self.meta_s = Parameter(torch.Tensor(self.num_super_nodes, self.prototype_dim))
        nn.init.xavier_uniform_(self.meta_s)
        self.adj_mask = Parameter(torch.ones(self.num_super_nodes, self.num_super_nodes)-torch.eye(self.num_super_nodes),requires_grad=False)

        self.shortcut=True

        self.fc_qs = nn.Linear(self.prototype_dim, self.hiden_channels)
        self.fc_ks = nn.Linear(self.in_channels, self.hiden_channels)
        self.fc_vs = nn.Linear(self.in_channels, self.hiden_channels)
        self.fc_0 = nn.Linear(self.in_channels+1, self.in_channels)
        self.fc_1 = nn.Linear(self.hiden_channels, self.hiden_channels)
        if self.ln:
            self.ln0 = nn.LayerNorm(self.hiden_channels)
            self.ln1 = nn.LayerNorm(self.hiden_channels)
        self.fc_2 = nn.Linear(self.hiden_channels, self.out_channels + 1)
        self.fc_3 = nn.Linear(2*self.num_super_nodes, self.num_super_nodes)
        self.fc_4 = nn.Linear(self.num_super_nodes,1)




    def forward(self, x, adj, mask):
        r"""
        Args:
            x (Tensor): Node feature tensor :math:`\mathbf{X} \in \mathbb{R}^{B
                \times N \times F}`, with batch-size :math:`B`, (maximum)
                number of nodes :math:`N` for each graph, and feature
                dimension :math:`F`.
            adj (Tensor): Adjacency tensor :math:`\mathbf{A} \in \mathbb{R}^{B
                \times N \times N}`. The adjacency tensor is broadcastable in
                the batch dimension, resulting in a shared adjacency matrix for
                the complete batch.
            mask (BoolTensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x
        # nan_check = torch.sum(torch.isnan(x))
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj

        B, N, _ = adj.size()
        meta_s_temp = self.meta_s.unsqueeze(0).expand(B,-1,-1)




        z = torch.relu(self.fc_0(torch.cat((x,mask),dim=-1)))


        # que_s = torch.relu(self.fc_qs(meta_s_temp))
        # key_z = torch.relu(self.fc_ks(z))
        # val_z = torch.relu(self.fc_vs(z))
        que_s = self.fc_qs(meta_s_temp)

        key_z = self.fc_ks(z)
        val_z = self.fc_vs(z)




        # dot mask or not
        # key_z = key_z * mask
        # val_z = val_z * mask

        attention = torch.matmul(que_s, key_z.transpose(2, 1)) / math.sqrt(self.hiden_channels)

        # dot mask or not
        # dot mask before softmax ?
        attention = torch.softmax(attention, dim=self.softmax4S)
        # attention = attention * mask.transpose(2,1)


        new_s = torch.matmul(attention, val_z)+que_s

        new_s = self.ln0(new_s)
        new_s = F.relu(self.fc_1(new_s)) + new_s
        new_s = self.ln1(new_s)
        new_s_fc = self.fc_2(new_s)
        new_s = torch.relu(new_s_fc[:,:,:-1])
        mask_out = torch.sigmoid(new_s_fc[:,:,-1]).view(B,self.num_super_nodes,1)
        # x_out = self.dense_gat(x=new_s, adj=out_adj)



        x_out=new_s
        attention = attention.transpose(2,1)
        mask_out_bp = mask_out.repeat(1, 1, N).transpose(-1, -2)
        mask_out_bp = torch.cat((mask_out_bp,attention),dim=-1)
        mask_out_bp = F.relu(self.fc_3(mask_out_bp))
        mask_out_bp = torch.sigmoid(self.fc_4(mask_out_bp))
        loss_2 = F.mse_loss(input=mask_out_bp, target=mask)
        out_adj,loss_0,loss_1 = self.link_loss_F(adj=adj,s=attention)

        out_adj = out_adj * self.adj_mask



        return x_out, out_adj, mask_out, loss_0, loss_1, loss_2

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)






class Embedded_Pool(torch.nn.Module):
    def __init__(self, num_super_nodes, in_channels, hiden_channels, out_channels):
        super(Embedded_Pool, self).__init__()
        self.in_channels = in_channels
        self.link_loss_type = 'mincut'
        self.num_super_nodes = num_super_nodes
        self.hiden_channels = hiden_channels
        self.out_channels = out_channels
        self.get_s = nn.Linear(in_features=self.in_channels, out_features=num_super_nodes)
        if self.link_loss_type=='mincut':
            print('mincut')
            self.pool = dense_mincut_pool
        else:
            print('diff')
            self.pool = dense_diff_pool
    def forward(self, x, adj, mask=None, add_loop=False):
        s = self.get_s(x)

        x, out_adj, loss_0, loss_1 = self.pool(x=x, adj=adj, s=s,mask=mask)

        return x, out_adj, None, loss_0, loss_1, torch.tensor(0.0)







class Net(torch.nn.Module):
    def __init__(self, feat_in, n_classes, n_chan=64):
        super(Net, self).__init__()

        num_nodes = math.ceil(0.4 * average_nodes)
        self.gnn1 = DenseSAGEConv(feat_in, n_chan)
        # self.pool1 = torch.nn.Linear(n_chan, num_nodes)
        # self.pool1 = Prototype_Pooling(num_nodes, n_chan, n_chan, n_chan)
        self.pool1 = Embedded_Pool(num_nodes, n_chan, n_chan, n_chan)

        num_nodes = math.ceil(0.4 * num_nodes)
        self.gnn2 = DenseSAGEConv(n_chan, n_chan)
        # self.pool2 = torch.nn.Linear(n_chan, num_nodes)
        # self.pool2 = Prototype_Pooling(num_nodes, n_chan, n_chan, n_chan)
        self.pool2 = Embedded_Pool(num_nodes, n_chan, n_chan, n_chan)

        self.gnn3 = DenseSAGEConv(n_chan, n_chan)

        self.lin1 = torch.nn.Linear(n_chan, n_chan)
        self.lin2 = torch.nn.Linear(n_chan, n_classes)

    def forward(self, x, adj, mask=None):
        B, N, C = x.size()
        mask = mask.view(B, N, 1).to(x.dtype)
        # x = self.gnn1(x, adj*torch.matmul(mask,mask.transpose(-1,-2)), mask)
        x = self.gnn1(x, adj, mask)
        # s = self.pool1(x)
        #
        # x, adj, mc1, o1 = dense_mincut_pool(x, adj, s, mask)

        x, adj,mask, mc1, o1 ,lx_1 = self.pool1(x,adj,mask)

        # x = self.gnn2(x, adj*torch.matmul(mask,mask.transpose(-1,-2)),mask)
        x = self.gnn2(x, adj, mask)
        # s = self.pool2(x)
        #
        # x, adj, mc2, o2 = dense_mincut_pool(x, adj, s)
        x, adj, mask, mc2, o2, lx_2 = self.pool2(x, adj, mask)



        # x = self.gnn3(x, adj*torch.matmul(mask,mask.transpose(-1,-2)),mask)
        x = self.gnn3(x, adj, mask)

        x = x.mean(dim=1)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        loss_c = mc1 + mc2
        loss_o = o1 + o2
        loss_x = lx_1+lx_2
        return F.log_softmax(x, dim=-1), loss_c, loss_o, loss_x




model = Net(dataset.num_features, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)


def train(epoch):
    model.train()
    loss_all = 0
    classify_loss_all = 0
    link_loss_all = 0
    ent_loss_all = 0
    mask_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output, link_loss, ent_loss, loss_x = model(data.x, data.adj, data.mask)
        classify_loss = F.nll_loss(output, data.y.view(-1))
        loss = classify_loss + link_loss + ent_loss +loss_x
        loss.backward()
        classify_loss_all += data.y.size(0) * classify_loss.item()
        link_loss_all += data.y.size(0) * link_loss.item()
        ent_loss_all += data.y.size(0) * ent_loss.item()
        mask_loss += data.y.size(0) * loss_x.item()

        loss_all += data.y.size(0) * loss.item()
        optimizer.step()

    return loss_all / len(train_dataset),classify_loss_all / len(train_dataset),link_loss_all / len(train_dataset), ent_loss_all/ len(train_dataset), mask_loss/ len(train_dataset)


def test(loader):
    model.eval()
    correct = 0

    for data in loader:
        data = data.to(device)
        pred, link_loss, ent_loss, loss_x = model(data.x, data.adj, data.mask)
        loss = F.nll_loss(pred, data.y.view(-1)) + link_loss + ent_loss + loss_x
        correct += pred.max(dim=1)[1].eq(data.y.view(-1)).sum().item()

    return loss, correct / len(loader.dataset)


best_val_acc = test_acc = 0
best_val_loss = 999999999
patience = start_patience = 1000
for epoch in range(1, 15000):
    train_loss,classify_loss,link_loss,ent_loss, mask_loss= train(epoch)
    _, train_acc = test(train_loader)
    val_loss, val_acc = test(val_loader)
    if val_loss < best_val_loss:
        test_loss, test_acc = test(test_loader)
        best_val_acc = val_acc
        patience = start_patience
    else:
        patience -= 1
        if patience == 0:
            break
    print('Epoch: {:03d}, '
          'Train Loss: {:.3f}, classify Loss: {:.3f}, link Loss: {:.3f},ent Loss: {:.3f},mask Loss: {:.3f},   Train Acc: {:.3f}, '
          'Val Loss: {:.3f}, Val Acc: {:.3f}, '
          'Test Loss: {:.3f}, Test Acc: {:.3f}'
          .format(epoch, train_loss,classify_loss,link_loss,ent_loss, mask_loss, train_acc, val_loss, val_acc,
                  test_loss, test_acc))









