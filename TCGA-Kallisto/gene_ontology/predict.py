import torch
import torch.nn as nn
import torch.utils.data as Data
import copy
from dataset import divide_data, GeneExpLabelDataset


# Hyper-parameters
EPOCH = 500
BATCH_SIZE = 64
LR = 0.001          # learning rate
K = 5               # number of epochs * 2 to continue if val loss is not decreasing


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output, dropout=0.2):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feature, n_hidden),  # hidden layer
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_hidden, n_output),   # output layer
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def test_loss(model, dataset, loss_func):
    model.eval()
    t_loss = 0
    loader = Data.DataLoader(dataset=dataset, batch_size=len(dataset), shuffle=True)
    # each data is of BATCH_SIZE samples
    for _, data in enumerate(loader):
        x, y = data
        x, y = x.float(), y.float()
        pred_y = model(x)
        t_loss += loss_func(torch.squeeze(pred_y), y)
    # test_loss /= len(loader.dataset)
    model.train()
    return t_loss


def accuracy(model, dataset):
    """Computes the accuracy for multiple binary predictions"""
    model.eval()
    loader = Data.DataLoader(dataset=dataset, batch_size=len(dataset), shuffle=True)
    for _, data in enumerate(loader):
        x, target = data
        x, target = x.float(), target.float()
        output = torch.squeeze(model(x))
    pred = output >= 0.5
    truth = target >= 0.5
    acc = float(pred.eq(truth).sum()) / float(target.numel())
    return acc


def train_ann():
    net = Net(n_feature=1049, n_hidden=256, n_output=1)
    net.net.apply(init_weights)
    optimizer = torch.optim.SGD(net.parameters(), lr=LR)
    loss_func = torch.nn.MSELoss()

    train, val, test = divide_data('../rnaseq_scaled_all_drug.csv')

    train_data = GeneExpLabelDataset(train)
    val_data = GeneExpLabelDataset(val)
    test_data = GeneExpLabelDataset(test)

    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

    net.train()

    lowest_loss = float("inf")
    counter = 0
    best_model = None
    for epoch in range(EPOCH):
        for step, data in enumerate(train_loader):
            x, y = data
            x = x.float()
            y = y.float()

            pred_y = net(x)

            loss = loss_func(torch.squeeze(pred_y), y)  # mean square error
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            if epoch % 2 == 0 and step % 50 == 0:
                val_loss = test_loss(net, val_data, loss_func)
                if val_loss < lowest_loss:
                    lowest_loss = val_loss
                    counter = 0
                    best_model = copy.deepcopy(net)
                else:
                    counter += 1

                if epoch % 10 == 0:
                    print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())
                    print('Validation set loss: {:.4f}'.format(val_loss))

        if counter > K:
            break
    print('====> Train set accuracy: {:.4f}\n'.format(accuracy(net, train_data)))
    print('====> Test set accuracy: {:.4f}\n'.format(accuracy(net, test_data)))
    return best_model


if __name__ == '__main__':
    print("\nTraining ANN -------------------------------------------------")
    net = train_ann()
