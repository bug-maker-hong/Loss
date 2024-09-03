from mindspore.dataset import NumpySlicesDataset
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
from CenterTripletLoss import CenterTripletLoss
from triplet_margin_loss import TripletMarginLoss
from mindspore import Model,LossMonitor
epochs = 3
batch_size = 64
learning_rate = 1e-2

class CenterLoss(nn.Cell):
    def __init__(self):
        super(CenterLoss,self).__init__()

    def construct(self, embeddings, labels):
        # CenterLoss
        y, _ = ops.unique(labels)
        center_loss = 0

        for label in y:

            indices = ops.equal(labels,label).astype(ms.int32)
            indices = [i for i, x in enumerate(indices) if x == 1]
            indices = ms.Tensor(indices,dtype=ms.int32)
            # print(indices)
            center = ops.mean(embeddings.gather(indices,axis=0),axis=0)
            
            center_loss += ops.sum(ops.square(embeddings[indices]-center)) / len(indices)

        return center_loss

def get_data(num):
    """生成数据及对应标签"""
    x = np.random.randn(num,10,10)
    y = np.random.randint(0,6,num)
    data = {'data':x.astype(np.float32),"label":y.astype(np.int32)}
    return data

def create_dataset(num_data, batch_size=16):
    """加载数据集"""
    dataset = NumpySlicesDataset(get_data(num_data))
    dataset = dataset.batch(batch_size)
    return dataset

class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_relu_sequential = nn.SequentialCell(
            nn.Dense(10*10, 16)
        )

    def construct(self, x):
        x = self.flatten(x)
        logits = self.dense_relu_sequential(x)
        return logits

train_dataset = create_dataset(num_data=160)
test_dataset = create_dataset(num_data=40)
network = Network()
loss_fn = CenterLoss()
optimizer = nn.Momentum(network.trainable_params(), learning_rate=learning_rate, momentum=0.9)

# 使用model接口将网络、损失函数和优化器关联起来
model = Model(network, loss_fn, optimizer)
model.train(10, train_dataset, callbacks=[LossMonitor(10)])

# # Define forward function
# def forward_fn(data, label):
#     logits = network(data)
#     loss = loss_fn(logits, label)
#     return loss, logits

# # Get gradient function
# grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

# # Define function of one-step training
# def train_step(data, label):
#     (loss, _), grads = grad_fn(data, label)
#     optimizer(grads)
#     return loss

# def train_loop(model, dataset):
#     size = dataset.get_dataset_size()
#     model.set_train()
#     for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
#         loss = train_step(data, label)

#         if batch % 100 == 0:
#             loss, current = loss.asnumpy(), batch
#             print(f"loss: {loss:>7f}  [{current:>3d}/{size:>3d}]")

# def test_loop(model, dataset, loss_fn):
#     num_batches = dataset.get_dataset_size()
#     model.set_train(False)
#     total, test_loss, correct = 0, 0, 0
#     for data, label in dataset.create_tuple_iterator():
#         pred = model(data)
#         total += len(data)
#         test_loss += loss_fn(pred, label).asnumpy()
#         correct += (pred.argmax(1) == label).asnumpy().sum()
#     test_loss /= num_batches
#     correct /= total
#     print(f"Test: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# for t in range(epochs):
#     print(f"Epoch {t+1}\n-------------------------------")
#     train_loop(network, train_dataset)
#     test_loop(network, test_dataset, loss_fn)
# print("Done!")
