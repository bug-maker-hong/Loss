import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops


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


if __name__ == '__main__':
    # Set batch size and number of classes
    batch_size = 8
    num_classes = 4

    # Generate random embedding vectors
    embedding_dim = 10
    embeddings = ms.Tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
        [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
        [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
        [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
        [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
        [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
    ], dtype=ms.float32)
    l2normalize = ops.L2Normalize(axis=1)
    embeddings = l2normalize(embeddings)
    # Generate labels
    labels = ms.Tensor([2, 2, 2, 1, 3, 1, 0, 3], dtype=ms.int32)

    loss_fn = CenterLoss()
    loss = loss_fn(embeddings,labels)
    print(loss)