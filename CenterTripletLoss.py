import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from CenterLoss import CenterLoss
from triplet_margin_loss import TripletMarginLoss
# from model.loss.CenterLoss import CenterLoss
# from model.loss.triplet_margin_loss import TripletMarginLoss


class CenterTripletLoss(nn.Cell):
    """
    batch all Strategy
    
    """
    def __init__(self, margin=1.0, alpha=0.5):
        super(CenterTripletLoss, self).__init__()
        self.alpha = alpha
        self.center_loss_fn = CenterLoss()
        self.triplet_loss_fn = TripletMarginLoss(margin=margin)

    def construct(self, embeddings, labels):
        return self.triplet_loss_fn(embeddings,labels) + self.alpha * self.center_loss_fn(embeddings,labels)
  
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
    # labels = ms.Tensor([0, 1, 1, 0], dtype=ms.int32)
    # Set margin value
    margin = 1.0

    # Test triplet loss
    print(f"Testing triplet loss with batch size of {batch_size}:")
    # loss_fn = TripletMarginLoss(margin=margin)
    loss_fn = CenterTripletLoss(margin=0.2, alpha=0.5)
    triplet_loss = loss_fn(embeddings, labels)
    print("CenterTriplet Loss:", triplet_loss)