import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

def masked_maximum(data, mask, dim=1):
    """Computes the axis wise maximum over chosen elements."""
    axis_minimums = ops.min(data, dim, keepdims=True)[0]
    masked_maximums = ops.max(ops.mul(data - axis_minimums, mask.astype(ms.float32)), dim, keepdims=True)[0] + axis_minimums
    return masked_maximums

def masked_minimum(data, mask, dim=1):
    """Computes the axis wise minimum over chosen elements."""
    axis_maximums = ops.max(data, dim, keepdims=True)[0]
    masked_minimums = ops.min(ops.mul(data - axis_maximums, mask.astype(ms.float32)), dim, keepdims=True)[0] + axis_maximums
    return masked_minimums

def pairwise_distance(embeddings, squared=True):
    """Computes the pairwise distance matrix with numerical stability."""
    pairwise_distances_squared = ops.sum(embeddings ** 2, dim=1, keepdim=True) + \
                                 ops.sum(embeddings.T ** 2, dim=0, keepdim=True) - \
                                 2.0 * ops.matmul(embeddings, embeddings.T)
    
    pairwise_distances_squared = ops.maximum(pairwise_distances_squared, 0.0)

    error_mask = ops.less_equal(pairwise_distances_squared, 0.0)

    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = ops.sqrt(pairwise_distances_squared 
                                      + error_mask* 1e-16
                                      )
    
    pairwise_distances = ops.mul(
        pairwise_distances,
        ops.logical_not(error_mask),
    )
    
    num_data = embeddings.shape[0]
    mask_offdiagonals = ops.ones((num_data, num_data)) - ops.eye(num_data)
    pairwise_distances = pairwise_distances * mask_offdiagonals

    return pairwise_distances

class TripletMarginLoss(nn.Cell):
    """
    batch all Strategy
    
    """
    def __init__(self, margin=1.0):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin

    def construct(self, embeddings, labels):
        labels = labels.view(-1, 1)
        pdist_matrix = pairwise_distance(embeddings, squared=False)

        adjacency = ops.equal(labels, labels.T)
        adjacency_not = ops.logical_not(adjacency)

        batch_size = labels.shape[0]

        pdist_matrix_tile = ops.tile(pdist_matrix, (batch_size, 1))
        mask = ops.logical_and(
            ops.tile(adjacency_not, (batch_size, 1)),
            ops.greater(pdist_matrix_tile, ops.reshape(ops.transpose(pdist_matrix, (1, 0)), (-1, 1)))
        )

        mask_final = ops.reshape(ops.greater(ops.sum(mask.astype(ms.float32), dim=1, keepdim=True), 0.0), (batch_size, batch_size))
        mask_final = ops.transpose(mask_final, (1, 0))

        negatives_outside = ops.reshape(
            masked_minimum(pdist_matrix_tile, mask), (batch_size, batch_size)
        )
        negatives_outside = ops.transpose(negatives_outside, (1, 0))

        negatives_inside = ops.tile(masked_maximum(pdist_matrix, adjacency_not), (1, batch_size))
        semi_hard_negatives = ops.select(mask_final, negatives_outside, negatives_inside)

        loss_mat = ops.relu(self.margin + pdist_matrix - semi_hard_negatives)

        mask_positives =  adjacency.astype(ms.float32) - ops.eye(batch_size)
        num_positives = ops.sum(mask_positives)

        triplet_loss = ops.sum(loss_mat * mask_positives) / (num_positives + 1e-16)

        return triplet_loss

if __name__ == '__main__':
    # Set batch size and number of classes
    batch_size = 8
    num_classes = 2

    # Generate random embedding vectors
    embedding_dim = 10
    embeddings = ms.Tensor([
        # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        # [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
        # [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
        [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
        [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
        [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
        [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
    ], dtype=ms.float32)
    l2normalize = ops.L2Normalize(axis=1)
    embeddings = l2normalize(embeddings)
    # Generate labels
    # labels = ms.Tensor([2, 2, 2, 1, 3, 1, 0, 3], dtype=ms.int32)
    labels = ms.Tensor([0, 1, 1, 0], dtype=ms.int32)
    # Set margin value
    margin = 1.0

    # Test triplet loss
    print(f"Testing triplet loss with batch size of {batch_size}:")
    loss_fn = TripletMarginLoss(margin=margin)
    triplet_loss = loss_fn(embeddings, labels)
    print("Triplet Loss:", triplet_loss)