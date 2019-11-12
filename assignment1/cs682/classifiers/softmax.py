import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  num_classes = W.shape[1]
  num_train = X.shape[0]

  # Initialize the loss and gradient to zero.
  dW = np.zeros(W.shape)
  loss = 0.0

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  # for each training image
  for i in range(num_train):
    # dot image and weights to get class scores, and subtract maximum class score
    # from all scores so entries are <= 0
    scores = X[i].dot(W)
    scores -= np.max(scores)

    # correct class score for current training image
    correct_class_score = scores[y[i]]

    # compute softmax of all class scores
    scores_exp = np.exp(scores)
    norm_factor = np.sum(scores_exp)
    class_probs = scores_exp / norm_factor

    loss -= np.log(class_probs[y[i]])

    for j in range(num_classes):
      # contribution from all pixels in image as a result of normalization
      dW[:, j] += X[i] * class_probs[j]

      # if the current class is the true class of the current image then the
      # gradient has one more contribution
      if j == y[i]:
        dW[:, j] -= X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # similarly, data loss part of the gradient needs to be averaged over all
  # training samples
  dW /= num_train

  # Add regularization to the loss, and add gradient of regularization to dW
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """

  num_train = X.shape[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  all_scores = X.dot(W)
  max_scores = np.amax(all_scores, axis=1, keepdims=True)
  all_scores -= max_scores

  all_prob_weights = np.exp(all_scores)
  norm_factor = np.sum(all_prob_weights, axis=1, keepdims=True)
  class_probs = all_prob_weights / norm_factor

  # correct scores for all classes, training images. use double slice with
  # index array to get correct class score corresponding to each training
  # image. correct_class_scores looks like [X[0, y[0]], X[1, y[1]], ...]
  y_idx = np.arange(num_train)
  correct_class_scores = all_scores[y_idx, y].reshape(num_train, 1)
  all_losses = - correct_class_scores + np.log(norm_factor)
  loss = np.sum(all_losses) / num_train
  loss += reg * np.sum(W * W)

  M = class_probs
  M[y_idx, y] -= 1
  dW = X.T.dot(M) / num_train + 2 * reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

