import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0

  # for each training image
  for i in range(num_train):
    # score for each class of current training image
    scores = X[i].dot(W)

    # correct class score for current training image
    correct_class_score = scores[y[i]]

    # for each class
    for j in range(num_classes):
      # if the current class is the true class of the current image then it
      # doesn't contribute to the loss in inner sum (over j != y[i])
      if j == y[i]:
        continue
      # current class is NOT the true class of the current image, so
      # we compute the difference between the score for the current class
      # and image, and the score for the true class of the current image,
      # plus delta
      margin = scores[j] - correct_class_score + 1  # note delta = 1

      # if the difference is > 0
      if margin > 0:
        # total loss for this image goes up
        loss += margin

        # we already know that j != y[i], so we can compute both sums in
        # the same loop without overcounting. outer sum is over all classes
        dW[:, j] += X[i]

        # inner sum is over all classes except the correct one for this image
        dW[:, y[i]] -= X[i]

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
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  num_train = X.shape[0]

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################

  # compute scores for all classes, training images
  all_scores = X.dot(W)

  # correct scores for all classes, training images. use double slice with
  # index array to get correct class score corresponding to each training
  # image. correct_class_scores looks like [X[0, y[0]], X[1, y[1]], ...]
  y_idx = np.arange(num_train)
  correct_class_scores = all_scores[y_idx, y].reshape(num_train, 1)

  # compute vectorized losses by subtracting correct scores from all
  # scores vector-wise
  losses = all_scores - correct_class_scores + 1

  # score of correct class doesn't contribute to loss
  losses[y_idx, y] = 0

  # set negative losses to zero and average over all losses > 0
  margins = np.where(losses > 0, losses, 0)
  loss = np.sum(margins) / num_train

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################0

  # get indexes where margin > 0, i.e. (image, class) pairs with margin > 0
  M = np.where(margins > 0, 1, 0)

  # compute row sum over j != y[i] for each image; this is the factor in the
  # inner sum when computing gradient w.r.t the true class of an image
  incorrect_classes_sum = np.sum(M, axis=1)

  # add sum over incorrect classes to margin matrix
  M[y_idx, y] = -incorrect_classes_sum.T

  # compute gradient
  dW = X.T.dot(M) / num_train + 2 * reg * W

  ############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
