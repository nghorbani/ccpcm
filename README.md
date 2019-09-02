# Cross-Category Product Choice Model
A PyTorch implementation for the paper 
[Cross-Category Product Choice: A Scalable Deep-Learning Model](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3402471).

The test set error is too much. This could have multiple reasons:
* Overfitting to train data
* Network has failed to learn anything useful from the training data that is applicable to test data
* Train data has nothing to do with the test data

Since we have implemented a network that has been tested in a publication before, we doubt the training data to be inefficient. 
This hypothesis can be tested for example by customer 1. In the test data customer 1 is predicted to purchase item 38 
with probability of 0.91, however his purchase frequency of item 38 from all the T is exactly zero, e.g. he never bought it before.
Since this item is also not discounted we tend to further doubt efficiency of the training data for the current test data.