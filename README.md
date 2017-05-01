# triclassify
Basic linear classifier for 3-class classification. Used for CS165B Machine 
Learning, Spring 2017 at UCSB.

Computes the centroid for each of the 3 classes, and then calculates a linear 
boundary between every pair of classes. Ties are resolved alphabetically. I.e. 
if a feature vector is equally between A and B, it will be classified into class
A.

Written in python3, with no additional libraries. I may want to revisit this 
project at some point and try implementing the classifier with scikit-learn.

### Included files:
* [`HW2-6data.html`](./HW2-6data.html) has the problem specification and the
 format of the training and testing data sets.
* `testingN.txt` and `trainingN.txt`: corresponding testing and training data 
sets.
* See [`output.txt`](./output.txt) for example output and usage. 
