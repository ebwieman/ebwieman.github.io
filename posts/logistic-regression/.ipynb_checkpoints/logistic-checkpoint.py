import numpy as np

class LogisticRegression():
    
    def __init__(self):
        # initialize LogisticRegression class with empty weights, loss history and score history
        self.w = np.zeros([])
        self.loss_history = []
        self.score_history = []
        
    def pad(self, X):
        # function for adding column of ones to array of data for convenience - taken from blog post assignment
        return np.append(X, np.ones((X.shape[0], 1)), 1)
        
    def fit(self, X, y, alpha, max_epochs):
        """
        Fits the logistic regression model given training data and labels, a learning rate, and max epochs.
        
        args:
            X: numpy array of training data with each row representing a training data point and each column representing a feature of the data
            y: one-dimensional numpy array containing true labels for each data point in X
            alpha: learning rate, used when updating the weights
            max_epochs: maximum number of iterations to perform the gradient descent update
        
        outputs:
            Doesn't return anything. Updates the LogisticRegression class with the latest weights and appends the latest accuracy score and loss to the loss_history and score_history variables.
        """
        
        X_ = self.pad(X) # add column of ones to X for convenience
        
        # set initial conditions before first loop
        done = False
        prev_loss = np.inf
        w = 0.5 - np.random.rand(X_.shape[1]) # initialize random weight vector
        epochs = 0
        
        # iterate through until algorithm converges or has run for max epochs
        while not done:
            grad = self.gradient(w, X_, y) # calculate gradient
            w -= alpha*grad # perform update
            self.w = w
            new_loss = self.loss(X_, y) # compute loss
            score = self.score(X_, y) # compute accuracy
            
            # update class variables, increment epochs
            self.loss_history.append(new_loss)
            self.score_history.append(score)
            epochs += 1
            
            # check if gradient is close to zero and terminate if so
            if np.allclose(grad, np.zeros(len(grad))) or epochs == max_epochs:
            #if np.isclose(new_loss, prev_loss) or epochs == max_epochs:          
                done = True
            
            # set current loss to previous loss prior to next iteration
            else:
                prev_loss = new_loss
                
    def predict(self, X):
        w_ = self.w[:,np.newaxis] # make w a px1 vector prior to calculating dot product - modified from lecture notes
        return np.sign(X@self.w) # return vector of predicted labels
        
    def score(self, X, y):
        y_hat = self.predict(X) # get vector of predicted labels
        return np.mean(1*(y_hat==(2*y-1))) # check predictions against true labels, return accuracy
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z)) # calculate sigmoid function, to be used in logistic loss - taken from lecture notes
    
    def logistic_loss(self, y_hat, y):
        return -y*np.log(self.sigmoid(y_hat)) - (1-y)*np.log(1-self.sigmoid(y_hat)) # function for calculating logistic loss
        
    def loss(self, X, y): # modified from lecture notes
        y_hat = self.predict(X) # get vector of predicted labels
        return self.logistic_loss(y_hat, y).mean() # calculate total loss
    
    def gradient(self, w, X, y): # formula taken from lecture notes
        y_hat = X@w # return dot product of X and w
        y_hat_ = y_hat[:,np.newaxis] # convert to nx1 vector
        y_ = y[:,np.newaxis] # convert y to nx1 vector
        return ((self.sigmoid(y_hat_)-y_)*X).mean(axis = 0) # calculate gradient with current weights
        
    def fit_stochastic(self, X, y, alpha, max_epochs, batch_size):
        """
        Similar to the fit function above, but performs stochastic gradient descent, meaning the data is split into batches and the gradient descent update is performed on each batch in one epoch, generally allowing the algorithm to convery faster.
        
        args:
            X: numpy array of training data with each row representing a training data point and each column representing a feature of the data
            y: one-dimensional numpy array containing true labels for each data point in X
            alpha: learning rate, used when updating the weights
            max_epochs: maximum number of iterations to perform the gradient descent update
            batch_size: the size of the batches to split the data into before performing gradient descent and updating the weights.
        
        outputs:
            Doesn't return anything. Updates the LogisticRegression class with the latest weights and appends the latest accuracy score and loss to the loss_history and score_history variables.
            """
        
        X_ = self.pad(X) # add column of ones to X for convenience
        
        # set initial conditions before first loop
        prev_loss = np.inf
        w = 0.5 - np.random.rand(X_.shape[1]) # initialize random weight vector
        
        # loop for performing update - batch splitting code taken from assignment
        n = X.shape[0]
        for j in np.arange(max_epochs):
            
            # split into batches
            order = np.arange(n)
            np.random.shuffle(order)
            
            # loop through batches and perform update
            for batch in np.array_split(order, n // batch_size + 1):
                x_batch = X_[batch,:]
                y_batch = y[batch]
                grad = self.gradient(w, x_batch, y_batch)
                w -= alpha*grad
                self.w = w
            
            # calculate loss and score after updates have been performed for each batch
            new_loss = self.loss(X_, y) # compute loss
            score = self.score(X_, y)
            
            # update loss and score histories
            self.loss_history.append(new_loss)
            self.score_history.append(score)
            
            # check if gradient is close to zero and terminate if so
            if np.allclose(grad, np.zeros(len(grad))):        
                break
            
            # set current loss to previous loss prior to next iteration
            else:
                prev_loss = new_loss