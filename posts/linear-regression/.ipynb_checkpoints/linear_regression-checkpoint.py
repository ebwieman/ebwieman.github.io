import numpy as np

class LinearRegression():
    
    def __init__(self):
        # initialize LinearRegression class with empty weights and score history
        self.w = np.zeros([])
        self.score_history = []
        
    def pad(self, X):
        # function for adding column of ones to array of data for convenience - taken from blog post assignment
        return np.append(X, np.ones((X.shape[0], 1)), 1)
            
    def fit(self, X, y, method="analytical", alpha=None, max_epochs=None):
        """
        Fits the linear regression model given training data and labels. Can fit the model using either analytical or gradient descent methods. If using gradient descent, also requires a learning rate and max number of epochs.
        
        args:
            X: numpy array of training data with each row representing a training data point and each column representing a feature of the data
            y: one-dimensional numpy array containing true labels for each data point in X
            alpha (optional): learning rate, used when updating the weights
            max_epochs (optional): maximum number of iterations to perform the gradient descent update
            method (default 'analytical'): specifies if weights should be calculated using analytical or gradient method
        
        outputs:
            Doesn't return anything. Updates the LinearRegression class with the latest weights. If using the gradient method, also updates the score_history variable.
        """
        X_ = self.pad(X) # add column of ones to X for convenience
        
        if method=="analytical":
            self.w = np.linalg.inv(X_.T@X_)@X_.T@y # calculates weights analytically - uses function from linear regression lecture
        
        else: # much of this code was repurposed from my logistic regression blog post
            # set initial conditions before first loop
            done = False
            w = 0.5 - np.random.rand(X_.shape[1]) # initialize random weight vector
            epochs = 0
            
            #compute P and q for calculating gradient - only do this once because not dependent on w
            P = X_.T@X_
            q = X_.T@y

            # iterate through until algorithm converges or has run for max epochs
            while not done:
                grad = self.gradient(w, P, q) # calculate gradient
                w -= alpha*grad # perform update
                self.w = w # update weights variable
                score = self.score(X, y) # compute accuracy

                # update score history, increment epochs
                self.score_history.append(score)
                epochs += 1

                # check if gradient is close to zero and terminate if so
                if np.allclose(grad, np.zeros(len(grad))) or epochs == max_epochs:        
                    done = True
                    
    def predict(self, X):
        X_ = self.pad(X) # add columns of 1s to X for convenience
        return X_@self.w # return vector of predictions

    def score(self,X,y):
        # calculate score using coefficient of determination - formula from linear regression lecture, assignment description
        y_bar = y.mean()
        y_hat = self.predict(X)

        num = ((y_hat - y)**2).sum()
        denom = ((y_bar - y)**2).sum()

        return 1 - num/denom
    
    def gradient(self,w,P,q): # formula taken from lecture notes
        return P@w-q # calculate gradient with current weights
    
    