import numpy as np

class Perceptron:
    """
    Creates a Perceptron model and stores its weights and its accuracy history over the course of training. Contains methods to fit provided data, predict the labels of data, and calculate model accuracy.
    """
    
    def __init__(self):
        #initialize the class with empty history and weights
        self.history = []
        self.w = np.zeros(3)
        
    def fit(self, X, y, max_steps):
        """
        Fits the perceptron model given training data and labels.
        
        arguments:
            X: numpy array of training data with each row representing a training data point and each column representing a feature of the data
            y: one-dimensional numpy array containing true labels for each data point in X
            max_steps: maximum number of iterations to perform the perceptron update
        
        outputs:
            Doesn't return anything. Updates the perceptron class with the latest weights and appends the latest accuracy score to the perceptron's history with each update.
        """
        
        #initialize random weight vector
        X_ = np.append(X, np.ones((X.shape[0], 1)), 1)
        w_ = np.random.rand(X_.shape[1])
        
        for k in range(max_steps):
        # perform the perceptron update and log the score in self.history
            i = np.random.randint(0,np.shape(X)[0]) # pick random point
            y_ = 2*y[i]-1 #calculate y tilde
            w_ += 1*((y_*np.dot(w_,X_[i]))<0)*y_*X_[i] #perform update   
            
            #store weights and history in perceptron object
            self.w = w_
            self.history = np.append(self.history, self.score(X,y))
            
            #exit when accuracy reaches 1.0
            if self.score(X, y) == 1.0:
                break
            
    def predict(self, X):
        """
        Predicts the labels of input data (X) using a fit perceptron object.
        
        arguments:
            X: Numpy array of data points. Each row represents one data point, each column represents a data feature.
        
        ouputs:
            y_hat: Numpy array containing the predicted labels for each data point in the provided dataset.
        """
        
        X_ = np.append(X, np.ones((X.shape[0], 1)), 1) #add a row of 1s to the end of the provided X matrix
        y_hat = np.array([]) #initialize array to store predicted labels
        
        y_hat = np.sign(X_@self.w) #predict a label for each data point

        return y_hat #returns vector of shape 1x100 with predicted labels of data
        
    def score(self, X, y):
        """
        Calculates the accuracy of the perceptron model on a dataset.
        
        arguments:
            X: Numpy array of data points. Each row represents one data point, each column represents a data feature.
            y: one-dimensional numpy array containing true labels for each data point in X
        
        outputs:
            The accuracy of the model on the provided dataset - the percentage of data points whose predicted label is the same as the true label.
        """
        y_hat = self.predict(X) #predict labels using predict function
        return np.mean(1*(y_hat==(2*y-1))) #calculate accuracy by comparing the predicted labels to the true labels
        