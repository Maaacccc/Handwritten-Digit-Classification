import numpy as np
import scipy.io as io
import copy
import time


def leastSquaresBasis(x, y, degree):
    # Construct Basis
    Xpoly = polyBasis(x, degree)

    # Solve least squares problem (assumes that Xpoly.T @ Xpoly is invertible)
    w = np.linalg.solve(Xpoly.T @ Xpoly, Xpoly.T @ y)

    # Return a dictionary that represents the model
    model = {'w': w, 'degree': degree, 'predict': predict}
    return model

def predict(model, Xtest):
    Xpoly = polyBasis(Xtest, model['degree'])
    yhat = Xpoly @ model['w']
    return yhat

def polyBasis(x, m):
    n = len(x)
    # Build a Vandermonde matrix in NumPy
    Xpoly = np.vander(x, m+1, increasing=True)
    return Xpoly

def linearInd2Binary(ind, nLabels):
    n = len(ind)
    y = -np.ones((n, nLabels))
    for i in range(n):
        y[i, ind[i] - 1] = 1  # Subtract 1 because MATLAB is 1-indexed and Python is 0-indexed
    return y

def logdet(M, errorDet):
    try:
        # Attempt to compute the Cholesky decomposition of M
        R = np.linalg.cholesky(M)
        # If successful, compute the log of the determinant of M
        l = 2 * np.sum(np.log(np.diag(R)))
    except np.linalg.LinAlgError:
        # If the Cholesky decomposition fails, return the error value
        l = errorDet
    return l

def standardizeCols(M, mu=None, sigma2=None):
    nrows, ncols = M.shape
    M = M.astype(float)
    if mu is None or sigma2 is None:
        mu = np.mean(M, axis=0)
        sigma2 = np.std(M, axis=0, ddof=1)  # ddof=1 for sample standard deviation, like MATLAB
        sigma2[sigma2 < np.finfo(float).eps] = 1  # np.finfo(float).eps is a small number close to machine precision

    S = M - np.tile(mu, (nrows, 1))
    if ncols > 0:
        S = S / np.tile(sigma2, (nrows, 1))
    return S, mu, sigma2

def MLPclassificationLoss(w, X, y, nHidden, nLabels):
    nInstances, nVars = X.shape
    
    # Form Weights
    offset = 0
    inputWeights = w[:offset+nVars*nHidden[0]].reshape(nVars, nHidden[0], order='F')
    offset += nVars*nHidden[0]
    hiddenWeights = []
    for h in range(1, len(nHidden)):
        size = nHidden[h-1] * nHidden[h]
        hiddenWeights.append(w[offset:offset+size].reshape(nHidden[h-1], nHidden[h], order='F'))
        offset += size
    outputWeights = w[offset:offset+nHidden[-1]*nLabels].reshape(nHidden[-1], nLabels, order='F')
    
    f = 0
    gInput  = np.zeros_like(inputWeights)
    gHidden = [np.zeros_like(hw) for hw in hiddenWeights]
    gOutput = np.zeros_like(outputWeights)

    # Compute Output
    for i in range(nInstances):
        ip = []
        fp = []
        
        # Input layer
        ip.append(X[i] @ inputWeights)
        fp.append(np.tanh(ip[-1]))
        # Hidden layers
        for h, hw in enumerate(hiddenWeights):
            ip.append(fp[-1] @ hw)
            fp.append(np.tanh(ip[-1]))

        # Output layer
        yhat = fp[-1] @ outputWeights
        relativeErr = yhat - y[i]
        f += np.sum(relativeErr ** 2)
       
        # Gradients
        err = 2 * relativeErr

        # Output Weights
        for c in range(nLabels):
            gOutput[:, c] += err[c] * fp[-1]
        
        if len(nHidden) > 1:
            # Backpropagation through hidden layers
            backprop = np.zeros((nLabels, nHidden[-1]))
            for c in range(nLabels):
                backprop[c, :] = err[c] * (sech(ip[-1]) ** 2 * outputWeights[:, c])
                gHidden[-1] += np.outer(fp[-2], backprop[c, :])
            backprop = np.sum(backprop, axis=0)
           
            for h in range(len(nHidden)-2, 0, -1):
                backprop = (backprop @ hiddenWeights[h].T) * (sech(ip[h]) ** 2)
                gHidden[h-1] += np.outer(fp[h-1], backprop)
            
            # Input Weights
            backprop = (backprop @ hiddenWeights[0].T) * (sech(ip[0]) ** 2)
            gInput += np.outer(X[i], backprop)
        else:
            # Single hidden layer
            for c in range(nLabels):    
                gInput += err[c] * np.outer(X[i], (sech(ip[-1]) ** 2 * outputWeights[:, c]))
    # Put Gradient into vector
    g = np.concatenate([gInput.T.flatten()] + [hw.T.flatten() for hw in gHidden] + [gOutput.T.flatten()])
    g = g.reshape(-1,1)
    return f, g


def sech(x):
    return 1 / np.cosh(x)


def MLPclassificationPredict(w, X, nHidden, nLabels):
    nInstances, nVars = X.shape
    # Form Weights
    offset = 0
    inputWeights = w[offset:offset + nVars * nHidden[0]].reshape(nVars, nHidden[0], order='F')
    offset += nVars * nHidden[0]
    hiddenWeights = []
    for h in range(1, len(nHidden)):
        current_size = nHidden[h - 1] * nHidden[h]
        hiddenWeights.append(w[offset:offset + current_size].reshape(nHidden[h - 1], nHidden[h], order='F'))
        offset += current_size
    outputWeights = w[offset:offset + nHidden[-1] * nLabels].reshape(nHidden[-1], nLabels, order='F')
    # Compute Output
    y = np.zeros((nInstances, nLabels))
    for i in range(nInstances):
        ip = [None] * len(nHidden)
        fp = [None] * len(nHidden)
        
        # Input layer
        ip[0] = np.dot(X[i, :], inputWeights)
        fp[0] = np.tanh(ip[0])
        # Hidden layers
        for h in range(1, len(nHidden)):
            ip[h] = np.dot(fp[h - 1], hiddenWeights[h - 1])
            fp[h] = np.tanh(ip[h])
        
        # Output layer
        y[i, :] = np.dot(fp[-1], outputWeights)
    # Get predicted labels
    y_pred = np.argmax(y, axis=1) + 1  # Return the index of the max value in each row
    y_pred = y_pred.reshape(-1,1)
    return y_pred

# Load data
data = io.loadmat('digits.mat')
X = data['X']
y = data['y']
Xvalid = data['Xvalid']
Xtest = data['Xtest']
yvalid = data['yvalid']
ytest = data['ytest']

n, d = X.shape
nLabels = np.max(y)
yExpanded = linearInd2Binary(y, nLabels)
t = Xvalid.shape[0]
t2 = Xtest.shape[0]

# Standardize columns and add bias
X, mu, sigma = standardizeCols(X)
X = np.hstack((np.ones((n, 1)), X))
d += 1

Xvalid, _, _ = standardizeCols(Xvalid, mu, sigma)
Xvalid = np.hstack((np.ones((t, 1)), Xvalid))
Xtest, _, _ = standardizeCols(Xtest, mu, sigma)
Xtest = np.hstack((np.ones((t2, 1)), Xtest))

# Choose network structure
nHidden = [64,32,10]

# Count number of parameters and initialize weights
nParams = d * nHidden[0]
for h in range(1, len(nHidden)):
    nParams += nHidden[h-1] * nHidden[h]
nParams += nHidden[-1] * nLabels

w = np.random.randn(nParams, 1)
# Initialize a variable to keep track of the previous weights 'w_prev'
w_prev = copy.deepcopy(w)

# Train with stochastic gradient
maxIter = 100000
# Learning rate and momentum hyperparameters
stepSize = 1e-3
beta_t = 0.9
learning_rate_decay_factor = 0.9
decay_every_n_epochs = 10000

start_time = time.time()

for iter in range(maxIter):
    if iter % decay_every_n_epochs == 0:
        stepSize *= learning_rate_decay_factor  # Decay the learning rate

    if iter % (maxIter // 20) == 0:
        yhat = MLPclassificationPredict(w, Xvalid, nHidden, nLabels)
        print(f'Training iteration = {iter}, validation error = {np.mean(yhat != yvalid)}')
    i = np.random.randint(0, n)
    f, g = MLPclassificationLoss(w, X[i:i+1, :], yExpanded[i:i+1, :], nHidden, nLabels)
    
    # Perform the weight update with momentum
    w_new = w - stepSize * g + beta_t * (w - w_prev)
    # Update w_prev to the current weights for the next iteration
    w_prev = np.copy(w)
    w = w_new

end_time = time.time()
training_time = end_time - start_time
print(f'Training time = {training_time}')
# Evaluate test error
yhat = MLPclassificationPredict(w, Xtest, nHidden, nLabels)
print(f'Test error with final model = {np.mean(yhat != ytest)}')
