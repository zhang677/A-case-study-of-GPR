import pandas as pd
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import gpytorch
import argparse
matplotlib.use('Agg')


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()+ gpytorch.kernels.MaternKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def std_norm(y_total):
    # y_total: np.array
    y_std = np.std(y_total)
    y_mean = sum(y_total)/len(y_total)
    y_norm = (y_total-y_mean)/y_std
    # y_norm: torch.tensor
    return (y_std,y_mean,y_norm)

def std_denorm(y_std, y_mean, y_norm):
    y_norm = np.array(y_norm)
    return y_norm*y_std + y_mean

def train_test_split(x_norm,y_norm,rate=0.9):
    idx = np.int(np.floor(rate*len(x_norm)))
    print(idx)
    X_train = x_norm[0:idx]
    X_test = x_norm[idx:]
    y_train = y_norm[0:idx]
    y_test = y_norm[idx:]    
    #from sklearn.model_selection import train_test_split
    #X_train, X_test, y_train, y_test = train_test_split(x_norm, y_norm, test_size=rate, random_state=42)
    return (X_train, X_test, y_train, y_test)
def get_mape(y_true, y_pred):
    """
    Compute mean absolute percentage error (MAPE)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description='GPR')
    parser.add_argument("--train", action='store_true', default=False,
                        help="True to train")
    parser.add_argument("--test", action='store_true', default=False,
                        help="True to test")
    parser.add_argument("--val", action='store_true', default=False,
                        help="True to validate")    
    parser.add_argument("--init", action='store_true', default=False,
                        help="True to plot initially")
    parser.add_argument("--inn", type=str, default='', 
                        help="load Model file name appendix")
    parser.add_argument("--out", type=str, default='', 
                        help="output Model file name appendix")    
    parser.add_argument("--lr", type=float, default=0.5, 
                        help="learning rate")
    parser.add_argument("--first", action='store_true', default=False,
                        help="The first time to train")
    parser.add_argument("--all", action='store_true', default=False,
                        help="Plot val and test")
    parser.add_argument("--iter", type = int, default=10,help="Training iterations")
    parser.add_argument("--dataset", type = str, default='IXIC',help="Cooperation's name")

    
    args = parser.parse_args()

    # read the data and init the date

    df = pd.read_csv(args.dataset+'.csv',index_col=0)
    y_total = np.array(df['Adj Close'])
    x_total = np.linspace(0, y_total.shape[0]-1,y_total.shape[0])
    print(y_total.shape)

    # split the train and test

    x_train, x_test, y_train, y_test = train_test_split(x_total, y_total,0.9)

    # normalize

    y_std,y_mean,y_norm = std_norm(y_train)
    x_std,x_mean,x_norm = std_norm(x_train)
    x_test = (x_test-x_mean)/x_std
    y_test = (y_test-y_mean)/y_std

    if args.init:
        # plot the train and test

        f, ax = plt.subplots(1, 1, figsize=(6, 5))
        ax.plot(x_total,y_total,'k')
        plt.xlabel('Days')
        plt.ylabel('Index')
        plt.title(args.dataset)
        plt.savefig('raw'+args.dataset+'.png')
        f, ax = plt.subplots(1, 1, figsize=(6, 5))
        ax.plot(x_norm,y_norm,'k')
        ax.plot(x_test,y_test,'r')
        plt.xlabel('Days')
        plt.ylabel('Index')
        plt.title(args.dataset)
        plt.savefig('train'+args.dataset+'.png')

    # transform to torch tensor

    x_train = torch.tensor(x_norm).float()
    y_train = torch.tensor(y_norm).float()
    x_test = torch.tensor(x_test).float()
    y_test = torch.tensor(y_test).float()

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(x_train, y_train, likelihood)
    if args.train:
        f = open(args.out+'.txt','w')
        f.write(args.dataset+' '+'iter:'+str(args.iter)+';'+'lr:'+str(args.lr)+'\n')
        training_iter = args.iter
        if not args.first:
            state_dict = torch.load('model_state'+args.inn+'.pth')
        model = ExactGPModel(x_train, y_train, likelihood)  # Create a new GP model
        if not args.first:
            model.load_state_dict(state_dict)
        # Find optimal model hyperparameters
        model.train()
        likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.3)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(x_train)
            # Calc loss and backprop gradients
            loss = -mll(output, y_train)
            loss.backward()
#            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
#                i + 1, training_iter, loss.item(),
#                model.covar_module.base_kernel.lengthscale.item(),
#                model.likelihood.noise.item()
#            ))
            print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()),file=f)
            optimizer.step()
        torch.save(model.state_dict(), 'model_state'+args.out+'.pth')

    if args.test:
        state_dict = torch.load('model_state'+args.inn+'.pth')
        model = ExactGPModel(x_train, y_train, likelihood)  # Create a new GP model
        model.load_state_dict(state_dict)

        model.eval()
        likelihood.eval()

        with torch.no_grad():
            y_pred = model(x_test)
            f, ax = plt.subplots(1, 1, figsize=(8, 6))
            # Get upper and lower confidence bounds
            lower, upper = y_pred.confidence_region()
            # Plot training data as black stars
            ax.plot(x_test.numpy(), y_test.numpy(), 'k')
            # Plot predictive means as blue line
            ax.plot(x_test.numpy(), y_pred.mean.numpy(), 'b')
            # Shade between the lower and upper confidence bounds
            ax.fill_between(x_test.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
            ax.legend(['Observed Data', 'Mean', 'Confidence'])
            mape = get_mape(y_test,y_pred.mean)
            plt.title(args.dataset+' mape:%0.3f%%'%mape)
            plt.xlabel('Days')
            plt.ylabel('Index')
            plt.savefig('test'+args.inn+'.png')
            

    if args.val:
        state_dict = torch.load('model_state'+args.inn+'.pth')
        model = ExactGPModel(x_train, y_train, likelihood)  # Create a new GP model
        model.load_state_dict(state_dict)

        model.eval()
        likelihood.eval()

        with torch.no_grad():
            y_pred = model(x_train)
            f, ax = plt.subplots(1, 1, figsize=(8, 6))
            # Get upper and lower confidence bounds
            lower, upper = y_pred.confidence_region()
            # Plot training data as black stars
            ax.plot(x_train.numpy(), y_train.numpy(), 'k')
            # Plot predictive means as blue line
            ax.plot(x_train.numpy(), y_pred.mean.numpy(), 'b')
            # Shade between the lower and upper confidence bounds
            ax.fill_between(x_train.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
            
            #ax.set_ylim([-4, 4])
            mape = get_mape(y_train,y_pred.mean)
            plt.title(args.dataset+' mape:%0.3f%%'%mape)
            ax.legend(['Observed Data', 'Mean', 'Confidence'])
            plt.savefig('val'+args.inn+'.png')
    
    if args.all:
        state_dict = torch.load('model_state'+args.inn+'.pth')
        model = ExactGPModel(x_train, y_train, likelihood)  # Create a new GP model
        model.load_state_dict(state_dict)

        model.eval()
        likelihood.eval()

        with torch.no_grad():
            y_pred = model(x_train)
            f, ax = plt.subplots(1, 1, figsize=(8, 6))
            # Get upper and lower confidence bounds
            lower, upper = y_pred.confidence_region()
            # Plot training data as black stars
            ax.plot(x_train.numpy(), y_train.numpy(), 'k')
            # Plot predictive means as blue line
            ax.plot(x_train.numpy(), y_pred.mean.numpy(), 'b')
            # Shade between the lower and upper confidence bounds
            ax.fill_between(x_train.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
            #ax.set_ylim([-4, 4])
            y_pred = model(x_test)
            lower, upper = y_pred.confidence_region()
            ax.plot(x_test.numpy(), y_test.numpy(), 'r')
            ax.plot(x_test.numpy(), y_pred.mean.numpy(), 'g')
            ax.fill_between(x_test.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
            ax.legend(['Observed Data', 'Mean', 'Test Data', 'Predict Mean','Confidence', 'Predict Confidence'])
            plt.title(args.dataset)
            plt.xlabel('Days')
            plt.ylabel('Index')
            plt.savefig('all'+args.inn+'.png')            



