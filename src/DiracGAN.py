import matplotlib.pyplot as plt
from matplotlib import gridspec, colors
import torch
import numpy as np

class DiracGAN:
    def __init__(self, step=0.5):
        self.step = step
        self.thetasPlot, self.psisPlot = torch.meshgrid(torch.linspace(-5, 5, 100), torch.linspace(-5, 5, 100), indexing="ij")
        self.lossesPlot = self.f(self.thetasPlot*self.psisPlot) - self.f(torch.tensor(0.))

        self.thetasQuiver,self. psisQuiver = torch.meshgrid(torch.linspace(-5, 5, 10), torch.linspace(-5, 5, 10), indexing="ij")
        newThetasQuiver, newPsisQuiver = self.UpdateOperator(self.thetasQuiver.reshape(-1,), self.psisQuiver.reshape(-1,))
        self.deltaThetasQuiver = newThetasQuiver.reshape(10, 10) - self.thetasQuiver
        self.deltaPsisQuiver = newPsisQuiver.reshape(10, 10) - self.psisQuiver

        self.eigvals = np.zeros(shape=(1, 2))
        self.jacs    = np.zeros(shape=(1, 2, 2))
        self.ComputeJacobiansAtEquilibrium()

    def f(self, x):
        raise NotImplementedError

    def fPrime(self, x):
        raise NotImplementedError

    def fSec(self, x):
        raise NotImplementedError

    def ComputeJacobiansAtEquilibrium(self):
        pass

    def UpdateOperator(self, thetas, psis):
        '''
        Input:
        - thetas    : torch tensor of shape (?,)
        - psis      : torch tensor of shape (?,)
        
        Output:
        - newThetas : torch tensor of shape (?,)
        - newPsis   : torch tensor of shape (?,)
        '''
        raise NotImplementedError

    def PlotJacobianEigenvalues(self):
        ts = np.linspace(0., 2*np.pi, 300)
        plt.figure(figsize=(8, 8))
        plt.grid()
        plt.scatter(np.real(self.eigvals[:, 0]), np.imag(self.eigvals[:, 0]), label=r"$\kappa_1$", c="b", s=7, alpha=0.1)
        plt.scatter(np.real(self.eigvals[:, 1]), np.imag(self.eigvals[:, 1]), label=r"$\kappa_2$", c="r", s=7, alpha=0.1)
        left, right = plt.xlim()
        bottom, top = plt.ylim()
        plt.plot(np.cos(ts), np.sin(ts), linewidth=1.5, c="k")
        plt.axis('equal')
        plt.xlim(left, right)
        plt.ylim(bottom, top)
        plt.legend(fontsize=12)
        plt.title("Eigenvalues of the Jacobian", fontsize=14)
        plt.show()

    def PlotTrajectory(self, thetasTraj, psisTraj):
        gs = gridspec.GridSpec(nrows=1, ncols=1, height_ratios=[1], width_ratios=[1])
        fig = plt.figure(figsize=(8, 8))
        axTmp = plt.subplot(gs[0, 0])
        cs = axTmp.contourf(self.thetasPlot, self.psisPlot, self.lossesPlot, levels=20, alpha=0.4)
        axTmp.quiver(self.thetasQuiver, self.psisQuiver, self.deltaThetasQuiver, self.deltaPsisQuiver, width=0.004)
        axTmp.scatter(thetasTraj, psisTraj, c=(np.arange(thetasTraj.shape[0])[::-1])**3, marker='x', cmap='Greys', alpha=1., linewidths=0.8)
        axTmp.set_xlabel(r"$\theta$", fontsize=12)
        axTmp.set_ylabel(r"$\psi$", fontsize=12)
        axTmp.set_title("Loss landscape", fontsize=14)
        axTmp.axis('scaled')
        norm = colors.Normalize(vmin=cs.cvalues.min(), vmax=cs.cvalues.max())
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cs.cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ticks=cs.levels[::2], alpha=0.4)
        cbar.set_label('Loss')
        plt.show()

class SimultaneousDGAN(DiracGAN):
    def __init__(self, step):
        super().__init__(step)

    def f(self, x):
        return - torch.log(1 + torch.exp(-x))

    def fPrime(self, x):
        return 1. / (1. + torch.exp(x))

    def fSec(self, x):
        fp = self.fPrime(x)
        return fp * (1. - fp)

    def ComputeJacobiansAtEquilibrium(self):
        nVals = 1000
        step  = np.linspace(0.001, 4., nVals)

        self.jacs = np.ones(shape=(nVals, 2, 2))
        self.jacs[:, 0, 0] = 1.
        self.jacs[:, 0, 1] = -step / 2
        self.jacs[:, 1, 0] = step / 2
        self.jacs[:, 1, 1] = 1.
        self.eigvals = np.linalg.eigvals(self.jacs)

    def GradVecField(self, thetas, psis):
        '''
        Input:
        - thetas    : torch tensor of shape (?,)
        - psis      : torch tensor of shape (?,)
        
        Output:
        - gradVecField : torch tensor of shape (2, ?)
        '''
        assert thetas.shape == psis.shape
        gradVecField = torch.zeros(size=(2, thetas.shape[0]))
        fpThetaPsi = self.fPrime(thetas * psis)
        gradVecField[0] = - psis * fpThetaPsi
        gradVecField[1] = thetas * fpThetaPsi
        return gradVecField

    def UpdateOperator(self, thetas, psis):
        '''
        Input:
        - thetas    : torch tensor of shape (?,)
        - psis      : torch tensor of shape (?,)
        - step      : step size used
        
        Output:
        - newThetas : torch tensor of shape (?,)
        - newPsis   : torch tensor of shape (?,)
        '''
        gradVecField = self.GradVecField(thetas, psis)
        newThetas = thetas + self.step * gradVecField[0]
        newPsis   = psis + self.step * gradVecField[1]
        return newThetas, newPsis

class CLC_SDGAN(SimultaneousDGAN):
    def __init__(self, step, weightReg):
        self.weightReg = weightReg
        super().__init__(step)

    def ComputeJacobiansAtEquilibrium(self):
        nVals = 1000
        weightReg = np.linspace(0.001, 4., nVals)
        step      = 0.8

        self.jacs = np.ones(shape=(nVals, 2, 2))
        self.jacs[:, 0, 0] = 1.
        self.jacs[:, 0, 1] = -step / 2
        self.jacs[:, 1, 0] = step / 2
        self.jacs[:, 1, 1] = (1. - weightReg * step)

        self.eigvals = np.linalg.eigvals(self.jacs)

    def GradVecField(self, thetas, psis):
        '''
        Input:
        - thetas    : torch tensor of shape (?,)
        - psis      : torch tensor of shape (?,)
        
        Output:
        - gradVecField : torch tensor of shape (2, ?)
        '''
        assert thetas.shape == psis.shape
        gradVecField = torch.zeros(size=(2, thetas.shape[0]))
        fpThetaPsi = self.fPrime(thetas * psis)
        gradVecField[0] = - psis * fpThetaPsi
        gradVecField[1] = thetas * fpThetaPsi - self.weightReg * psis
        return gradVecField

class AlternatingDGAN(DiracGAN):
    def __init__(self, step):
        super().__init__(step)

    def f(self, x):
        return - torch.log(1 + torch.exp(-x))

    def fPrime(self, x):
        return 1. / (1. + torch.exp(x))

    def fSec(self, x):
        fp = self.fPrime(x)
        return fp * (1. - fp)

    def ComputeJacobiansAtEquilibrium(self):
        nVals = 1000
        step  = np.linspace(0.001, 4., nVals)

        self.jacs = np.ones(shape=(nVals, 2, 2))
        self.jacs[:, 0, 0] = 1.
        self.jacs[:, 0, 1] = -step / 2
        self.jacs[:, 1, 0] = step / 2
        self.jacs[:, 1, 1] = 1. - step ** 2 / 4
        self.eigvals = np.linalg.eigvals(self.jacs)

    def GradVecFieldGen(self, thetas, psis):
        '''
        Input:
        - thetas    : torch tensor of shape (?,)
        - psis      : torch tensor of shape (?,)
        
        Output:
        - gradVecField : torch tensor of shape (?,)
        '''
        assert thetas.shape == psis.shape
        fpThetaPsi   = self.fPrime(thetas * psis)
        gradVecField = - psis * fpThetaPsi
        return gradVecField

    def GradVecFieldDisc(self, thetas, psis):
        '''
        Input:
        - thetas    : torch tensor of shape (?,)
        - psis      : torch tensor of shape (?,)
        
        Output:
        - gradVecField : torch tensor of shape (?,)
        '''
        assert thetas.shape == psis.shape
        fpThetaPsi   = self.fPrime(thetas * psis)
        gradVecField = thetas * fpThetaPsi
        return gradVecField

    def UpdateOperator(self, thetas, psis):
        '''
        Input:
        - thetas    : torch tensor of shape (?,)
        - psis      : torch tensor of shape (?,)
        
        Output:
        - newThetas : torch tensor of shape (?,)
        - newPsis   : torch tensor of shape (?,)
        '''
        gradVecFieldGen = self.GradVecFieldGen(thetas, psis)
        newThetas = thetas + self.step * gradVecFieldGen
        gradVecFieldDisc = self.GradVecFieldDisc(newThetas, psis)
        newPsis   = psis + self.step * gradVecFieldDisc
        return newThetas, newPsis

class CLC_ADGAN(AlternatingDGAN):
    def __init__(self, step, weightReg):
        self.weightReg = weightReg
        super().__init__(step)

    def ComputeJacobiansAtEquilibrium(self):
        nVals = 1000
        weightReg = np.linspace(0.001, 4., nVals)
        step      = 0.8

        self.jacs = np.ones(shape=(nVals, 2, 2))
        self.jacs[:, 0, 0] = 1.
        self.jacs[:, 0, 1] = -step / 2
        self.jacs[:, 1, 0] = step / 2
        self.jacs[:, 1, 1] = (1. - weightReg * step) - step ** 2 / 4

        self.eigvals = np.linalg.eigvals(self.jacs)

    def GradVecFieldDisc(self, thetas, psis):
        '''
        Input:
        - thetas    : torch tensor of shape (?,)
        - psis      : torch tensor of shape (?,)
        
        Output:
        - gradVecField : torch tensor of shape (?,)
        '''
        assert thetas.shape == psis.shape
        fpThetaPsi   = self.fPrime(thetas * psis)
        gradVecField = thetas * fpThetaPsi - self.weightReg * psis
        return gradVecField