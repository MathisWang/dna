from numpy import log
from scipy.special import psi
from utiles import numba_psi
from numba import jit


@jit(nopython=True, fastmath=True)
def optimize_numba(mu, B, Y, G, H, Bb, Bc, i, j):
    dFG1 = -log(B[i][j])*mu[i][j]*H[:, j]/B[i][j]
    dFG2 = -numba_psi(mu[i][j]/B[i][j])*mu[i][j]*H[:, j]/B[i][j]
    dFG3 = numba_psi(Y[i][j] + mu[i][j]/B[i][j])*mu[i][j]*H[:, j]/B[i][j]
    dFG4 = -mu[i][j]*H[:, j]*log(1 + 1/B[i][j])/B[i][j]
    dFG = dFG1 + dFG2 + dFG3 + dFG4

    dFH1 = -log(B[i][j]) * mu[i][j] * G[i, :] / B[i][j]
    dFH2 = -numba_psi(mu[i][j] / B[i][j]) * mu[i][j] * G[i, :] / B[i][j]
    dFH3 = numba_psi(Y[i][j] + mu[i][j] / B[i][j]) * mu[i][j] * G[i, :] / B[i][j]
    dFH4 = -mu[i][j] * G[i, :] * log(1 + 1 / B[i][j]) / B[i][j]
    dFH = dFH1 + dFH2 + dFH3 + dFH4

    dBb1 = mu[i][j]*Bc[:, j]/(B[i][j]**2)*(log(B[i][j])-1)
    dBb2 = numba_psi(mu[i][j]/B[i][j])*mu[i][j]*Bc[:, j]/(B[i][j]**2)
    dBb3 = -numba_psi(Y[i][j]+mu[i][j]/B[i][j])*mu[i][j]*Bc[:, j]/(B[i][j]**2)
    dBb4 = mu[i][j]*Bc[:, j]/B[i][j]*log(1+1/B[i][j]) + (Y[i][j]+mu[i][j]/B[i][j])*Bc[:, j]/((1+B[i][j])*B[i][j])
    dBb = dBb1 + dBb2 + dBb3 + dBb4

    dBc1 = mu[i][j]*Bb[i, :]/(B[i][j]**2)*(log(B[i][j])-1)
    dBc2 = numba_psi(mu[i][j] / B[i][j]) * mu[i][j] * Bb[i, :] / (B[i][j]**2)
    dBc3 = -numba_psi(Y[i][j] + mu[i][j] / B[i][j]) * mu[i][j] * Bb[i, :] / (B[i][j]**2)
    dBc4 = mu[i][j] * Bb[i, :] / (B[i][j]**2) * log(1 + 1 / B[i][j]) + (Y[i][j] + mu[i][j] / B[i][j]) * Bb[i, :] / (B[i][j]*(
                1 + B[i][j]))
    dBc = dBc1 + dBc2 + dBc3 + dBc4

    return dFG, dFH, dBb, dBc



def optimize(mu, B, Y, G, H, Bb, Bc, i, j):
    dFG1 = -log(B[i][j])*mu[i][j]*H[:, j]/B[i][j]
    dFG2 = -psi(mu[i][j]/B[i][j])*mu[i][j]*H[:, j]/B[i][j]
    dFG3 = psi(Y[i][j] + mu[i][j]/B[i][j])*mu[i][j]*H[:, j]/B[i][j]
    dFG4 = -mu[i][j]*H[:, j]*log(1 + 1/B[i][j])/B[i][j]
    dFG = dFG1 + dFG2 + dFG3 + dFG4

    dFH1 = -log(B[i][j]) * mu[i][j] * G[i, :] / B[i][j]
    dFH2 = -psi(mu[i][j] / B[i][j]) * mu[i][j] * G[i, :] / B[i][j]
    dFH3 = psi(Y[i][j] + mu[i][j] / B[i][j]) * mu[i][j] * G[i, :] / B[i][j]
    dFH4 = -mu[i][j] * G[i, :] * log(1 + 1 / B[i][j]) / B[i][j]
    dFH = dFH1 + dFH2 + dFH3 + dFH4

    dBb1 = mu[i][j]*Bc[:, j]/(B[i][j]**2)*(log(B[i][j])-1)
    dBb2 = psi(mu[i][j]/B[i][j])*mu[i][j]*Bc[:, j]/(B[i][j]**2)
    dBb3 = -psi(Y[i][j]+mu[i][j]/B[i][j])*mu[i][j]*Bc[:, j]/(B[i][j]**2)
    dBb4 = mu[i][j]*Bc[:, j]/B[i][j]*log(1+1/B[i][j]) + (Y[i][j]+mu[i][j]/B[i][j])*Bc[:, j]/((1+B[i][j])*B[i][j])
    dBb = dBb1 + dBb2 + dBb3 + dBb4

    dBc1 = mu[i][j]*Bb[i, :]/(B[i][j]**2)*(log(B[i][j])-1)
    dBc2 = psi(mu[i][j] / B[i][j]) * mu[i][j] * Bb[i, :] / (B[i][j]**2)
    dBc3 = -psi(Y[i][j] + mu[i][j] / B[i][j]) * mu[i][j] * Bb[i, :] / (B[i][j]**2)
    dBc4 = mu[i][j] * Bb[i, :] / (B[i][j]**2) * log(1 + 1 / B[i][j]) + (Y[i][j] + mu[i][j] / B[i][j]) * Bb[i, :] / (B[i][j]*(
                1 + B[i][j]))
    dBc = dBc1 + dBc2 + dBc3 + dBc4

    return dFG, dFH, dBb, dBc