
import numpy as np
from copy import copy
import scipy.linalg
from scipy.optimize import minimize

from delight.utils import approx_DL
from delight.photoz_kernels import Photoz_mean_function, Photoz_kernel

log_2_pi = np.log(2*np.pi)


class PhotozGP:
    """
    Photo-z Gaussian process, with physical kernel and mean function.
    """
    def __init__(self,
                 alpha, bandCoefAmplitudes, bandCoefPositions, bandCoefWidths,
                 lines_pos, lines_width, V_C, V_L, alpha_C, alpha_L,
                 redshiftGridGP, use_interpolators=True,
                 lambdaRef=4.5e3, g_AB=1.0):

        DL = approx_DL()
        self.bands = np.arange(bandCoefAmplitudes.shape[0])
        self.mean_fct = Photoz_mean_function(
            alpha, bandCoefAmplitudes, bandCoefPositions, bandCoefWidths,
            g_AB=g_AB, lambdaRef=lambdaRef, DL_z=DL)
        self.kernel = Photoz_kernel(
            bandCoefAmplitudes, bandCoefPositions, bandCoefWidths,
            lines_pos, lines_width, V_C, V_L, alpha_C, alpha_L,
            g_AB=g_AB, DL_z=DL, redshiftGrid=redshiftGridGP,
            use_interpolators=use_interpolators)
        self.redshiftGridGP = redshiftGridGP

    def setData(self, X, Y, Yvar):
        """
        Set data content for the Gaussian process
        """
        self.X = X
        self.Y = Y.reshape((-1, 1))
        self.Yvar = Yvar.reshape((-1, 1))
        self.D = self.Y - self.mean_fct.f(X)
        self.KXX = self.kernel.K(self.X)
        self.A = self.KXX + np.diag(self.Yvar.flatten())
        self.L = scipy.linalg.cholesky(self.A, lower=True)
        self.beta = scipy.linalg.cho_solve((self.L, True), self.D)

    def getCore(self):
        """
        Returns core matrices, useful to re-use the GP elsewhere.
        """
        return self.mean_fct.alpha, self.KXX, self.L, self.D

    def setCore(self, alpha, KXX, L, D):
        """
        Set core matrices
        """
        self.mean_fct.alpha = alpha
        self.KXX = KXX
        self.L = L
        self.D = D.reshape((-1, 1))
        self.beta = scipy.linalg.cho_solve((self.L, True), self.D)

    def margLike(self):
        """
        Returns marginalized likelihood of GP
        """
        logdet = np.log(scipy.linalg.det(self.KXX))
        return\
            0.5 * np.sum(self.beta * self.D) +\
            0.5 * logdet + 0.5 * self.Y.size * log_2_pi

    def predict(self, x_pred):
        """
        Raw way to predict outputs with the GP
        """
        assert x_pred.shape[1] == 3
        KXXp = self.kernel.K(x_pred, self.X)
        KXpXp = self.kernel.K(x_pred)
        y_pred = np.dot(KXXp, self.beta) + self.mean_fct.f(x_pred)
        v = scipy.linalg.cho_solve((self.L, True), KXXp.T)
        y_pred_fullcov = KXpXp - KXXp.dot(v)
        return y_pred, y_pred_fullcov

    def predictAndInterpolate(self, redshiftGrid, ell=1.0, z=None):
        """
        Convenient way to get flux predictions on a redshift/band grid.
        First compute on the coarce GP grid and then interpolate on finer grid.
        ell should be set to reference luminosity used in the GP.
        z is an additional redshift to compute predictions at.
        """
        numBands = self.bands.size
        numZGP = self.redshiftGridGP.size
        redshiftGridGP_loc = 1 * self.redshiftGridGP
        if z is not None:
            zloc = np.abs(z - redshiftGridGP_loc).argmin()
            redshiftGridGP_loc[zloc] = z
        xv, yv = np.meshgrid(redshiftGridGP_loc, self.bands,
                             sparse=False, indexing='xy')
        X_pred = np.ones((numBands*numZGP, 3))
        X_pred[:, 0] = yv.flatten()
        X_pred[:, 1] = xv.flatten()
        X_pred[:, 2] = ell
        y_pred, y_pred_fullcov = self.predict(X_pred)
        model_mean = np.zeros((redshiftGrid.size, numBands))
        model_var = np.zeros((redshiftGrid.size, numBands))
        for i in range(numBands):
            y_pred_bin = y_pred[i*numZGP:(i+1)*numZGP].ravel() / ell
            y_var_bin = np.diag(y_pred_fullcov)[i*numZGP:(i+1)*numZGP] / ell**2
            model_mean[:, i] = np.interp(redshiftGrid,
                                         redshiftGridGP_loc, y_pred_bin)
            model_var[:, i] = np.interp(redshiftGrid,
                                        redshiftGridGP_loc, y_var_bin)
        if z is None:
            return model_mean, model_var
        else:
            return model_mean, model_var, redshiftGridGP_loc

    def optimizeAlpha(self):
        """
        Optimize alpha with marglike as objective.
        """
        x0 = 0.0  # [0.0, self.X[0, 2]]
        res = minimize(self.updateAlphaAndReturnMarglike, x0,
                       method='L-BFGS-B', tol=1e-6,
                       bounds=[(-3e-4, 3e-4)])
        # , (1e-3*self.X[0, 2], 1e3*self.X[0, 2])])
        self.mean_fct.alpha = res.x[0]
        # self.X[:, 2] = res.x[1]

    def updateAlphaAndReturnMarglike(self, alpha):
        """
        For optimizing alpha with the marglike as objective using scipy.
        """
        self.mean_fct.alpha = alpha[0]
        self.D = self.Y - self.mean_fct.f(self.X)
        self.beta = scipy.linalg.cho_solve((self.L, True), self.D)
        return self.margLike()
