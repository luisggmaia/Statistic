import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import norm


class Sensor:
    """
    Sensor class for calibration and reading of sensor data.
    """

    def __init__(self):

        self.calib_model = None

    def calibrate(self, q_x, q_y, degree = 1, n_med = 1, max_outliers = 3):

        self.calib_model = CalibModel(q_x, q_y)
        self.calib_model.calibrate(degree = degree, n_med = n_med, max_outliers = max_outliers)

    def read(self, q, alpha = 0.05):

        return self.calib_model.predict(q, alpha = alpha)


class CalibModel:
    """
    Calibration model for sensor data.
    """
    def __init__(self, q_x, q_y):

        self.q_x = q_x
        self.q_y = q_y
        
        self.degree = None
        self.model = None
        self.mask = np.ones_like(q_x.x, dtype = bool)
        self.outliers = np.zeros_like(q_x.x, dtype = bool)

    def calibrate(self, degree = 1, n_med = 1, max_outliers = 3):

        self.degree = degree
        self.weights = n_med/self.q_x.sigma_x**2

        X = np.vander(self.q_x.x, N = self.degree + 1, increasing = True)
        y = self.q_y.y
        weights = self.weights

        while True:

            self.model = sm.WLS(y, X, weights = weights).fit()

            n = len(y)
            delta = norm.ppf(1 - 1/(4*n), 0, 1)

            residuals = y - self.model.predict(X)
            sigma = np.std(residuals)

            outliers = np.abs(residuals) > delta * sigma
            mask = ~outliers
            self.outliers[self.mask] = outliers
            self.mask[self.mask] = mask

            if np.sum(outliers) == 0 or np.sum(self.outliers) > max_outliers:
                break

            X = X[mask]
            y = y[mask]
            weights = weights[mask]
    
    def predict(self, q, alpha = 0.05):

        X = np.vander(q.x, N = self.degree + 1, increasing = True)
        pred = self.model.get_prediction(X)
        
        y = pred.predicted_mean
        ic = pred.conf_int(alpha = alpha)

        sigma_yy = (ic[:, 1] - ic[:, 0])/2
        sigma_xy = q.sigma_x*((np.vander(q.x, N = self.degree, increasing = True) * range(1, self.degree + 2)) @ self.model.params)
        sigma_y = np.sqrt(sigma_xy**2 + sigma_yy**2)

        return (y, sigma_y)

    def plot_residuals(self):

        # later - "sugou" :(

        pass

    def plot_calibration(self, figsize = (10, 6), marker = 'o', markersize = 4, markercolor = 'black', boundscolor = 'royalblue', outlierscolor = 'gray', markeredgewidth = 0.75, markerlabel = 'Data', outlierslabel = 'Outliers', linecolor = 'black', boundslabel = 'Pred. bounds', linewidth = 1, linelabel = 'Fit', xlabel = 'x', ylabel = 'y', title = 'Calibration Plot', num_points = 100, alpha = 0.05):

        x = np.linspace(np.min(self.q_x.x), np.max(self.q_x.x), num_points)
        y, sigma_y = self.predict(x, sigma_x = 0, alpha = alpha)

        fig, ax = plt.subplots(figsize = figsize)

        ax.errorbar(self.q_x.x[self.mask], self.q_y.y[self.mask], xerr = self.q_x.sigma_x[self.mask], yerr = self.q_y.sigma_y[self.mask], marker = marker, ls = '', label = markerlabel, markersize = markersize, markeredgecolor = markercolor, markerfacecolor = markercolor, markeredgewidth = markeredgewidth, ecolor = markercolor)
        ax.errorbar(self.q_x.x[self.outliers], self.q_y.y[self.outliers], xerr = self.q_x.sigma_x[self.outliers], yerr = self.q_y.sigma_y[self.outliers], marker = marker, ls = '', label = outlierslabel, markersize = markersize, markeredgecolor = outlierscolor, markerfacecolor = outlierscolor, markeredgewidth = markeredgewidth, ecolor = outlierscolor)
        ax.plot(x, y, '-', label = linelabel, color = linecolor, linewidth = linewidth)
        ax.plot(x, y + sigma_y, '--', color = boundscolor, linewidth = linewidth, label = boundslabel)
        ax.plot(x, y - sigma_y, '--', color = boundscolor, linewidth = linewidth)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(frameon = False)

        plt.show()
