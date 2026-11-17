"""Matérn-5/2 Gaussian process with a fitted constant mean."""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

JITTER = 1e-8
SEED = 20260812
PARAMETERS = ('variance_D', 'length_D', 'noise_variance')
BOUNDS = [(-6.907755, 6.907755), (-4.605170, 4.605170), (-9.210340, 1.386294)]


class GaussianProcess:
    def __init__(self, inputs, train, responses):
        self.train = np.asarray(train, dtype=int)
        inputs = np.asarray(inputs, dtype=float)
        responses = np.asarray(responses, dtype=float)
        self.offset = float(responses.mean())
        self.scale = max(float(responses.std()), 1e-8)
        self.target = (responses - self.offset) / self.scale
        self.mean = inputs[self.train].mean(axis=0)
        self.sd = inputs[self.train].std(axis=0)
        self.keep = np.flatnonzero(self.sd > 1e-10)
        self.z = (inputs[:, self.keep] - self.mean[self.keep]) / self.sd[self.keep]
        self.identity = np.eye(len(self.train))
        self.ones = np.ones(len(self.train))
        z_train = self.z[self.train]
        self.train_distances = (z_train[:, None, :] - z_train[None, :, :]) ** 2 / z_train.shape[1]

    def kernel(self, theta, rows, columns, gradient=False, training=False):
        amplitude, length = np.exp(theta[:2])
        squared_distance = self.train_distances if training else (
            (self.z[rows, None, :] - self.z[None, columns, :]) ** 2 / self.z.shape[1]
        )
        radius_squared = 5.0 * (squared_distance / length ** 2).sum(axis=2)
        radius = np.sqrt(np.maximum(radius_squared, 0))
        exponential = np.exp(-radius)
        covariance = amplitude * (1 + radius + radius_squared / 3) * exponential
        if gradient:
            derivatives = [covariance, amplitude * (1 + radius) * exponential * radius_squared / 3,
                           np.exp(theta[2]) * self.identity]
            return covariance, derivatives
        return covariance

    def objective(self, theta):
        covariance, derivatives = self.kernel(theta, self.train, self.train, True, True)
        covariance = covariance + (np.exp(theta[2]) + JITTER) * self.identity
        factor = cho_factor(covariance, lower=True, check_finite=False)
        inverse = cho_solve(factor, self.identity, check_finite=False)
        precision_mean = inverse @ self.ones
        mean_precision = precision_mean.sum()
        intercept = precision_mean @ self.target / mean_precision
        residual = self.target - intercept
        alpha = inverse @ residual
        value = 0.5 * (2 * np.log(np.diag(factor[0])).sum() + np.log(mean_precision)
                       + residual @ alpha + (len(self.train) - 1) * np.log(2 * np.pi))
        weights = inverse - np.outer(precision_mean, precision_mean) / mean_precision - np.outer(alpha, alpha)
        gradient = np.array([0.5 * np.sum(weights * derivative) for derivative in derivatives])
        return float(value), gradient

    def fit(self, parameters=None):
        if parameters is None:
            rng = np.random.default_rng(SEED)
            bounds = np.asarray(BOUNDS)
            starts = [np.array([0., 0., np.log(0.1)])]
            starts.extend(rng.uniform(bounds[:, 0], bounds[:, 1]) for _ in range(5))
            solutions = [minimize(self.objective, start, jac=True, method='L-BFGS-B', bounds=BOUNDS,
                                  options=dict(maxiter=250, ftol=1e-9, gtol=1e-6)) for start in starts]
            self.theta = min(solutions, key=lambda solution: solution.fun).x
        else:
            self.theta = np.array([parameters['log_parameters'][name] for name in PARAMETERS])
        self.noise = float(np.exp(self.theta[2]))
        covariance = self.kernel(self.theta, self.train, self.train, training=True)
        self.factor = cho_factor(covariance + (self.noise + JITTER) * self.identity,
                                 lower=True, check_finite=False)
        self.precision_mean = cho_solve(self.factor, self.ones, check_finite=False)
        self.mean_precision = self.precision_mean.sum()
        self.intercept = self.precision_mean @ self.target / self.mean_precision
        self.alpha = cho_solve(self.factor, self.target - self.intercept, check_finite=False)
        return self

    def predict(self, rows):
        rows = np.asarray(rows, dtype=int)
        cross = self.kernel(self.theta, rows, self.train)
        mean = self.offset + self.scale * (self.intercept + cross @ self.alpha)
        diagonal = np.diag(self.kernel(self.theta, rows, rows))
        solved = cho_solve(self.factor, cross.T, check_finite=False)
        variance = np.maximum(diagonal - np.sum(cross * solved.T, axis=1)
                              + (1 - cross @ self.precision_mean) ** 2 / self.mean_precision, 0)
        return mean, np.sqrt(variance) * self.scale, np.sqrt(variance + self.noise) * self.scale

    def predict_external(self, inputs, uncertainty=True):
        z = (np.asarray(inputs)[:, self.keep] - self.mean[self.keep]) / self.sd[self.keep]
        amplitude, length = np.exp(self.theta[:2])
        radius = np.sqrt(5.) * cdist(z, self.z[self.train]) / np.sqrt(z.shape[1]) / length
        cross = amplitude * (1 + radius + radius * radius / 3) * np.exp(-radius)
        mean = self.offset + self.scale * (self.intercept + cross @ self.alpha)
        if not uncertainty:
            return mean
        solved = cho_solve(self.factor, cross.T, check_finite=False)
        variance = np.maximum(amplitude - np.sum(cross * solved.T, axis=1)
                              + (1 - cross @ self.precision_mean) ** 2 / self.mean_precision, 0)
        return mean, np.sqrt(variance) * self.scale, np.sqrt(variance + self.noise) * self.scale

    def latent_covariance(self, rows):
        rows = np.asarray(rows, dtype=int)
        cross = self.kernel(self.theta, rows, self.train)
        residual = 1 - cross @ self.precision_mean
        covariance = (self.kernel(self.theta, rows, rows)
                      - cross @ cho_solve(self.factor, cross.T, check_finite=False)
                      + np.outer(residual, residual) / self.mean_precision) * self.scale ** 2
        covariance = (covariance + covariance.T) / 2
        np.fill_diagonal(covariance, np.maximum(np.diag(covariance), 0))
        return covariance

    def parameters(self):
        return dict(train_indices=self.train.tolist(),
                    log_parameters=dict(zip(PARAMETERS, self.theta.tolist())))
