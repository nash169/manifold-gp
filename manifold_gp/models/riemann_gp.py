#!/usr/bin/env python
# encoding: utf-8

import math
import torch
import gpytorch
import numpy as np

from linear_operator import LinearOperator
from typing import Union

from ..operators import SubBlockOperator, SchurComplementOperator


class RiemannGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel, labels=None, vanilla_model=None):
        super().__init__(train_x, train_y, likelihood)

        # This models have constant mean
        self.mean_module = gpytorch.means.ConstantMean()

        # Riemann kernel (add check for it)
        self.covar_module = kernel

        # Store labels in case of semi-supervised scenario
        self.labels = labels

        # Ambient space vanilla model
        if vanilla_model is not None:
            self.vanilla_model = vanilla_model

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    # def eval(self):
    #     # Generate eigenfunctions for kernel evaluation
    #     if hasattr(self.covar_module, 'base_kernel'):
    #         self.covar_module.base_kernel.generate_eigenpairs()
    #     else:
    #         self.covar_module.generate_eigenpairs()

    #     return super().eval()

    def precision(self, noise=True):
        # Supervised / Semisupervised Learning
        if self.labels is not None:
            # opt = SemiSupervisedWrapper(self.labels, self.covar_module.base_kernel.precision() if hasattr(self.covar_module, 'base_kernel') else self.covar_module.precision())
            opt = SchurComplementOperator(self.covar_module.base_kernel.precision() if hasattr(self.covar_module, 'base_kernel') else self.covar_module.precision(), self.labels)
        else:
            opt = self.covar_module.base_kernel.precision() if hasattr(self.covar_module, 'base_kernel') else self.covar_module.precision()

        # Scale output
        if hasattr(self.covar_module, 'outputscale'):
            opt = ScaleWrapper(self.covar_module.outputscale, opt)

        return NoiseWrapper(self.likelihood.noise, opt) if noise else opt

    def manifold_informed_train(self, lr=1e-1, iter=100, decay_step_size=1000, decay_magnitude=1.0, norm_step_size=10, norm_rand_vec=100, verbose=False, save=False):
        self.train()
        self.likelihood.train()

        # Training targets
        y = self.train_targets.unsqueeze(-1)

        # Deactivate optimization mean parameters
        self.mean_module.raw_constant.requires_grad = False

        # Activate optimization of the Laplacian lengthscale
        try:
            self.covar_module.raw_epsilon.requires_grad = True
        except:
            self.covar_module.base_kernel.raw_epsilon.requires_grad = True

        # Normalize Operator
        if hasattr(self.covar_module, 'outputscale'):
            var_norm = self._normalized_variance(num_rand_vec=norm_rand_vec)
            self.covar_module.outputscale /= var_norm

        # Optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-8)

        # Scheduler
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_step_size, gamma=decay_magnitude)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=200,
            threshold=1e-3, threshold_mode='rel', cooldown=0,
            min_lr=0, eps=1e-8, verbose=True
        )

        # Save log
        if save:
            train_log = np.zeros((iter, sum(p.numel() for p in self.parameters())))

        # Iterations
        for i in range(iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()

            # Operator
            opt = self.precision()

            # Loss
            with gpytorch.settings.cg_tolerance(10000):
                loss = 0.5 * sum([torch.dot(y.squeeze(), opt.matmul(y).squeeze()), -opt.inv_quad_logdet(logdet=True)[1], y.shape[0] * math.log(2 * math.pi)])

            # Add log probs of priors on the (functions of) parameters
            loss_ndim = loss.ndim
            for _, module, prior, closure, _ in self.named_priors():
                prior_term = prior.log_prob(closure(module))
                loss.sub_(prior_term.view(*prior_term.shape[:loss_ndim], -1).sum(dim=-1))

            # Gradient
            loss.backward()

            # Print step information
            if verbose:
                print(f"Iter: {i}, Loss: {loss.item():0.3f}, NoiseVar: {self.likelihood.noise.item():0.3f}", end='')  # LR: {scheduler.get_last_lr()[0]:0.3f},
                if hasattr(self.covar_module, 'outputscale'):
                    print(f", SignalVar: {self.covar_module.outputscale.item():0.5f}", end='')
                if hasattr(self.covar_module, 'base_kernel'):
                    print(f", Lengthscale: {self.covar_module.base_kernel.lengthscale.item():0.3f}, Epsilon: {self.covar_module.base_kernel.epsilon.item():0.3f}")
                else:
                    print(f", Lengthscale: {self.covar_module.lengthscale.item():0.3f}, Epsilon: {self.covar_module.epsilon.item():0.3f}")

            # Step
            optimizer.step()

            # # Re-normalize variance
            # if i % norm_step_size == 0 and i and hasattr(self.covar_module, 'outputscale'):
            #     # self.covar_module.outputscale *= var_norm
            #     # var_norm = self._normalized_variance(num_rand_vec=norm_rand_vec)
            #     # self.covar_module.outputscale /= var_norm
            #     var_norm = self._normalized_variance(num_rand_vec=norm_rand_vec)
            #     self.covar_module.outputscale = var_norm.pow(-1)

            # Scheduler
            scheduler.step(loss)
            # scheduler.step()

            if save:
                # Loss
                train_log[i, 0] = loss.item()
                # Noise Variance
                train_log[i, 1] = self.likelihood.noise.item()
                # Lengthscale & Bandwidth
                if hasattr(self.covar_module, 'base_kernel'):
                    train_log[i, 2] = self.covar_module.base_kernel.lengthscale.item()
                    train_log[i, 3] = self.covar_module.base_kernel.epsilon.item()
                else:
                    train_log[i, 2] = self.covar_module.lengthscale.item()
                    train_log[i, 3] = self.covar_module.epsilon.item()
                # Signal Variance
                if hasattr(self.covar_module, 'outputscale'):
                    train_log[i, 4] = self.covar_module.outputscale.item()

        # Activate optimization mean parameters
        self.mean_module.raw_constant.requires_grad = True

        # Deactivate optimization of the Laplacian lengthscale
        try:
            self.covar_module.raw_epsilon.requires_grad = False
        except:
            self.covar_module.base_kernel.raw_epsilon.requires_grad = False

        # Set signal variance for feature-based kernel
        if hasattr(self.covar_module, 'outputscale'):
            # self.covar_module.outputscale *= var_norm
            self.covar_module.outputscale = self._normalized_variance(num_rand_vec=norm_rand_vec, signal_variance=self.covar_module.outputscale)

        # Save Train Log
        if save:
            np.savetxt('train_log.csv', train_log)

    def _normalized_variance(self, num_rand_vec=100, signal_variance=None):
        precision = self.covar_module.base_kernel.precision() if signal_variance is None else ScaleWrapper(signal_variance, self.covar_module.base_kernel.precision())
        num_points = self.covar_module.base_kernel.num_samples
        rand_idx = torch.randint(0, num_points-1, (1, num_rand_vec))
        rand_vec = torch.zeros(num_points, num_rand_vec).scatter_(0, rand_idx, 1.0).to(self.covar_module.base_kernel.memory_device)

        with torch.no_grad(), gpytorch.settings.cg_tolerance(10000): # gpytorch.settings.max_cholesky_size(1):
            norm_var = precision.inv_quad_logdet(inv_quad_rhs=rand_vec, logdet=False)[0]/num_rand_vec

        return norm_var

    def vanilla_train(self, lr=1e-1, iter=100, verbose=False):
        self.train()
        self.likelihood.train()

        # Deactivate optimization mean parameters
        self.mean_module.raw_constant.requires_grad = False

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        for i in range(iter):
            optimizer.zero_grad()
            output = self(self.train_inputs[0])
            loss = -mll(output, self.train_targets)
            loss.backward()

            if verbose:
                print(f"Iter: {i}, Loss: {loss.item():0.3f}, NoiseVar: {self.likelihood.noise.item():0.3f}", end='')  # LR: {scheduler.get_last_lr()[0]:0.3f},
                if hasattr(self.covar_module, 'outputscale'):
                    print(f", SignalVar: {self.covar_module.outputscale.item():0.5f}", end='')
                if hasattr(self.covar_module, 'base_kernel'):
                    print(f", Lengthscale: {self.covar_module.base_kernel.lengthscale.item():0.3f}, Epsilon: {self.covar_module.base_kernel.epsilon.item():0.3f}")
                else:
                    print(f", Lengthscale: {self.covar_module.lengthscale.item():0.3f}, Epsilon: {self.covar_module.epsilon.item():0.3f}")

            optimizer.step()

        # Activate optimization mean parameters
        self.mean_module.raw_constant.requires_grad = True

    # generate the posteriors
    def posterior(self, x, noise=False):
        with torch.no_grad():
            # get manifold posterior
            self.posterior_manifold = self.likelihood(self(x)) if noise else self(x)

            if hasattr(self, "vanilla_model"):
                # this in theory has been already calculate within the features (find a way to optimize)
                if hasattr(self.covar_module, 'base_kernel'):
                    self.scale_vanilla = 1-self.covar_module.base_kernel.bump_function(x)
                else:
                    self.scale_vanilla = 1-self.covar_module.bump_function(x)

                # get vanilla posterior
                self.posterior_vanilla = self.vanilla_model.likelihood(self.vanilla_model(x)) if noise else self.vanilla_model(x)

    # get posterior mean
    def mean(self, method="manifold"):
        if method == "vanilla":
            return self.posterior_vanilla.mean
        elif method == "hybrid":
            return self.posterior_manifold.mean + self.scale_vanilla*self.posterior_vanilla.mean
        else:
            return self.posterior_manifold.mean

    # get posterior standard deviation
    def stddev(self, method="manifold"):
        if method == "vanilla":
            return self.posterior_vanilla.stddev
        elif method == "hybrid":
            return self.posterior_manifold.stddev + self.scale_vanilla*self.posterior_vanilla.stddev
        else:
            return self.posterior_manifold.stddev


class ScaleWrapper(LinearOperator):
    def __init__(self, outputscale, operator):
        super(ScaleWrapper, self).__init__(outputscale, operator)

    def _matmul(self, x):
        # avg_var = self._args[1]._average_variance(num_rand_vec=100)
        # return self._args[1]._matmul(x)/self._args[0]*avg_var
        return self._args[1]._matmul(x)/self._args[0]

    def _size(self):
        return self._args[1]._size()

    def _transpose_nonbatch(self):
        return self


class NoiseWrapper(LinearOperator):
    def __init__(self, noise, operator):
        super(NoiseWrapper, self).__init__(noise, operator)

    def _matmul(self, x):
        return self._args[1]._matmul(x - self._args[0]*self._args[1]._matmul(x - self._args[0]*self._args[1]._matmul(x)))

    def _size(self):
        return self._args[1]._size()

    def _transpose_nonbatch(self):
        return self


class SemiSupervisedWrapper(LinearOperator):
    def __init__(self, indices, operator):
        super(SemiSupervisedWrapper, self).__init__(indices, operator)

    def _matmul(self, x):
        # mask = torch.zeros(self._args[1]._size()[0], dtype=torch.bool).scatter_(0, self._args[0], 1)
        # labeled = self._args[0]
        # unlabeled = torch.masked_select(torch.arange(self._args[1]._size()[0]), ~mask)

        y = torch.zeros(self._args[1]._size()[0], x.shape[1]).to(x.device)
        y[self._args[0], :] = x
        y = self._args[1]._matmul(y)

        Qxx_y = y[self._args[0], :]

        # z = SubBlockOperator(self._args[1], unlabeled, unlabeled).solve(y[unlabeled, :])
        # z = SubBlockOperator(self._args[1], labeled, unlabeled)._matmul(z)

        # y[self._args[0], :] = 0.0
        # v = torch.rand_like(y)
        # # y = self._args[1].solve(y)
        # y = self._args[1].solve(y+v) - self._args[1].solve(v)

        # not_labaled = torch.ones_like(y)
        # not_labaled[self._args[0], :] = 0.0
        # z = y*not_labaled
        # z = self._args[1]._matmul(z)

        # return Q_xx - z[self._args[0], :]
        return Qxx_y  # -z

    def _size(self):
        return torch.Size([self._args[0].shape[0], self._args[0].shape[0]])

    def _transpose_nonbatch(self):
        return self
