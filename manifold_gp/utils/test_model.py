#!/usr/bin/env python
# encoding: utf-8

# import gc
import torch
import gpytorch
import numpy as np


def test_model(model, input, output, noisy_test=False, base_model=None, max_cholesky=800, cg_tolerance=1e-2, cg_iterations=1000):
    with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.max_cholesky_size(max_cholesky), gpytorch.settings.cg_tolerance(cg_tolerance),  gpytorch.settings.max_cg_iterations(cg_iterations):
        model.likelihood.eval()
        model.eval()

        if base_model is not None:
            model.posterior(input, noisy_posterior=noisy_test, base_model=base_model)
        else:
            model.posterior(input, noisy_posterior=noisy_test)

        error = output - model.posterior_mean
        rmse = (error.square().mean()).sqrt()

        inv_quad, logdet = model.posterior_covar.inv_quad_logdet(inv_quad_rhs=error.unsqueeze(-1), logdet=True)
        nll = 0.5 * sum([inv_quad, logdet, error.size(-1) * np.log(2 * np.pi)])/error.size(-1)

        # del model, input, output
        # gc.collect()
        torch.cuda.empty_cache()

        return rmse, nll

# def test_model2(model, input, output, noisy_test=False, max_cholesky=800, cg_tolerance=1e-2, cg_iterations=1000):
#     with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.max_cholesky_size(max_cholesky), gpytorch.settings.cg_tolerance(cg_tolerance),  gpytorch.settings.max_cg_iterations(cg_iterations):
#         posterior = model.likelihood(model(input)) if noisy_test else model(input)

#         error = output - posterior.mean
#         rmse = (error.square().mean()).sqrt()

#         covar = posterior.lazy_covariance_matrix.evaluate_kernel()
#         inv_quad, logdet = covar.inv_quad_logdet(inv_quad_rhs=error.unsqueeze(-1), logdet=True)
#         nll = 0.5 * sum([inv_quad, logdet, error.size(-1) * np.log(2 * np.pi)])/error.size(-1)

#         # del model, input, output
#         # gc.collect()
#         torch.cuda.empty_cache()

#         return rmse, nll
