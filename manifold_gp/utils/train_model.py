#!/usr/bin/env python
# encoding: utf-8

# import gc
import torch
import gpytorch
import math


def vanilla_train(model, optimizer, max_iter=100, max_cholesky=800, tolerance=1e-2, cg_tolerance=1e-2, cg_max_iter=1000, scheduler=None, verbose=False):
    model.train()
    model.likelihood.train()
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    epoch = 0
    prev_loss = 1e6

    while epoch <= max_iter:
        optimizer.zero_grad()
        output = model(model.train_inputs[0])
        with gpytorch.settings.max_cholesky_size(max_cholesky), gpytorch.settings.cg_tolerance(cg_tolerance),  gpytorch.settings.max_cg_iterations(cg_max_iter):
            loss = -mll(output, model.train_targets)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step(loss)

        if verbose:
            msg = [f"Iteration: {epoch}, Loss: {loss.item():0.3f}, Lr: {scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr']}"]
            if hasattr(model, 'likelihood'):
                msg += [f"Noise Variance: {model.likelihood.noise.item():0.3f}"]
            if hasattr(model.covar_module, 'outputscale'):
                msg += [f"Signal Variance: {model.covar_module.outputscale.item():0.3f}"]
            if model.base_kernel.has_lengthscale:
                msg += [f"Lengthscale: {model.base_kernel.lengthscale.item():0.3f}"]
            print(',\t'.join(msg))

        epoch += 1
        if abs(loss.item() - prev_loss) <= tolerance:
            break

    # gc.collect()
    torch.cuda.empty_cache()

    return loss.item()


def manifold_informed_train(model, optimizer, max_iter=100, tolerance=1e-2, update_norm=None, num_rand_vec=100, max_cholesky=800, cg_tolerance=1e-2, cg_max_iter=1000, scheduler=None, verbose=False):
    model.train()
    model.likelihood.train()

    if hasattr(model.covar_module, 'outputscale'):
        with torch.no_grad(), gpytorch.settings.max_cholesky_size(max_cholesky), gpytorch.settings.cg_tolerance(cg_tolerance),  gpytorch.settings.max_cg_iterations(cg_max_iter):
            model.covar_module.outputscale /= model.covar_module.base_kernel.precision()._average_variance(num_rand_vec=num_rand_vec)

    epoch = 0
    prev_loss = 1e6

    while epoch <= max_iter:
        optimizer.zero_grad()
        precision_operator = model.precision()

        # marginal likelihood
        with gpytorch.settings.max_cholesky_size(max_cholesky), gpytorch.settings.cg_tolerance(cg_tolerance),  gpytorch.settings.max_cg_iterations(cg_max_iter):
            loss = 0.5 * sum([torch.dot(model.train_targets, precision_operator.matmul(model.train_targets.view(-1, 1)).squeeze()),
                              -precision_operator.inv_quad_logdet(logdet=True)[1],
                              model.train_targets.shape[0] * math.log(2 * math.pi)])
        # add log probs of priors on the (functions of) parameters
        loss_ndim = loss.ndim
        for _, module, prior, closure, _ in model.named_priors():
            prior_term = prior.log_prob(closure(module))
            loss.sub_(prior_term.view(*prior_term.shape[:loss_ndim], -1).sum(dim=-1))

        if verbose:
            msg = [f"Iteration: {epoch}, Loss: {loss.item():0.3f}, Lr: {scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr']}"]
            if hasattr(model, 'likelihood'):
                msg += [f"Noise Variance: {model.likelihood.noise.item():0.3f}"]
            if hasattr(model.covar_module, 'outputscale'):
                msg += [f"Signal Variance: {model.covar_module.outputscale.item():0.3f}"]
            msg += [f"Lengthscale: {model.base_kernel.lengthscale.item():0.3f}, Graphbandwidth: {model.base_kernel.graphbandwidth.item():0.3f}"]
            print(',\t'.join(msg))

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step(loss)

        epoch += 1
        if abs(loss.item() - prev_loss) <= tolerance:
            break

        if update_norm is not None and epoch % (update_norm+1) == 0:
            print("Update covariance normalization at epoch: ", epoch)
            with torch.no_grad(), gpytorch.settings.max_cholesky_size(max_cholesky), gpytorch.settings.cg_tolerance(cg_tolerance),  gpytorch.settings.max_cg_iterations(cg_max_iter):
                # model.covar_module.outputscale /= model.covar_module.base_kernel.precision()._average_variance(num_rand_vec=num_rand_vec)
                model.covar_module.outputscale = 1/model.covar_module.base_kernel.precision()._average_variance(num_rand_vec=num_rand_vec)

    if hasattr(model.covar_module, 'outputscale'):
        with torch.no_grad(), gpytorch.settings.max_cholesky_size(max_cholesky), gpytorch.settings.cg_tolerance(cg_tolerance),  gpytorch.settings.max_cg_iterations(cg_max_iter):
            model.covar_module.outputscale *= model.covar_module.base_kernel.precision()._average_variance(num_rand_vec=num_rand_vec)

    # gc.collect()
    torch.cuda.empty_cache()

    return loss.item()
