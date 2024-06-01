import torch
import time
from math import pi
import numpy as np
from os.path import join

from ..utils.tensorboard import Tensorboard
from ..utils.output import progress
from .convergence import Convergence
from ..model.deepmod import DeepMoD
from typing import Optional


def train(model: DeepMoD,
          data: torch.Tensor,
          target: torch.Tensor,
          optimizer,
          sparsity_scheduler,
          test = 'mse',
          split: float = 0.8,
          log_dir: Optional[str] = None,
          max_iterations: int = 10000,
          write_iterations: int = 25,
          **convergence_kwargs) -> None:
    """[summary]

    Args:
        model (DeepMoD): [description]
        data (torch.Tensor): [description]
        target (torch.Tensor): [description]
        optimizer ([type]): [description]
        sparsity_scheduler ([type]): [description]
        log_dir (Optional[str], optional): [description]. Defaults to None.
        max_iterations (int, optional): [description]. Defaults to 10000.
    """
    start_time = time.time()
    board = Tensorboard(log_dir)  # initializing tb board

    # Splitting data, assumes data is already randomized
    n_train = int(split * data.shape[0])
    n_test = data.shape[0] - n_train
    data_train, data_test = torch.split(data, [n_train, n_test], dim=0)
    target_train, target_test = torch.split(target, [n_train, n_test], dim=0)
    
    # Training
    convergence = Convergence(**convergence_kwargs)
    print('| Iteration | Progress | Time remaining |     Loss |      MSE |      Reg |    L1 norm |')
    for iteration in np.arange(0, max_iterations + 1):
        # ================== Training Model ============================
        prediction, time_derivs, thetas = model(data_train)

        MSE = torch.mean((prediction - target_train)**2, dim=0)  # loss per output
        Reg = torch.stack([torch.mean((dt - theta @ coeff_vector)**2)
                           for dt, theta, coeff_vector in zip(time_derivs, thetas, model.constraint_coeffs(scaled=False, sparse=True))])
        loss = torch.sum(MSE + Reg) 

        # Optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        if iteration % write_iterations == 0:
            # ================== Validation costs ================
            prediction_test, coordinates = model.func_approx(data_test)
            time_derivs_test, thetas_test = model.library((prediction_test, coordinates))
            with torch.no_grad():
                MSE_test = torch.mean((prediction_test - target_test)**2, dim=0)  # loss per output
                Reg_test = torch.stack([torch.mean((dt - theta @ coeff_vector)**2)
                           for dt, theta, coeff_vector in zip(time_derivs_test, thetas_test, model.constraint_coeffs(scaled=False, sparse=True))])
                loss_test = torch.sum(MSE_test + Reg_test) 
            
            # ====================== Logging =======================
            _ = model.sparse_estimator(thetas, time_derivs) # calculating l1 adjusted coeffs but not setting mask
            estimator_coeff_vectors = model.estimator_coeffs()
            l1_norm = torch.sum(torch.abs(torch.cat(model.constraint_coeffs(sparse=True, scaled=True), dim=1)), dim=0)
            progress(iteration, start_time, max_iterations, loss.item(),
                     torch.sum(MSE).item(), torch.sum(Reg).item(), torch.sum(l1_norm).item())
            board.write(iteration, loss, MSE, Reg, l1_norm, model.constraint_coeffs(sparse=True, scaled=True), model.constraint_coeffs(sparse=True, scaled=False), estimator_coeff_vectors, MSE_test=MSE_test, Reg_test=Reg_test, loss_test=loss_test)
            
            # ================== Sparsity update =============
            # Updating sparsity and or convergence
            #sparsity_scheduler(iteration, l1_norm)
            if iteration % write_iterations == 0:
                if test == 'mse':
                    sparsity_scheduler(iteration, torch.sum(MSE_test), model, optimizer)
                else:
                    sparsity_scheduler(iteration, loss_test, model, optimizer) 
                    
                if sparsity_scheduler.apply_sparsity is True:
                    with torch.no_grad():
                        model.constraint.sparsity_masks = model.sparse_estimator(thetas, time_derivs)
                        sparsity_scheduler.reset()

            # ================= Checking convergence
            convergence(iteration, torch.sum(l1_norm))
            if convergence.converged is True:
                print('Algorithm converged. Stopping training.')
                break
    board.close()
    if log_dir is None: 
        path = 'model.pt'
    else:
        path = join(log_dir, 'model.pt')
    torch.save(model.state_dict(), path)

def train_multitask(model: DeepMoD,
          data: torch.Tensor,
          target: torch.Tensor,
          optimizer,
          sparsity_scheduler,
          test = 'mse',
          split: float = 0.8,
          log_dir: Optional[str] = None,
          max_iterations: int = 10000,
          write_iterations: int = 25,
          **convergence_kwargs) -> None:
    """[summary]

    Args:
        model (DeepMoD): [description]
        data (torch.Tensor): [description]
        target (torch.Tensor): [description]
        optimizer ([type]): [description]
        sparsity_scheduler ([type]): [description]
        log_dir (Optional[str], optional): [description]. Defaults to None.
        max_iterations (int, optional): [description]. Defaults to 10000.
    """
    start_time = time.time()
    board = Tensorboard(log_dir)  # initializing tb board

    # Splitting data, assumes data is already randomized
    n_train = int(split * data.shape[0])
    n_test = data.shape[0] - n_train
    data_train, data_test = torch.split(data, [n_train, n_test], dim=0)
    target_train, target_test = torch.split(target, [n_train, n_test], dim=0)
    
    # Training
    convergence = Convergence(**convergence_kwargs)
    print('| Iteration | Progress | Time remaining |     Loss |      MSE |      Reg |    L1 norm |')
    for iteration in np.arange(0, max_iterations + 1):
        # ================== Training Model ============================
        prediction, time_derivs, thetas = model(data_train)

        MSE = torch.mean((prediction - target_train)**2, dim=0)  # loss per output
        Reg = torch.stack([torch.mean((dt - theta @ coeff_vector)**2)
                           for dt, theta, coeff_vector in zip(time_derivs, thetas, model.constraint_coeffs(scaled=False, sparse=True))])
        loss = torch.sum(torch.exp(-model.s[:, 0]) * MSE + torch.exp(-model.s[:, 1]) * Reg + torch.sum(model.s)) 

        # Optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        if iteration % write_iterations == 0:
            # ================== Validation costs ================
            prediction_test, coordinates = model.func_approx(data_test)
            time_derivs_test, thetas_test = model.library((prediction_test, coordinates))
            with torch.no_grad():
                MSE_test = torch.mean((prediction_test - target_test)**2, dim=0)  # loss per output
                Reg_test = torch.stack([torch.mean((dt - theta @ coeff_vector)**2)
                           for dt, theta, coeff_vector in zip(time_derivs_test, thetas_test, model.constraint_coeffs(scaled=False, sparse=True))])
                loss_test = torch.sum(torch.exp(-model.s[:, 0]) * MSE_test + torch.exp(-model.s[:, 1]) * Reg_test + torch.sum(model.s)) 
            
            # ====================== Logging =======================
            _ = model.sparse_estimator(thetas, time_derivs) # calculating l1 adjusted coeffs but not setting mask
            estimator_coeff_vectors = model.estimator_coeffs()
            l1_norm = torch.sum(torch.abs(torch.cat(model.constraint_coeffs(sparse=True, scaled=True), dim=1)), dim=0)
            progress(iteration, start_time, max_iterations, loss.item(),
                     torch.sum(MSE).item(), torch.sum(Reg).item(), torch.sum(l1_norm).item())
            board.write(iteration, loss, MSE, Reg, l1_norm, model.constraint_coeffs(sparse=True, scaled=True), model.constraint_coeffs(sparse=True, scaled=False), estimator_coeff_vectors, MSE_test=MSE_test, Reg_test=Reg_test, loss_test=loss_test, s=model.s)
            
            # ================== Sparsity update =============
            # Updating sparsity and or convergence
            #sparsity_scheduler(iteration, l1_norm)
            if iteration % write_iterations == 0:
                if test == 'mse':
                    sparsity_scheduler(iteration, torch.sum(MSE_test), model, optimizer)
                else:
                    sparsity_scheduler(iteration, loss_test, model, optimizer) 
                    
                if sparsity_scheduler.apply_sparsity is True:
                    with torch.no_grad():
                        model.constraint.sparsity_masks = model.sparse_estimator(thetas, time_derivs)
                        sparsity_scheduler.reset()

            # ================= Checking convergence
            convergence(iteration, torch.sum(l1_norm))
            if convergence.converged is True:
                print('Algorithm converged. Stopping training.')
                break
    board.close()
    if log_dir is None: 
        path = 'model.pt'
    else:
        path = join(log_dir, 'model.pt')
    torch.save(model.state_dict(), path)


def train_multitask_capped(model: DeepMoD,
          data: torch.Tensor,
          target: torch.Tensor,
          optimizer,
          sparsity_scheduler,
          test = 'mse',
          split: float = 0.8,
          log_dir: Optional[str] = None,
          max_iterations: int = 10000,
          write_iterations: int = 25,
          **convergence_kwargs) -> None:
    """[summary]

    Args:
        model (DeepMoD): [description]
        data (torch.Tensor): [description]
        target (torch.Tensor): [description]
        optimizer ([type]): [description]
        sparsity_scheduler ([type]): [description]
        log_dir (Optional[str], optional): [description]. Defaults to None.
        max_iterations (int, optional): [description]. Defaults to 10000.
    """
    start_time = time.time()
    board = Tensorboard(log_dir)  # initializing tb board

    # Splitting data, assumes data is already randomized
    n_train = int(split * data.shape[0])
    n_test = data.shape[0] - n_train
    data_train, data_test = torch.split(data, [n_train, n_test], dim=0)
    target_train, target_test = torch.split(target, [n_train, n_test], dim=0)
    
    #cutoff = torch.full((model.func_approx.architecture[-1], 1), 1e-5).to(target.device)
    cutoff = torch.tensor(15.).to(target.device)

    # Training
    convergence = Convergence(**convergence_kwargs)
    print('| Iteration | Progress | Time remaining |     Loss |      MSE |      Reg |    L1 norm |')
    for iteration in np.arange(0, max_iterations + 1):
        # ================== Training Model ============================
        prediction, time_derivs, thetas = model(data_train)

        MSE = torch.mean((prediction - target_train)**2, dim=0)  # loss per output
        Reg = torch.stack([torch.mean((dt - theta @ coeff_vector)**2, dim=0)
                           for dt, theta, coeff_vector in zip(time_derivs, thetas, model.constraint_coeffs(scaled=False, sparse=True))])

        s_capped = torch.min(torch.max(model.s, -cutoff), cutoff)
        loss = torch.sum(torch.exp(-s_capped[:, 0]) * MSE + torch.exp(-s_capped[:, 1]) * Reg + torch.sum(s_capped)) 

        # Optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        if iteration % write_iterations == 0:
            # ================== Validation costs ================
            prediction_test, coordinates = model.func_approx(data_test)
            time_derivs_test, thetas_test = model.library((prediction_test, coordinates))
            with torch.no_grad():
                MSE_test = torch.mean((prediction_test - target_test)**2, dim=0)  # loss per output
                Reg_test = torch.stack([torch.mean((dt - theta @ coeff_vector)**2)
                           for dt, theta, coeff_vector in zip(time_derivs_test, thetas_test, model.constraint_coeffs(scaled=False, sparse=True))])
                loss_test = torch.sum(torch.exp(-s_capped[:, 0]) * MSE_test + torch.exp(-s_capped[:, 1]) * Reg_test + torch.sum(s_capped)) 
            
            # ====================== Logging =======================
            _ = model.sparse_estimator(thetas, time_derivs) # calculating l1 adjusted coeffs but not setting mask
            estimator_coeff_vectors = model.estimator_coeffs()
            l1_norm = torch.sum(torch.abs(torch.cat(model.constraint_coeffs(sparse=True, scaled=True), dim=1)), dim=0)
            progress(iteration, start_time, max_iterations, loss.item(),
                     torch.sum(MSE).item(), torch.sum(Reg).item(), torch.sum(l1_norm).item())
            board.write(iteration, loss, MSE, Reg, l1_norm, model.constraint_coeffs(sparse=True, scaled=True), model.constraint_coeffs(sparse=True, scaled=False), estimator_coeff_vectors, MSE_test=MSE_test, Reg_test=Reg_test, loss_test=loss_test, s=model.s)
            
            # ================== Sparsity update =============
            # Updating sparsity and or convergence
            #sparsity_scheduler(iteration, l1_norm)
            if iteration % write_iterations == 0:
                if test == 'mse':
                    sparsity_scheduler(iteration, torch.sum(MSE_test), model, optimizer)
                else:
                    sparsity_scheduler(iteration, loss_test, model, optimizer) 
                    
                if sparsity_scheduler.apply_sparsity is True:
                    with torch.no_grad():
                        model.constraint.sparsity_masks = model.sparse_estimator(thetas, time_derivs)
                        sparsity_scheduler.reset()

            # ================= Checking convergence
            convergence(iteration, torch.sum(l1_norm))
            if convergence.converged is True:
                print('Algorithm converged. Stopping training.')
                break
    board.close()
    if log_dir is None: 
        path = 'model.pt'
    else:
        path = join(log_dir, 'model.pt')
    torch.save(model.state_dict(), path)

def train_gradnorm(model: DeepMoD,
          data: torch.Tensor,
          target: torch.Tensor,
          optimizer,
          sparsity_scheduler,
          alpha,
          test = 'mse',
          split: float = 0.8,
          log_dir: Optional[str] = None,
          max_iterations: int = 10000,
          write_iterations: int = 25,
          **convergence_kwargs) -> None:
    """[summary]

    Args:
        model (DeepMoD): [description]
        data (torch.Tensor): [description]
        target (torch.Tensor): [description]
        optimizer ([type]): [description]
        sparsity_scheduler ([type]): [description]
        log_dir (Optional[str], optional): [description]. Defaults to None.
        max_iterations (int, optional): [description]. Defaults to 10000.
    """
    start_time = time.time()
    board = Tensorboard(log_dir)  # initializing tb board

    # Splitting data, assumes data is already randomized
    n_train = int(split * data.shape[0])
    n_test = data.shape[0] - n_train
    data_train, data_test = torch.split(data, [n_train, n_test], dim=0)
    target_train, target_test = torch.split(target, [n_train, n_test], dim=0)

    # Training
    convergence = Convergence(**convergence_kwargs)
    print('| Iteration | Progress | Time remaining |     Loss |      MSE |      Reg |    L1 norm |')
    for iteration in np.arange(0, max_iterations + 1):
        # ================== Training Model ============================
        prediction, time_derivs, thetas = model(data_train)

        MSE = torch.mean((prediction - target_train)**2, dim=0)  # loss per output
        Reg = torch.cat([torch.mean((dt - theta @ coeff_vector)**2, dim=0)
                           for dt, theta, coeff_vector in zip(time_derivs, thetas, model.constraint_coeffs(scaled=False, sparse=True))])
        task_loss = (torch.exp(model.weights) * torch.stack((MSE, Reg), axis=1)).flatten() # weighted losses
        loss = torch.sum(task_loss)

        if iteration == 0: # Getting initial loss
            ini_loss = task_loss.data
        if torch.any(task_loss.data > ini_loss):
            ini_loss[task_loss.data > ini_loss] = task_loss.data[task_loss.data > ini_loss]

        # Getting original grads
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        model.weights.grad.data = model.weights.grad.data * 0.0 # setting weight grads to zero

        # Getting Grads to normalize
        G = torch.tensor([torch.norm(torch.autograd.grad(loss_i, list(model.parameters())[-2], retain_graph=True, create_graph=True)[0], 2) for loss_i in task_loss]).to(data.device)
        G_mean = torch.mean(G)  

        # Calculating relative losses
        rel_loss = task_loss / ini_loss
        inv_train_rate = rel_loss / torch.mean(rel_loss)

        # Calculating grad norm loss
        grad_norm_loss = torch.sum(torch.abs(G - G_mean * inv_train_rate ** alpha))

        # Setting grads
        model.weights.grad = torch.autograd.grad(grad_norm_loss, model.weights)[0]
    
        # do a step with the optimizer
        optimizer.step()
        
        # renormalize
        normalize_coeff = task_loss.shape[0] / torch.sum(model.weights)
        model.weights.data = torch.log(torch.exp(model.weights.data) * normalize_coeff)
        
        if iteration % write_iterations == 0:
            # ================== Validation costs ================
            prediction_test, coordinates = model.func_approx(data_test)
            time_derivs_test, thetas_test = model.library((prediction_test, coordinates))
            with torch.no_grad():
                MSE_test = torch.mean((prediction_test - target_test)**2, dim=0)  # loss per output
                Reg_test = torch.stack([torch.mean((dt - theta @ coeff_vector)**2)
                           for dt, theta, coeff_vector in zip(time_derivs_test, thetas_test, model.constraint_coeffs(scaled=False, sparse=True))])
                loss_test = model.weights @ torch.stack((MSE, Reg), axis=0)
            
            # ====================== Logging =======================
            _ = model.sparse_estimator(thetas, time_derivs) # calculating l1 adjusted coeffs but not setting mask
            estimator_coeff_vectors = model.estimator_coeffs()
            l1_norm = torch.sum(torch.abs(torch.cat(model.constraint_coeffs(sparse=True, scaled=True), dim=1)), dim=0)
            progress(iteration, start_time, max_iterations, loss.item(),
                     torch.sum(MSE).item(), torch.sum(Reg).item(), torch.sum(l1_norm).item())
            board.write(iteration, loss, MSE, Reg, l1_norm, model.constraint_coeffs(sparse=True, scaled=True), model.constraint_coeffs(sparse=True, scaled=False), estimator_coeff_vectors, MSE_test=MSE_test, Reg_test=Reg_test, loss_test=loss_test, w=model.weights)
            
            # ================== Sparsity update =============
            # Updating sparsity and or convergence
            #sparsity_scheduler(iteration, l1_norm)
            if iteration % write_iterations == 0:
                if test == 'mse':
                    sparsity_scheduler(iteration, torch.sum(MSE_test), model, optimizer)
                else:
                    sparsity_scheduler(iteration, loss_test, model, optimizer) 
                    
                if sparsity_scheduler.apply_sparsity is True:
                    with torch.no_grad():
                        model.constraint.sparsity_masks = model.sparse_estimator(thetas, time_derivs)
                        sparsity_scheduler.reset()

            # ================= Checking convergence
            convergence(iteration, torch.sum(l1_norm))
            if convergence.converged is True:
                print('Algorithm converged. Stopping training.')
                break
    board.close()
    if log_dir is None: 
        path = 'model.pt'
    else:
        path = join(log_dir, 'model.pt')
    torch.save(model.state_dict(), path)


def train_SBL(model: DeepMoD,
          data: torch.Tensor,
          target: torch.Tensor,
          optimizer,
          extra_params, 
          sparsity_scheduler,
          split = 0.8,
          exp_ID: str = None,
          log_dir: str = None,
          max_iterations: int = 10000,
          write_iterations: int = 25,
          **convergence_kwargs) -> None:
    """Trains the DeepMoD model. This function automatically splits the data set in a train and test set. 

    Args:
        model (DeepMoD):  A DeepMoD object.
        data (torch.Tensor):  Tensor of shape (n_samples x (n_spatial + 1)) containing the coordinates, first column should be the time coordinate.
        target (torch.Tensor): Tensor of shape (n_samples x n_features) containing the target data.
        optimizer ([type]):  Pytorch optimizer.
        sparsity_scheduler ([type]):  Decides when to update the sparsity mask.
        split (float, optional):  Fraction of the train set, by default 0.8.
        exp_ID (str, optional): Unique ID to identify tensorboard file. Not used if log_dir is given, see pytorch documentation.
        log_dir (str, optional): Directory where tensorboard file is written, by default None.
        max_iterations (int, optional): [description]. Max number of epochs , by default 10000.
        write_iterations (int, optional): [description]. Sets how often data is written to tensorboard and checks train loss , by default 25.
    """
    logger = Logger(exp_ID, log_dir)
    sparsity_scheduler.path = logger.log_dir # write checkpoint to same folder as tb output.
    
    t, a, l = extra_params
    
    # Splitting data, assumes data is already randomized
    n_train = int(split * data.shape[0])
    n_test = data.shape[0] - n_train
    data_train, data_test = torch.split(data, [n_train, n_test], dim=0)
    target_train, target_test = torch.split(target, [n_train, n_test], dim=0)
    
    M = 12
    N = data_train.shape[0]
    threshold = 1e4
    # Training
    convergence = Convergence(**convergence_kwargs)
    for iteration in torch.arange(0, max_iterations):
        # ================== Training Model ============================
        prediction, time_derivs, thetas = model(data_train)
        
        tau_ = torch.exp(t)
        alpha_ = torch.min(torch.exp(a), torch.tensor(1e8, dtype=torch.float32))
        lambda_ = torch.min(torch.exp(l), torch.tensor(2e4, dtype=torch.float32))
                            
        y = time_derivs[0]
        X = thetas[0] / torch.norm(thetas[0], dim=0, keepdim=True)
        
        p_MSE = N / 2 * (tau_ * torch.mean((prediction - target_train)**2, dim=0) - t + np.log(2*np.pi))
        
        A = torch.diag(lambda_) + alpha_ * X.T @ X
        mn = (lambda_ < threshold)[:, None] * (alpha_ * torch.inverse(A) @ X.T @ y)
        E = alpha_ * torch.sum((y - X @ mn)**2) + mn.T @ torch.diag(lambda_) @ mn
        p_reg = 1/2 * (E + torch.sum(torch.log(torch.diag(A)[lambda_ < threshold])) - (torch.sum(l[lambda_ < threshold]) + N * a) - N * np.log(2*np.pi))

        MSE = torch.mean((prediction - target_train)**2, dim=0)  # loss per output
        Reg = torch.stack([torch.mean((dt - theta @ coeff_vector)**2)
                           for dt, theta, coeff_vector in zip(time_derivs, thetas, model.constraint_coeffs(scaled=False, sparse=True))])
        loss = torch.sum(p_MSE + p_reg)

        # Optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if iteration % write_iterations == 0:
            # ================== Validation costs ================
            with torch.no_grad():
                prediction_test = model.func_approx(data_test)[0]
                MSE_test = torch.mean((prediction_test - target_test)**2, dim=0)  # loss per output
         
            # ====================== Logging =======================
            _ = model.sparse_estimator(thetas, time_derivs) # calculating estimator coeffs but not setting mask
            logger(iteration, 
                   loss, MSE, Reg,
                   model.constraint_coeffs(sparse=True, scaled=True), 
                   model.constraint_coeffs(sparse=True, scaled=False),
                   model.estimator_coeffs(),
                   MSE_test=MSE_test,
                   p_MSE = p_MSE,
                   p_reg = p_reg,
                   tau = tau_,
                   alpha=alpha_,
                  lambda_=lambda_,
                   mn=mn)

            # ================== Sparsity update =============
            # Updating sparsity 
            update_sparsity = sparsity_scheduler(iteration, torch.sum(MSE_test), model, optimizer)
            if update_sparsity: 
                model.constraint.sparsity_masks = model.sparse_estimator(thetas, time_derivs)

            # ================= Checking convergence
            l1_norm = torch.sum(torch.abs(torch.cat(model.constraint_coeffs(sparse=True, scaled=True), dim=1)))
            converged = convergence(iteration, l1_norm)
            if converged:
                break
    logger.close(model)