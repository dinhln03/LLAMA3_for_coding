import torch
import time
from math import pi
import numpy as np

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

    # Training
    convergence = Convergence(**convergence_kwargs)
    print('| Iteration | Progress | Time remaining |     Loss |      MSE |      Reg |    L1 norm |')
    for iteration in np.arange(0, max_iterations + 1):
        # ================== Training Model ============================
        prediction, time_derivs, thetas = model(data)

        MSE = torch.mean((prediction - target)**2, dim=0)  # loss per output
        Reg = torch.stack([torch.mean((dt - theta @ coeff_vector)**2)
                           for dt, theta, coeff_vector in zip(time_derivs, thetas, model.constraint_coeffs(scaled=False, sparse=True))])
        loss = torch.sum(MSE + Reg)  # 1e-5 for numerical stability

        # Optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ====================== Logging =======================
        # We calculate the normalization factor and the l1_norm
        l1_norm = torch.sum(torch.abs(torch.cat(model.constraint_coeffs(sparse=True, scaled=True), dim=1)), dim=0)

        # Write progress to command line and tensorboard
        if iteration % write_iterations == 0:
            _ = model.sparse_estimator(thetas, time_derivs) # calculating l1 adjusted coeffs but not setting mask
            progress(iteration, start_time, max_iterations, loss.item(),
                     torch.sum(MSE).item(), torch.sum(Reg).item(), torch.sum(l1_norm).item())

            if model.estimator_coeffs() is None:
                estimator_coeff_vectors = [torch.zeros_like(coeff) for coeff in model.constraint_coeffs(sparse=True, scaled=False)] # It doesnt exist before we start sparsity, so we use zeros
            else:
                estimator_coeff_vectors = model.estimator_coeffs()

            board.write(iteration, loss, MSE, Reg, l1_norm, model.constraint_coeffs(sparse=True, scaled=True), model.constraint_coeffs(sparse=True, scaled=False), estimator_coeff_vectors)
            
        # ================== Validation and sparsity =============
        # Updating sparsity and or convergence
        sparsity_scheduler(iteration, torch.sum(l1_norm))
        if sparsity_scheduler.apply_sparsity is True:
            with torch.no_grad():
                model.constraint.sparsity_masks = model.sparse_estimator(thetas, time_derivs)
                sparsity_scheduler.reset()
                print(model.sparsity_masks)

        # Checking convergence
        convergence(iteration, torch.sum(l1_norm))
        if convergence.converged is True:
            print('Algorithm converged. Stopping training.')
            break
    board.close()


def train_auto_split(model: DeepMoD,
          data: torch.Tensor,
          target: torch.Tensor,
          optimizer,
          sparsity_scheduler,
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

        # ====================== Logging =======================
        # We calculate the normalization factor and the l1_norm
        l1_norm = torch.sum(torch.abs(torch.cat(model.constraint_coeffs(sparse=True, scaled=True), dim=1)), dim=0)
        
        # Validation loss
        with torch.no_grad():
            prediction_test = model.func_approx(data_test)[0]
            MSE_test = torch.mean((prediction_test - target_test)**2, dim=0)  # loss per output
        # Write progress to command line and tensorboard
        if iteration % write_iterations == 0:
            _ = model.sparse_estimator(thetas, time_derivs) # calculating l1 adjusted coeffs but not setting mask
            estimator_coeff_vectors = model.estimator_coeffs()

            progress(iteration, start_time, max_iterations, loss.item(),
                     torch.sum(MSE).item(), torch.sum(Reg).item(), torch.sum(l1_norm).item())
            board.write(iteration, loss, MSE, Reg, l1_norm, model.constraint_coeffs(sparse=True, scaled=True), model.constraint_coeffs(sparse=True, scaled=False), estimator_coeff_vectors, MSE_test=MSE_test)
            
        # ================== Validation and sparsity =============
        # Updating sparsity and or convergence
        sparsity_scheduler(iteration, torch.sum(MSE_test), model, optimizer)
        #sparsity_scheduler(torch.sum(MSE_test), model, optimizer)
        if sparsity_scheduler.apply_sparsity is True:
            with torch.no_grad():
                model.constraint.sparsity_masks = model.sparse_estimator(thetas, time_derivs)
                sparsity_scheduler.reset()
                print(model.sparsity_masks)
    

        # Checking convergence
        convergence(iteration, torch.sum(l1_norm))
        if convergence.converged is True:
            print('Algorithm converged. Stopping training.')
            break
    board.close()

    

    
def train_auto_split_scaled(model: DeepMoD,
          data: torch.Tensor,
          target: torch.Tensor,
          optimizer,
          sparsity_scheduler,
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
        
        theta_norms = [torch.norm(theta, dim=0) for theta in thetas]
        time_deriv_norms = [torch.norm(dt, dim=0) for dt in time_derivs]
        normed_thetas = [theta / norm for theta, norm in zip(thetas, theta_norms)]
        normed_time_derivs = [dt / norm for dt, norm in zip(time_derivs, time_deriv_norms)]

        Reg = torch.stack([torch.mean((dt - theta @ coeff_vector)**2)
                           for dt, theta, coeff_vector in zip(normed_time_derivs, normed_thetas, model.constraint_coeffs(scaled=True, sparse=True))])
        loss = torch.sum(MSE + Reg)  # 1e-5 for numerical stability

        # Optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ====================== Logging =======================
        # We calculate the normalization factor and the l1_norm
        l1_norm = torch.sum(torch.abs(torch.cat(model.constraint_coeffs(sparse=True, scaled=True), dim=1)), dim=0)
        

        # Validation loss
        prediction_test, coordinates = model.func_approx(data_test)
        time_derivs_test, thetas_test = model.library((prediction_test, coordinates))
        MSE_test = torch.mean((prediction_test - target_test)**2, dim=0)  # loss per output
        Reg_test = torch.stack([torch.mean((dt - theta @ coeff_vector)**2)
                           for dt, theta, coeff_vector in zip(time_derivs_test, thetas_test, model.constraint_coeffs(scaled=False, sparse=True))])
        loss_test = torch.sum(MSE_test + Reg_test) 

        # Write progress to command line and tensorboard
        if iteration % write_iterations == 0:
            _ = model.sparse_estimator(thetas, time_derivs) # calculating l1 adjusted coeffs but not setting mask
            estimator_coeff_vectors = model.estimator_coeffs()

            progress(iteration, start_time, max_iterations, loss.item(),
                     torch.sum(MSE).item(), torch.sum(Reg).item(), torch.sum(l1_norm).item())
            board.write(iteration, loss, MSE, Reg, l1_norm, model.constraint_coeffs(sparse=True, scaled=True), model.constraint_coeffs(sparse=True, scaled=False), estimator_coeff_vectors, MSE_test=MSE_test, Reg_test=Reg_test, loss_test=loss_test)
            
        # ================== Validation and sparsity =============
        # Updating sparsity and or convergence
        sparsity_scheduler(loss_test, model, optimizer)
        #sparsity_scheduler(torch.sum(MSE_test), model, optimizer)
        if sparsity_scheduler.apply_sparsity is True:
            with torch.no_grad():
                checkpoint = torch.load(sparsity_scheduler.path)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                model.constraint.sparsity_masks = model.sparse_estimator(thetas, time_derivs)
                sparsity_scheduler.reset()
                print(model.sparsity_masks)

        # Checking convergence
        convergence(iteration, torch.sum(l1_norm))
        if convergence.converged is True:
            print('Algorithm converged. Stopping training.')
            break
    board.close()


def train_auto_split_MSE(model: DeepMoD,
          data: torch.Tensor,
          target: torch.Tensor,
          optimizer,
          sparsity_scheduler,
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

    # Training
    convergence = Convergence(**convergence_kwargs)
    print('| Iteration | Progress | Time remaining |     Loss |      MSE |      Reg |    L1 norm |')
    for iteration in np.arange(0, max_iterations + 1):
        # ================== Training Model ============================
        prediction, time_derivs, thetas = model(data)

        MSE = torch.mean((prediction - target)**2, dim=0)  # loss per output
        #Reg = torch.stack([torch.mean((dt - theta @ coeff_vector)**2)
        #                   for dt, theta, coeff_vector in zip(time_derivs, thetas, model.constraint_coeffs(scaled=False, sparse=True))])
        loss = torch.sum(MSE)  # 1e-5 for numerical stability

        # Optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ====================== Logging =======================
        with torch.no_grad():
             # We calculate the normalization factor and the l1_norm
            l1_norm = torch.sum(torch.abs(torch.cat(model.constraint.coeff_vectors, dim=1)), dim=0)
            # Validation loss
            prediction_test = model.func_approx(data)[0]
            MSE_test = torch.mean((prediction_test - target)**2, dim=0)  # loss per output
            # Write progress to command line and tensorboard
            if iteration % write_iterations == 0:
                _ = model.sparse_estimator(thetas, time_derivs) # calculating l1 adjusted coeffs but not setting mask
                estimator_coeff_vectors = model.estimator_coeffs()

                progress(iteration, start_time, max_iterations, loss.item(),
                        torch.sum(MSE).item(), torch.sum(MSE).item(), torch.sum(l1_norm).item())
                board.write(iteration, loss, MSE, MSE, l1_norm, model.constraint.coeff_vectors, model.constraint.coeff_vectors, estimator_coeff_vectors, MSE_test=MSE_test)
            
        # ================== Validation and sparsity =============
        # Updating sparsity and or convergence
        sparsity_scheduler(iteration, torch.sum(MSE_test), model, optimizer)
        if sparsity_scheduler.apply_sparsity is True:
            with torch.no_grad():
                model.constraint.sparsity_masks = model.sparse_estimator(thetas, time_derivs)
                sparsity_scheduler.reset()
                print(model.sparsity_masks)

        # Checking convergence
        convergence(iteration, torch.sum(l1_norm))
        if convergence.converged is True:
            print('Algorithm converged. Stopping training.')
            break
    board.close()


def train_split_full(model: DeepMoD,
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