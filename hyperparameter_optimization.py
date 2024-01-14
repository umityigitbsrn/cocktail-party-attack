import os.path

import optuna
import logging

from attack import cocktail_party_attack


def maximize_psnr(model_config, checkpoint_path, data_type, data_path, batch_size, num_of_trials, logger=None, **kwargs):
    """the function for optimizing the hyperparameters by maximizing the mean psnr metric values

    Args:
        model_config (str): path to model config (whole path)
        checkpoint_path (str): path to model checkpoint (whole path)
        data_type (str): choice either [tiny-imagenet, cifar10, cifar100, mnist]
        data_path (str): root data folder path (generally './data')
        batch_size (int): batch size
        num_of_trials (int): number of trials for hyperparameter optimization
        logger (str, optional): log output file to redirect stdout to a logger for hyperparam optimization. Defaults to None.
    """
    def objective(trial):
        t_param = trial.suggest_float('t_param', 0.00001, 10)
        total_variance_loss_param = trial.suggest_float('total_variance_loss_param', 0.00001, 10)
        mutual_independence_loss_param = trial.suggest_float('mutual_independence_loss_param', 0.00001, 10)
        result_dict = cocktail_party_attack(model_config, checkpoint_path, data_type, data_path, batch_size, t_param,
                                            total_variance_loss_param, mutual_independence_loss_param, verbose=False,
                                            **kwargs)
        if isinstance(result_dict, dict):
            return result_dict['psnr']['mean_psnr']
        else:
            return result_dict

    if logger is not None:
        logger = os.path.join('./hyperparamater_optimization_log', logger)
        if not os.path.exists(logger):
            os.makedirs(logger)

        logging_logger = logging.getLogger()

        logging_logger.setLevel(logging.INFO)  # Setup the root logging_logger.
        logging_logger.addHandler(logging.FileHandler(logger, mode="w"))

        optuna.logging.enable_propagation()  # Propagate logs to the root logging_logger.
        optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=num_of_trials)
    
    return study

def minimize_lpips(model_config, checkpoint_path, data_type, data_path, batch_size, num_of_trials, logger=None, **kwargs):
    """the function for optimizing the hyperparameters by minimizing the mean lpips metric values

        Args:
        model_config (str): path to model config (whole path)
        checkpoint_path (str): path to model checkpoint (whole path)
        data_type (str): choice either [tiny-imagenet, cifar10, cifar100, mnist]
        data_path (str): root data folder path (generally './data')
        batch_size (int): batch size
        num_of_trials (int): number of trials for hyperparameter optimization
        logger (str, optional): log output file to redirect stdout to a logger for hyperparam optimization. Defaults to None.
    """
    def objective(trial):
        t_param = trial.suggest_float('t_param', 0.00001, 10)
        total_variance_loss_param = trial.suggest_float('total_variance_loss_param', 0.00001, 10)
        mutual_independence_loss_param = trial.suggest_float('mutual_independence_loss_param', 0.00001, 10)
        result_dict = cocktail_party_attack(model_config, checkpoint_path, data_type, data_path, batch_size, t_param,
                                            total_variance_loss_param, mutual_independence_loss_param, verbose=False,
                                            **kwargs)

        if isinstance(result_dict, dict):
            return result_dict['lpips']['mean_lpips']
        else:
            return result_dict
    
    if logger is not None:
        logger = os.path.join('./hyperparamater_optimization_log', logger)
        if not os.path.exists(logger):
            os.makedirs(logger)

        logging_logger = logging.getLogger()

        logging_logger.setLevel(logging.INFO)  # Setup the root logging_logger.
        logging_logger.addHandler(logging.FileHandler(logger, mode="w"))

        optuna.logging.enable_propagation()  # Propagate logs to the root logging_logger.
        optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=num_of_trials)

    return study


def minimize_lpips_for_specific_image_with_id(model_config, checkpoint_path, data_type, data_path, batch_size,
                                              num_of_trials, target_id, logger=None, **kwargs):
    """the function for optimizing the hyperparameters by minimizing the individual lpips metric value of an target image

        Args:
        model_config (str): path to model config (whole path)
        checkpoint_path (str): path to model checkpoint (whole path)
        data_type (str): choice either [tiny-imagenet, cifar10, cifar100, mnist]
        data_path (str): root data folder path (generally './data')
        batch_size (int): batch size
        num_of_trials (int): number of trials for hyperparameter optimization
        target_id (int): target reference image id in a batch to reconstruct the image based on the individual lpips score
        logger (str, optional): log output file to redirect stdout to a logger for hyperparam optimization. Defaults to None.
    """
    def objective(trial):
        t_param = trial.suggest_float('t_param', 0.00001, 10)
        total_variance_loss_param = trial.suggest_float('total_variance_loss_param', 0.00001, 10)
        mutual_independence_loss_param = trial.suggest_float('mutual_independence_loss_param', 0.00001, 10)
        result_dict = cocktail_party_attack(model_config, checkpoint_path, data_type, data_path, batch_size, t_param,
                                            total_variance_loss_param, mutual_independence_loss_param, verbose=False,
                                            return_specific_with_id=target_id, **kwargs)

        if isinstance(result_dict, dict):
            return result_dict['lpips_with_id']['lpips']
        else:
            return result_dict

    if logger is not None:
        logger = os.path.join('./hyperparamater_optimization_log', logger)
        if not os.path.exists(logger):
            os.makedirs(logger)

        logging_logger = logging.getLogger()

        logging_logger.setLevel(logging.INFO)  # Setup the root logging_logger.
        logging_logger.addHandler(logging.FileHandler(logger, mode="w"))

        optuna.logging.enable_propagation()  # Propagate logs to the root logging_logger.
        optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=num_of_trials)

    return study
