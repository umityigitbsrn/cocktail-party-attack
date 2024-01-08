import optuna

from attack import cocktail_party_attack


def maximize_psnr(model_config, checkpoint_path, data_type, data_path, batch_size, num_of_trials, **kwargs):
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

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=num_of_trials)


def minimize_lpips(model_config, checkpoint_path, data_type, data_path, batch_size, num_of_trials, **kwargs):
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

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=num_of_trials)


def minimize_lpips_for_specific_image_with_id(model_config, checkpoint_path, data_type, data_path, batch_size,
                                              num_of_trials, target_id, **kwargs):
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

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=num_of_trials)
