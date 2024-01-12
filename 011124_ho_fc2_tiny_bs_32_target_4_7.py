from hyperparameter_optimization import minimize_lpips_for_specific_image_with_id
from attack import cocktail_party_attack

model_config = './model_config/fc2_cocktail_party_tiny_imagenet_instance.json'
checkpoint_path =  './checkpoints/010424_fc2_cocktail_party_tiny_imagenet_pretraining_wout_bias_wout_normalization.pth'
data_type = 'tiny-imagenet'
data_path = './data'
batch_size = 32
num_of_trials = 200
target_image_arr = list(range(4, 8))
height, width = 64, 64
device = 1
logger = './hyperparameter_optimization_log/011124_ho_fc2_tiny_imagenet_bs_32_target_{}.log'

for target_id in target_image_arr:
    study = minimize_lpips_for_specific_image_with_id(model_config, checkpoint_path, data_type, data_path,
                                                      batch_size, num_of_trials, target_id,
                                                      logger=logger.format(target_id),
                                                      height=height, width=width, device_number=device)
    t_param = study.best_trial.params['t_param']
    total_variance_loss_param = study.best_trial.params['total_variance_loss_param']
    mutual_independence_loss_param = study.best_trial.params['mutual_independence_loss_param']
    save_results = '011124_fc2_tiny_imagenet_bs_32_target_{}'.format(target_id)
    result_dict = cocktail_party_attack(model_config, checkpoint_path, data_type, data_path,
                                        batch_size, t_param, total_variance_loss_param,
                                        mutual_independence_loss_param, height=height, width=width,
                                        device_number=device, return_specific_with_id=target_id,
                                        plot_shape=(8, 4), save_results=save_results, save_json=True,
                                        save_figure=True, plot_verbose=False) 
