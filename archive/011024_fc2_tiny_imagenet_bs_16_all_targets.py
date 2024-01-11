from attack import cocktail_party_attack
import os

model_config = './model_config/fc2_cocktail_party_tiny_imagenet_instance.json'
checkpoint = './checkpoints/010424_fc2_cocktail_party_tiny_imagenet_pretraining_wout_bias_wout_normalization.pth'
data_type = 'tiny-imagenet'
data_path = '../data'
batch_size = 16
log_files = os.listdir('./hyperparameter_optimization_log/')
for log_file in log_files:
    target_id = int(log_file.split('.')[0].split('_')[-1])
    with open(os.path.join('./hyperparameter_optimization_log/', log_file), 'r') as fp:
        lines = fp.readlines()
        best_trial = lines[int(lines[-1][lines[-1].index('Best is trial '):].split(' ')[3]) + 1]
        params = [float(x.split(' ')[-1]) for x in best_trial.split('{')[-1].split('}')[0].split(', ')]
        t_param, total_variance_loss_param, mutual_independence_loss_param = params
    
    print('#####################################################')
    print('experiment with target id {} started'.format(target_id))
    result_dict = cocktail_party_attack(model_config, checkpoint, data_type, data_path, batch_size, t_param,
                                    total_variance_loss_param,
                                    mutual_independence_loss_param, height=64, width=64, return_specific_with_id=target_id,
                                    plot_shape=(4, 4),
                                    save_results='011024_fc2_tiny_imagenet_bs_16_target_{}'.format(target_id),
                                    save_json=True,
                                    save_figure=True, plot_verbose=False)
    print('#####################################################\n')