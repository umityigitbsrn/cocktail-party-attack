from attack import cocktail_party_attack

model_config = './model_config/fc2_cocktail_party_tiny_imagenet_instance.json'
checkpoint = './checkpoints/010424_fc2_cocktail_party_tiny_imagenet_pretraining_wout_bias_wout_normalization.pth'
data_type = 'tiny-imagenet'
data_path = '../data'
batch_size = 16
t_param = 7.40907780051146
total_variance_param = 0.28220679759448913
mutual_independence_loss_param = 7.963207128296047
result_dict = cocktail_party_attack(model_config, checkpoint, data_type, data_path, batch_size, t_param,
                                    total_variance_param,
                                    mutual_independence_loss_param, height=64, width=64,
                                    plot_shape=(4, 4),
                                    save_results='010924_fc2_tiny_imagenet_bs_16',
                                    save_json=True,
                                    save_figure=True, plot_verbose=False)