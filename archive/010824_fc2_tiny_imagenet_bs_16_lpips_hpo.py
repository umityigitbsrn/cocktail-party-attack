from hyperparameter_optimization import minimize_lpips

batch_size = 16
num_of_trials = 200
minimize_lpips('./model_config/fc2_cocktail_party_tiny_imagenet_instance.json',
               '../checkpoints/010424_fc2_cocktail_party_tiny_imagenet_pretraining_wout_bias_wout_normalization.pth',
               'tiny-imagenet', './data', batch_size, 200,
               logger='./hyperparameter_optimization_log/010824_fc2_tiny_imagenet_bs_16_lpips_hpo.log', height=64, width=64, device_number=0)