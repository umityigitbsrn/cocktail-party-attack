nohup python main_hyperparameter_optimization.py -hot lpips_specific \
                                           -mc ./model_config/fc2_cocktail_party_tiny_imagenet_instance.json \
                                           -cp ./checkpoints/010424_fc2_cocktail_party_tiny_imagenet_pretraining_wout_bias_wout_normalization.pth \
                                           -dt tiny-imagenet \
                                           -dp ./data \
                                           -bs 16 \
                                           -not 200 \
                                           -ti 8 \
                                           -l ./hyperparameter_optimization_log/010924_ho_fc2_tiny_imagenet_bs_16_ti_8.log \
                                           -he 64 \
                                           -w 64 \
                                           -dn 0 \
                                           > nohup_ho_fc2_tiny_imagenet_bs_16_ti_8.out &

nohup python main_hyperparameter_optimization.py -hot lpips_specific \
                                           -mc ./model_config/fc2_cocktail_party_tiny_imagenet_instance.json \
                                           -cp ./checkpoints/010424_fc2_cocktail_party_tiny_imagenet_pretraining_wout_bias_wout_normalization.pth \
                                           -dt tiny-imagenet \
                                           -dp ./data \
                                           -bs 16 \
                                           -not 200 \
                                           -ti 9 \
                                           -l ./hyperparameter_optimization_log/010924_ho_fc2_tiny_imagenet_bs_16_ti_9.log \
                                           -he 64 \
                                           -w 64 \
                                           -dn 1 \
                                           > nohup_ho_fc2_tiny_imagenet_bs_16_ti_9.out &

nohup python main_hyperparameter_optimization.py -hot lpips_specific \
                                           -mc ./model_config/fc2_cocktail_party_tiny_imagenet_instance.json \
                                           -cp ./checkpoints/010424_fc2_cocktail_party_tiny_imagenet_pretraining_wout_bias_wout_normalization.pth \
                                           -dt tiny-imagenet \
                                           -dp ./data \
                                           -bs 16 \
                                           -not 200 \
                                           -ti 10 \
                                           -l ./hyperparameter_optimization_log/010924_ho_fc2_tiny_imagenet_bs_16_ti_10.log \
                                           -he 64 \
                                           -w 64 \
                                           -dn 2 \
                                           > nohup_ho_fc2_tiny_imagenet_bs_16_ti_10.out &

nohup python main_hyperparameter_optimization.py -hot lpips_specific \
                                           -mc ./model_config/fc2_cocktail_party_tiny_imagenet_instance.json \
                                           -cp ./checkpoints/010424_fc2_cocktail_party_tiny_imagenet_pretraining_wout_bias_wout_normalization.pth \
                                           -dt tiny-imagenet \
                                           -dp ./data \
                                           -bs 16 \
                                           -not 200 \
                                           -ti 11 \
                                           -l ./hyperparameter_optimization_log/010924_ho_fc2_tiny_imagenet_bs_16_ti_11.log \
                                           -he 64 \
                                           -w 64 \
                                           -dn 3 \
                                           > nohup_ho_fc2_tiny_imagenet_bs_16_ti_11.out &