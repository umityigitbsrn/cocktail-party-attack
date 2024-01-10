import argparse
from hyperparameter_optimization import maximize_psnr, minimize_lpips, minimize_lpips_for_specific_image_with_id

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('-hot', '--hyperparameter_optimization_type', type=str, required=True)
    parse.add_argument('-mc', '--model_config', type=str, required=True)
    parse.add_argument('-cp', '--checkpoint_path', type=str, required=True)
    parse.add_argument('-dt', '--data_type', type=str, required=True)
    parse.add_argument('-dp', '--data_path', type=str, required=True)
    parse.add_argument('-bs', '--batch_size', type=int, required=True)
    parse.add_argument('-not', '--num_of_trials', type=int, required=True)
    parse.add_argument('-l', '--logger', type=str, default=argparse.SUPPRESS)
    parse.add_argument('-he', '--height', type=int, default=argparse.SUPPRESS)
    parse.add_argument('-w', '--width', type=int, default=argparse.SUPPRESS)
    parse.add_argument('-rs', '--random_seed', type=int, default=argparse.SUPPRESS)
    parse.add_argument('-dn', '--device_number', type=int, default=argparse.SUPPRESS)
    parse.add_argument('-ti', '--target_id', type=int, default=argparse.SUPPRESS)

    args = parse.parse_args()

    hyperparameter_optimization_type = args.hyperparameter_optimization_type
    kwargs = vars(args)
    del kwargs['hyperparameter_optimization_type']

    if hyperparameter_optimization_type == 'psnr':
        maximize_psnr(**vars(args))
    elif hyperparameter_optimization_type == 'lpips':
        minimize_lpips(**vars(args))
    elif hyperparameter_optimization_type == 'lpips_specific':
        minimize_lpips_for_specific_image_with_id(**vars(args))
    else:
        print('the hyperparameter optimization type is not supported')