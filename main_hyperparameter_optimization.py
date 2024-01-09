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
    parse.add_argument('-l', '--logger', type=str)
    parse.add_argument('-he', '--height', type=int)
    parse.add_argument('-w', '--width', type=int)
    parse.add_argument('-rs', '--random_seed', type=int)
    parse.add_argument('-dn', '--device_number', type=int)
    parse.add_argument('-rm', '--return_metrics', type=bool)
    parse.add_argument('-rma', '--return_matches', type=bool)
    parse.add_argument('-rswi', '--return_specific_with_id', type=int)
    parse.add_argument('-v', '--verbose', type=bool)
    parse.add_argument('-ps', '--plot_shape', type=tuple)
    parse.add_argument('-sr', '--save_results', type=str)
    parse.add_argument('-sj', '--save_json', type=bool)
    parse.add_argument('-sf', '--save_figure', type=bool)
    parse.add_argument('-pv', '--plot_verbose', type=bool)


    args = parse.parse_args()


    hyperparameter_optimization_type = args.hyperparameter_optimization_type
    model_config = args.model_config
    checkpoint_path = args.checkpoint_path
    data_type = args.data_type
    data_path = args.data_path
    batch_size = args.batch_size
    num_of_trials = args.num_of_trials
    logger = args.logger
    height = args.height
    width = args.width
    random_seed = args.random_seed
    device_number = args.device_number
    return_metrics = args.return_metrics
    return_matches = args.return_matches
    return_specific_with_id = args.return_specific_with_id
    verbose = args.verbose
    plot_shape = args.plot_shape
    save_results = args.save_results
    save_json = args.save_json
    save_figure = args.save_figure
    plot_verbose = args.plot_verbose

    if hyperparameter_optimization_type == 'psnr':
        pass
    elif hyperparameter_optimization_type == 'lpips':
        minimize_lpips(hyperparameter_optimization_type, model_config, checkpoint_path, data_type, data_path, batch_size, num_of_trials, logger=logger, height=height, width=width, random_seed=random_seed, device_number=device_number, return_metrics=return_metrics, return_matches=return_matches, return_specific_with_id=return_specific_with_id, verbose=verbose, plot_shape=plot_shape, save_results=save_results, save_json=save_json, save_figure=save_figure, plot_verbose=plot_verbose)
    elif hyperparameter_optimization_type == 'lpips_specific':
        pass
    else:
        print('the hyperparameter optimization type is not supported')