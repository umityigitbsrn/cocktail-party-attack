from hyperparameter_optimization import maximize_psnr, minimize_lpips, minimize_lpips_for_specific_image_with_id
from attack import cocktail_party_attack

class CocktailPartyAttack(object):
    def __init__(self, model_config, checkpoint_path, data_type, data_path, batch_size, *args, return_specific_with_id=None,
                 t_param=None, total_variance_loss_param=None, mutual_independence_loss_param=None,
                 height=32, width=32, random_seed=2024, device_number=0, use_gradient_difference=False, **kwargs):
        super(CocktailPartyAttack, self).__init__(*args, **kwargs)
        self.model_config = model_config
        self.checkpoint_path = checkpoint_path
        self.data_type = data_type
        self.data_path = data_path
        self.batch_size = batch_size

        self.target_id = return_specific_with_id
        self.t_param = t_param
        self.total_variance_loss_param = total_variance_loss_param
        self.mutual_independence_loss_param = mutual_independence_loss_param
        self.height = height
        self.width = width
        self.random_seed = random_seed
        self.device_number = device_number
        self.use_gradient_difference = use_gradient_difference

        # hyper parameter optimization study
        self.study = None
        self.optimized_regarding = None

        # attack result dict
        self.attack_result_dict = None


    def attack(self, verbose=True, plot_shape=None,
               save_results=None, save_json=False, save_figure=False, plot_verbose=True,
               save_estimations_and_references=False):
        self.attack_result_dict = cocktail_party_attack(self.model_config, self.checkpoint_path,
                                                        self.data_type, self.data_path, self.batch_size,
                                                        self.t_param, self.total_variance_loss_param,
                                                        self.mutual_independence_loss_param,
                                                        height=self.height, width=self.width,
                                                        random_seed=self.random_seed, device_number=self.device_number,
                                                        return_specific_with_id=self.target_id, verbose=verbose, plot_shape=plot_shape,
                                                        save_results=save_results, save_json=save_json, save_figure=save_figure,
                                                        plot_verbose=plot_verbose, save_estimations_and_references=save_estimations_and_references,
                                                        use_gradient_difference=self.use_gradient_difference)

    def optimize_hyperparameters(self, optimization_type, num_of_trials, logger=None):
        """_summary_

        Args:
            optimization_type (_type_): _description_
            num_of_trials (_type_): _description_
            logger (_type_, optional): _description_. Defaults to None.
        """
        if optimization_type == 'psnr':
            self.study = maximize_psnr(self.model_config, self.checkpoint_path, self.data_type, self.data_path, self.batch_size,
                                  num_of_trials, logger=logger, height=self.height, width=self.width, 
                                  random_seed=self.random_seed, device_number=self.device_number,
                                  use_gradient_difference=self.use_gradient_difference)
        elif optimization_type == 'lpips':
            self.study = minimize_lpips(self.model_config, self.checkpoint_path, self.data_type, self.data_path, self.batch_size,
                                   num_of_trials, logger=logger, height=self.height, width=self.width, 
                                   random_seed=self.random_seed, device_number=self.device_number,
                                   use_gradient_difference=self.use_gradient_difference)
        elif optimization_type == 'lpips-target':
            self.study = minimize_lpips_for_specific_image_with_id(self.model_config, self.checkpoint_path,
                                                              self.data_type, self.data_path, self.batch_size,
                                                              num_of_trials, self.target_id, logger=logger,
                                                              height=self.height, width=self.width,
                                                              random_seed=self.random_seed, device_number=self.device_number,
                                                              use_gradient_difference=self.use_gradient_difference)
        else:
            try:
                raise Exception('Unsupported optimization type')
            except Exception as e:
                print('Exception raised: {}'.format(e))
        self.optimized_regarding = optimization_type
        if self.study is not None:
            self.t_param = self.study.best_trial.params['t_param']
            self.total_variance_loss_param = self.study.best_trial.params['total_variance_loss_param']
            self.mutual_independence_loss_param = self.study.best_trial.params['mutual_independence_loss_param']
        
    def optimize_hyperparameters_then_attack(self, optimization_type, num_of_trials, logger=None, verbose=True, plot_shape=None,
                                             save_results=None, save_json=False, save_figure=False, plot_verbose=True,
                                             save_estimations_and_references=False):
        self.optimize_hyperparameters(optimization_type, num_of_trials, logger=logger)
        self.attack(verbose=verbose, plot_shape=plot_shape, save_results=save_results, save_json=save_json,
                    save_figure=save_figure, plot_verbose=plot_verbose,
                    save_estimations_and_references=save_estimations_and_references)