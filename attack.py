# Now, there should be two kind of experiment structures one of them for attacking data with default data loaders
# the other one is the data with a dict structure, the below implementation is a generalized version of these two

import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import json
import copy
import os

from data import load_mnist_dataloaders, load_cifar10_dataloaders, load_cifar100_dataloaders, \
    load_tiny_imagenet_dataloaders, WhiteningTransformation
from metrics import psnr, lpips_gpu, lpips_matching_specific_image_gpu
from model import Network
from criterion import ReconstructImageFromFCLoss

LPIPS_MAX = 100000000


def _turn_tensors_to_list(dict_elem):
    for key, value in dict_elem.items():
        if isinstance(value, torch.Tensor):
            dict_elem[key] = value.tolist()
        elif isinstance(value, dict):
            dict_elem[key] = _turn_tensors_to_list(value)
        elif isinstance(value, list):
            new_value = []
            for item in value:
                if isinstance(item, torch.Tensor):
                    new_value.append(item.tolist())
                else:
                    new_value.append(item)
            dict_elem[key] = new_value
    return dict_elem

def _load_pretrained_model(model_config, checkpoint_path, device):
    model = Network(model_config)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    return model

def _get_reference_image_batch(data_type, data_path, batch_size, transform, device, save_estimations_and_references, result_dict):
    val_dataloader = None
    if data_type == 'mnist':
        _, val_dataloader = load_mnist_dataloaders(data_path, batch_size, transform)
    elif data_type == 'cifar10':
        _, val_dataloader = load_cifar10_dataloaders(data_path, batch_size, transform)
    elif data_type == 'cifar100':
        _, val_dataloader = load_cifar100_dataloaders(data_path, batch_size, transform)
    elif data_type == 'tiny-imagenet':
        _, val_dataloader = load_tiny_imagenet_dataloaders(batch_size)

    if val_dataloader is None:
        try:
            raise Exception('The data type is not correct')
        except Exception as e:
            print('The data type is not correct with exception: {}'.format(e))

    # if tiny imagenet then the data object is a dict therefore to send them to 
    # the GPU it needs to be handled
    if data_type == 'tiny-imagenet':
        data_dict = next(iter(val_dataloader))
        selected_val_batch_data, selected_val_batch_label = data_dict['image'], data_dict['label']
    else:
        selected_val_batch_data, selected_val_batch_label = next(iter(val_dataloader))
    if save_estimations_and_references:
        result_dict['reference_img_batch']: selected_val_batch_data.detach().to('cpu')
    selected_val_batch_data = selected_val_batch_data.to(device)
    selected_val_batch_label = selected_val_batch_label.to(device)
    return selected_val_batch_data, selected_val_batch_label

def _get_whitened_gradients(model, selected_val_batch_data, selected_val_batch_label, batch_size, device):
    # receiving gradients
    model.zero_grad()
    criterion = nn.CrossEntropyLoss()
    output = model(selected_val_batch_data.reshape(batch_size, -1))
    loss = criterion(output, selected_val_batch_label)
    loss.backward()
    gradient_of_layers = []
    for param in model.parameters():
        gradient_of_layers.append(param.grad.data.clone().to('cpu'))

    # whitening the first fully connected layer
    whitening_transform = WhiteningTransformation()
    whitened_gradient = torch.from_numpy(whitening_transform.transform(gradient_of_layers[0].detach().numpy().T)).to(
        torch.float32).T
    whitened_gradient = whitened_gradient.to(device)
    return whitened_gradient

def _main_cocktail_party_attack(selected_val_batch_data, whitened_gradient,
                                t_param, total_variance_loss_param, mutual_independence_loss_param,
                                height, width, device, save_estimations_and_references, verbose, result_dict):
    if verbose:
        print('############# ATTACK IS STARTED #############')

    # criterion output testing
    unmixing_matrix = torch.rand((selected_val_batch_data.size(0), whitened_gradient.size(0)), requires_grad=True,
                                 device=device)
    reconstruction_loss = ReconstructImageFromFCLoss(height, width, 3, t_param, total_variance_loss_param,
                                                     mutual_independence_loss_param)
    optimizer = torch.optim.Adam([unmixing_matrix])

    for iter_idx in range(25000):
        optimizer.zero_grad()
        # out_score, non_gaussianity_score, total_variance_score, mutual_independence_score
        loss, _, _, _ = reconstruction_loss(unmixing_matrix, whitened_gradient)
        loss.backward()
        optimizer.step()

        if torch.isinf(loss).item() or torch.isnan(loss).item():
            return LPIPS_MAX

        if verbose:
            if (iter_idx + 1) % 1000 == 0 or iter_idx == 0:
                print('loss: {}'.format(loss.item()))

    unmixing_matrix = unmixing_matrix.detach().to('cpu')
    whitened_gradient = whitened_gradient.detach().to('cpu')
    estimated_img_batch = unmixing_matrix @ whitened_gradient
    if save_estimations_and_references:
        result_dict['estimated_img_batch'] = estimated_img_batch

    if verbose:
        print('############# ATTACK IS FINISHED #############')
    return estimated_img_batch

def _get_metrics(estimated_img_batch, selected_val_batch_data, height, width, return_specific_with_id, result_dict, verbose):
    lpips_match, lpips_is_positive, mean_lpips = lpips_gpu(estimated_img_batch, selected_val_batch_data, height=height, width=width)
    psnr_match, psnr_is_positive, mean_psnr = psnr(estimated_img_batch, selected_val_batch_data, height=height, width=width)
    lpips_key, psnr_key = 'lpips', 'psnr'
    result_dict[lpips_key] = {'matches': lpips_match, 'is_positive': lpips_is_positive, 'mean_lpips': mean_lpips}
    result_dict[psnr_key] = {'matches': psnr_match, 'is_positive': psnr_is_positive, 'mean_psnr': mean_psnr}
    if verbose:
        print('LPIPS ({}) and PSNR ({}) values are calculated and added to list of results'.format(mean_lpips,
                                                                                                    mean_psnr))

    if return_specific_with_id is not None:
        best_estimation_id, pos_estimation, min_lpips_value = lpips_matching_specific_image_gpu(estimated_img_batch,
                                                                                                selected_val_batch_data[return_specific_with_id],
                                                                                                height=height,
                                                                                                width=width)
        result_dict['lpips_with_id'] = {'best_estimation': best_estimation_id, 'is_positive': pos_estimation,
                                        'lpips': min_lpips_value}
        if verbose:
            print('The best estimation for the reference image with id {} is {} {} with LPIPS score: {}'.format(
                return_specific_with_id, 'positive' if pos_estimation else 'negative', best_estimation_id,
                min_lpips_value))
            
def _plot_results(estimated_img_batch, selected_val_batch_data,
                  plot_shape, return_specific_with_id, height, width,
                  save_results, save_figure, result_dict,
                  verbose, plot_verbose):
    lpips_match = result_dict['lpips']['matches']
    lpips_is_positive = result_dict['lpips']['is_positive']
    best_estimation_id, pos_estimation = result_dict['lpips_with_id']['best_estimation'], result_dict['lpips_with_id']['is_positive'] 
    if plot_shape is not None:
        fig, axes = plt.subplots(*plot_shape)

        for match, ax in zip(lpips_match, axes.flatten()):
            if return_specific_with_id is not None and match[0].item() == best_estimation_id:
                ax.spines['bottom'].set_color('red')
                ax.spines['top'].set_color('red')
                ax.spines['right'].set_color('red')
                ax.spines['left'].set_color('red')
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.axis('off')

            estimate = estimated_img_batch[match[0].item()]
            estimate_coeff = 1 if lpips_is_positive[match[0].item()] else -1
            img = colors.Normalize()(
                np.asarray(estimate_coeff * estimate).reshape(3, height, width).transpose(1, 2, 0))
            ax.imshow(img)
        if save_figure:
            figure_path = os.path.join(save_results, 'estimated_images.png')
            plt.savefig(figure_path, dpi=500)
            if verbose:
                print('The estimated images are saved under {}'.format(figure_path))
        if plot_verbose:
            plt.show()
        else:
            plt.clf()

        fig, axes = plt.subplots(*plot_shape)

        for match, ax in zip(lpips_match, axes.flatten()):
            if return_specific_with_id is not None and match[1].item() == return_specific_with_id:
                ax.spines['bottom'].set_color('red')
                ax.spines['top'].set_color('red')
                ax.spines['right'].set_color('red')
                ax.spines['left'].set_color('red')
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.axis('off')

            estimate = selected_val_batch_data[match[1].item()]
            img = transforms.ToPILImage()(estimate.reshape(3, height, width))
            img = np.asarray(img)
            ax.imshow(img)
        if save_figure:
            figure_path = os.path.join(save_results, 'reference_images.png')
            plt.savefig(figure_path, dpi=500)
            if verbose:
                print('The reference images are saved under {}'.format(figure_path))
        if plot_verbose:
            plt.show()
        else:
            plt.clf()

    if return_specific_with_id is not None:
        if plot_shape is not None:
            fig, axes = plt.subplots(1, 2)
            print_estimate = True
            for ax in axes.flatten():
                ax.axis('off')
                if print_estimate:
                    estimate = estimated_img_batch[best_estimation_id]
                    estimate_coeff = 1 if lpips_is_positive[pos_estimation] else -1
                    img = colors.Normalize()(
                        np.asarray(estimate_coeff * estimate).reshape(3, height, width).transpose(1, 2, 0))
                    ax.imshow(img)
                    print_estimate = False
                else:
                    estimate = selected_val_batch_data[return_specific_with_id]
                    img = transforms.ToPILImage()(estimate.reshape(3, height, width))
                    img = np.asarray(img)
                    ax.imshow(img)

            if save_figure:
                figure_path = os.path.join(save_results,
                                        'target_img_{}_estimation_{}.png'.format(return_specific_with_id,
                                                                                    best_estimation_id))
                plt.savefig(figure_path, dpi=500)
                if verbose:
                    print('The estimated images are saved under {}'.format(figure_path))
            if plot_verbose:
                plt.show()
            else:
                plt.clf()

def cocktail_party_attack(model_config, checkpoint_path, data_type, data_path, batch_size, t_param,
                          total_variance_loss_param, mutual_independence_loss_param,
                          height=32, width=32, random_seed=2024, device_number=0, return_specific_with_id=None, verbose=True, plot_shape=None,
                          save_results=None, save_json=False, save_figure=False, plot_verbose=True, 
                          save_estimations_and_references=False):
    device = 'cuda:{}'.format(device_number) if torch.cuda.is_available() else 'cpu'
    
    ################ LOADING THE PRETRAINED MODEL ######################
    model = _load_pretrained_model(model_config, checkpoint_path, device)
    ####################################################################

    # the all results will be stored in that dictionary    
    result_dict = {}

    # transform images to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # reproducibility
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(random_seed)

    ############ GETTING THE VALIDATION DATA LOADER ###################
    selected_val_batch_data, selected_val_batch_label = _get_reference_image_batch(data_type, data_path, batch_size, 
                                                                                   transform, device, 
                                                                                   save_estimations_and_references, result_dict)
    
    if verbose:
        print('reference data batch is loaded and sent to device {}'.format(device))
    #####################################################################
        
    ################ GET WHITENED GRADIENTS ######################
    whitened_gradient = _get_whitened_gradients(model, selected_val_batch_data, selected_val_batch_label, batch_size, device)

    if verbose:
        print('whitened and centered gradient of the first fc layer is received')
    ###############################################################
    
    ################ COCKTAIL PARTY ATTACK ######################
    estimated_img_batch = _main_cocktail_party_attack(selected_val_batch_data, whitened_gradient,
                                                      t_param, total_variance_loss_param, mutual_independence_loss_param,
                                                      height, width, device, save_estimations_and_references, verbose, result_dict)
    if estimated_img_batch == LPIPS_MAX:
        return LPIPS_MAX
    #############################################################

    with torch.no_grad():
        selected_val_batch_data = selected_val_batch_data.detach().to('cpu')
        ################ RETRIEVE METRICS ######################
        _get_metrics(estimated_img_batch, selected_val_batch_data, height, width, return_specific_with_id, result_dict, verbose)
        #########################################################

        if save_results is not None:
            save_results = os.path.join('./attack_results', save_results)
            if not os.path.exists(save_results):
                os.makedirs(save_results)

        ################ PLOT RESULTS ######################
                _plot_results(estimated_img_batch, selected_val_batch_data,
                              plot_shape, return_specific_with_id, height, width,
                              save_results, save_figure, result_dict, verbose, plot_verbose)
        ####################################################    
        
        new_dict = copy.deepcopy(result_dict)
        new_dict = _turn_tensors_to_list(new_dict)

        if save_json:
            json_path = os.path.join(save_results, 'results.json')
            with open(json_path, 'w') as fp:
                # turning tensors to list
                json.dump(new_dict, fp, indent=4)
            if verbose:
                print('The results are saved under {}'.format(json_path))

        return result_dict
