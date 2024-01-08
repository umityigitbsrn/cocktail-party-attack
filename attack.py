# Now, there should be two kind of experiment structures one of them for attacking data with deafult data loader setup
# the other one is the data with a dict structure, the below implementation is a generalized version of these two

import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from data import load_mnist_dataloaders, load_cifar10_dataloaders, load_cifar100_dataloaders, load_tiny_imagenet_dataloaders, WhiteningTransformation
from metrics import psnr, lpips_gpu, lpips_matching_specific_image_gpu
from model import Network
from criterion import ReconstructImageFromFCLoss

# TODO: for plotting add a save_path argument maybe for the parameters as well
def cocktail_party_attack(model_config, checkpoint_path, data_type, data_path, batch_size, t_param, total_variance_loss_param, mutual_independence_loss_param, 
                          height=32, width=32, random_seed=2024, device_number=0, return_metrics=True, return_matches=True, return_specific_with_id=None, verbose=True, plot_shape=None):
    # load to model
    device = 'cuda:{}'.format(device_number) if torch.cuda.is_available() else 'cpu'
    model = Network(model_config)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)

    # get val loader
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # reproducibility
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(random_seed)
    if data_type == 'mnist':
        _, val_dataloader = load_mnist_dataloaders(data_path, batch_size, transform)
    elif data_type == 'cifar10':
        _, val_dataloader = load_cifar10_dataloaders(data_path, batch_size, transform)
    elif data_type == 'cifar100':
        _, val_dataloader = load_cifar100_dataloaders(data_path, batch_size, transform)
    elif data_type == 'tiny-imagenet':
        _, val_dataloader = load_tiny_imagenet_dataloaders(batch_size)
    
    if data_type == 'tiny-imagenet':
        data_dict = next(iter(val_dataloader))
        selected_val_batch_data, selected_val_batch_label = data_dict['image'], data_dict['label']
    else:
        selected_val_batch_data, selected_val_batch_label = next(iter(val_dataloader))
    result_dict = {'reference_img_batch' : selected_val_batch_data.detach().to('cpu')}
    selected_val_batch_data = selected_val_batch_data.to(device)
    selected_val_batch_label = selected_val_batch_label.to(device)

    if verbose:
        print('reference data batch is loaded and sent to device {}'.format(device))
    
    # receiving gradients
    model.zero_grad()
    criterion = nn.CrossEntropyLoss()
    output = model(selected_val_batch_data.reshape(batch_size, -1))
    loss = criterion(output, selected_val_batch_label)
    loss.backward()
    gradient_of_layers = []
    for param in model.parameters():
        gradient_of_layers.append(param.grad.data.clone().to('cpu'))
    if verbose:
        gradient_sizes = [x.size() for x in gradient_of_layers]
        print('gradients with size: {} {} received'.format(gradient_sizes, 'are' if len(gradient_sizes) > 1 else 'is'))
    
    whitening_transform = WhiteningTransformation()
    whitened_gradient = torch.from_numpy(whitening_transform.transform(gradient_of_layers[0].detach().numpy().T)).to(torch.float32).T
    whitened_gradient = whitened_gradient.to(device)

    if verbose:
        print('whitened and centered gradient of the first fc layer is received')    
    
    if verbose:
        print('############# ATTACK IS STARTED #############')

    # criterion output testing
    unmixing_matrix = torch.rand((selected_val_batch_data.size(0), gradient_of_layers[0].size(0)), requires_grad=True, device=device)
    reconstruction_loss = ReconstructImageFromFCLoss(height, width, 3, t_param, total_variance_loss_param, mutual_independence_loss_param)
    optimizer = torch.optim.Adam([unmixing_matrix])
    
    for iter_idx in range(25000):
        optimizer.zero_grad()
        # out_score, non_gaussianity_score, total_variance_score, mutual_independence_score
        loss, _, _, _ = reconstruction_loss(unmixing_matrix, whitened_gradient)
        loss.backward()
        optimizer.step()

        # TODO: reconsider this part
        # if torch.isinf(loss).item() or torch.isnan(loss).item():
        #     return 100000000000
        
        if verbose:
            if (iter_idx + 1) % 1000 == 0 or iter_idx == 0:
                print('loss: {}'.format(loss.item()))
            
    unmixing_matrix = unmixing_matrix.detach().to('cpu')
    whitened_gradient = whitened_gradient.detach().to('cpu')
    estimated_img_batch = unmixing_matrix @ whitened_gradient 
    
    result_dict['estimated_img_batch'] = estimated_img_batch

    if verbose:
        print('############# ATTACK IS FINISHED #############')
     
    with torch.no_grad():
        if return_metrics:
            if return_matches:
                lpips_match, lpips_is_positive, mean_lpips = lpips_gpu(estimated_img_batch, selected_val_batch_data.detach().to('cpu'), return_matches=return_matches, height=height, width=width)
                psnr_match, psnr_is_positive, mean_psnr = psnr(estimated_img_batch, selected_val_batch_data.detach().to('cpu'), return_matches=return_matches, height=height, width=width)
                result_dict['lpips'] = {'matches' : lpips_match, 'is_positive' : lpips_is_positive, 'mean_lpips' : mean_lpips}
                result_dict['psnr'] = {'matches' : psnr_match, 'is_positive' : psnr_is_positive, 'mean_psnr' : mean_psnr}
            else:
                lpips_is_positive, mean_lpips = lpips_gpu(estimated_img_batch, selected_val_batch_data.detach().to('cpu'), return_matches=return_matches, height=height, width=width)
                psnr_is_positive, mean_psnr = psnr(estimated_img_batch, selected_val_batch_data.detach().to('cpu'), return_matches=return_matches, height=height, width=width)
                result_dict['lpips'] = {'is_positive' : lpips_is_positive, 'mean_lpips' : mean_lpips}
                result_dict['psnr'] =  {'is_positive' : psnr_is_positive, 'mean_psnr' : mean_psnr}
            if verbose:
                print('LPIPS ({}) and PSNR ({}) values are calculated and added to list of results'.format(mean_lpips, mean_psnr))
        
        if return_specific_with_id is not None:
            best_estimation_id, pos_estimation, min_lpips_value = lpips_matching_specific_image_gpu(estimated_img_batch, selected_val_batch_data.detach().to('cpu')[return_specific_with_id], height=height, width=width)
            result_dict['lpips_with_id'] = {'best_estimation' : best_estimation_id, 'is_positive' : pos_estimation, 'lpips' : min_lpips_value}
            if verbose:
                print('The best estimation for the reference image with id {} is {} {} with LPIPS score: {}'.format(return_specific_with_id, 'positive' if pos_estimation else 'negative', best_estimation_id, min_lpips_value))

        # plotting estimated and reference images 
        if return_matches:
            if plot_shape is not None:    
                fig, axes = plt.subplots(*plot_shape)
                
                for match, ax in zip(lpips_match, axes.flatten()):
                    if match[0].item() == best_estimation_id:
                        ax.spines['bottom'].set_color('red')
                        ax.spines['top'].set_color('red') 
                        ax.spines['right'].set_color('red')
                        ax.spines['left'].set_color('red')
                        ax.set_xticks([])
                        ax.set_yticks([])

                    estimate = estimated_img_batch[match[0].item()]
                    estimate_coeff = 1 if lpips_is_positive[match[0].item()] else -1
                    img = colors.Normalize()(np.asarray(estimate_coeff * estimate).reshape(3, height, width).transpose(1, 2, 0))
                    ax.imshow(img)
                plt.show()

                fig, axes = plt.subplots(*plot_shape)
        
                for match, ax in zip(lpips_match, axes.flatten()):
                    if match[1].item() == return_specific_with_id:
                        ax.spines['bottom'].set_color('red')
                        ax.spines['top'].set_color('red') 
                        ax.spines['right'].set_color('red')
                        ax.spines['left'].set_color('red')
                        ax.set_xticks([])
                        ax.set_yticks([])

                    estimate = selected_val_batch_data[match[1].item()]
                    img = transforms.ToPILImage()(estimate.reshape(3, height, width))
                    img = np.asarray(img)
                    ax.imshow(img)
                plt.show()

    return result_dict