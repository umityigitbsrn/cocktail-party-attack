import torch
import torch.nn as nn

from model import Network
import torch.optim as optim
import os


def train_one_epoch(model, optimizer, criterion, train_loader, val_loader, device, save_path, save_interval, linearize,
                    curr_epoch):
    prev_val_accuracy = 0
    if save_path:
        if os.path.exists(save_path):
            print('Restoring model...')
            checkpoint = torch.load(save_path)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            prev_val_accuracy = checkpoint['final_accuracy']

    model.train()
    running_train_loss = []
    running_val_loss = []
    running_val_accuracy = []

    for iter_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        if linearize:
            images = images.reshape(images.shape[0], -1)
        labels = labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_train_loss.append(loss.item())
        if save_path:
            if (iter_idx + 1) % save_interval == 0:
                model.eval()
                curr_accuracy = 0
                curr_loss = 0
                with torch.no_grad():
                    for val_images, val_labels in val_loader:
                        val_images = val_images.to(device)
                        if linearize:
                            val_images = val_images.reshape(val_images.shape[0], -1)
                        val_labels = val_labels.to(device)
                        outputs = model(val_images)
                        val_loss = criterion(outputs, val_labels)
                        curr_accuracy += torch.count_nonzero(torch.argmax(outputs, dim=1) == val_labels).item()
                        curr_loss += val_loss.item()

                    curr_loss = curr_loss / len(val_loader)
                    curr_accuracy = curr_accuracy / len(val_loader)
                    running_val_loss.append(curr_loss)
                    running_val_accuracy.append(curr_accuracy)
                    if prev_val_accuracy < curr_accuracy:
                        print('Saving model...')
                        prev_val_accuracy = curr_accuracy
                        torch.save({
                            'state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'final_accuracy': curr_accuracy
                        }, save_path)
                print('epoch: {}, iter: {}/{}, train_loss: {:.4f}, val_loss: {:.4f}, val_accuracy: {:.4f}'.format(
                    curr_epoch + 1, iter_idx + 1, len(train_loader), running_train_loss[-1], running_val_loss[-1],
                    running_val_accuracy[-1]))
                model.train()
        else:
            if (iter_idx + 1) % save_interval == 0:
                print('epoch: {}, iter: {}/{}, train_loss: {:.4f}'.format(curr_epoch + 1, iter_idx + 1,
                                                                          len(train_loader), running_train_loss[-1]))

    if len(running_val_accuracy) == 0 and len(running_val_loss) == 0:
        running_val_accuracy, running_val_loss = None, None
    return model, optimizer, running_train_loss, running_val_loss, running_val_accuracy


def train_all(model_conf_path, optimizer_type, lr, criterion_type, num_epochs, train_loader, val_loader=None,
              device=None, save_path=None, save_interval=200, linearize=False):
    # init model
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Network(model_conf_path).to(device)
    model.train()
    criterion = getattr(nn, criterion_type)()
    optimizer = getattr(optim, optimizer_type)(model.parameters(), lr=lr)
    train_loss_arr = []
    val_loss_arr = []
    val_accuracy_arr = []
    for epoch in range(num_epochs):
        model, optimizer, train_loss, val_loss, val_accuracy = train_one_epoch(model, optimizer, criterion,
                                                                               train_loader, val_loader, device,
                                                                               save_path, save_interval, linearize,
                                                                               epoch)
        train_loss_arr.append(train_loss)
        if val_loss is not None:
            val_loss_arr.append(val_loss)

        if val_accuracy is not None:
            val_accuracy_arr.append(val_accuracy)
    if len(val_accuracy_arr) == 0:
        val_accuracy_arr = None

    if len(val_loss_arr) == 0:
        val_loss_arr = None
    if save_path:
        checkpoint = torch.load(save_path)
        checkpoint['train_loss'] = train_loss_arr
        checkpoint['val_loss'] = val_loss_arr
        checkpoint['val_accuracy'] = val_accuracy_arr
        torch.save(checkpoint, save_path)
        print('Training log is saved to {}'.format(save_path))
    return train_loss_arr, val_loss_arr, val_accuracy_arr
