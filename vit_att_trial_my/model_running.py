import torch
from torch.autograd import Variable
import wandb
import os
import time



def Train(criterion, device, label_names, model, optimizer, train_loader, val_loader, epochs, test_loader):
    iter = 0
    for epoch in range(epochs):
        t0 = time.time()
        print(f'epoch: {epoch}')
        for i, (images, labels) in enumerate(train_loader):
            # if iter == 501:
            #     break
            images = Variable(images).to(device)
            labels = Variable(labels).to(device)

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Forward pass to get output/logits
            outputs = model(images)

            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, labels)
            # Getting gradients w.r.t. parameters
            loss.backward()
            # Updating parameters
            optimizer.step()

            iter += 1
            if iter % 50 == 0:
                print(f'iter : {iter}')
                print(loss)
                wandb.log({"loss": loss, "epoch": epoch})
            if iter % 500 == 0:
                Validation(device, iter, label_names, loss, model, val_loader)
        t1 = time.time()

        time_per_epoch = t1 - t0
        wandb.log({"time_per_epoch": time_per_epoch})
        # save model:
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, f'model_epoch_{epoch}_.pt'))

        #########################################################################################################
        #                                                 TESTING                                               #
        #########################################################################################################
        print("TESTING TIMZZZ")

        Testing(device, label_names, model, test_loader)


def Validation(device, iter, label_names, loss, model, val_loader):
    # Calculate Accuracy
    correct = 0.0
    correct_arr = [0.0] * 10
    total = 0.0
    total_arr = [0.0] * 10
    # Iterate through test dataset
    with torch.no_grad():
        for images, labels in val_loader:
            images = Variable(images).to(device)
            labels = labels.to(device)
            # Forward pass only to get logits/output
            outputs = model(images)

            # Get predictions from the maximum value
            _, predicted = torch.max(outputs.data, 1)

            # Total number of labels
            total += labels.size(0)
            correct += (predicted == labels).sum()

            for label in range(4):
                correct_arr[label] += (((predicted == labels) & (labels == label)).sum())
                total_arr[label] += (labels == label).sum()

        accuracy = correct / total

        metrics = {'val accuracy': accuracy}
        for label in range(4):
            metrics['Val Accuracy ' + label_names[label]] = correct_arr[label] / total_arr[label]

        wandb.log(metrics)

        # wandb.log({"conf_mat": wandb.plot.confusion_matrix(probs=None,
        # y_true = ground_truth, preds = predictions,
        #                                class_names = class_names)})

        # Print Loss
        print('Iteration: {0} Loss: {1:.2f} Accuracy: {2:.2f}'.format(iter, loss, accuracy))


def Testing(device, label_names, model, test_loader):
    correct = 0.0
    correct_arr = [0.0] * 10
    total = 0.0
    total_arr = [0.0] * 10
    predictions = None
    ground_truth = None
    # Iterate through test dataset
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = Variable(images).to(device)
            labels = labels.to(device)
            # Forward pass only to get logits/output
            outputs = model(images)

            # Get predictions from the maximum value
            _, predicted = torch.max(outputs.data, 1)

            # Total number of labels
            total += labels.size(0)
            correct += (predicted == labels).sum()

            for label in range(4):
                correct_arr[label] += (((predicted == labels) & (labels == label)).sum())
                total_arr[label] += (labels == label).sum()

            if i == 0:
                predictions = predicted
                ground_truth = labels
            else:
                predictions = torch.cat((predictions, predicted), 0)
                ground_truth = torch.cat((ground_truth, labels), 0)
        accuracy = correct / total

        metrics = {'Test Accuracy': accuracy}
        for label in range(4):
            metrics['Test Accuracy ' + label_names[label]] = correct_arr[label] / total_arr[label]
        wandb.log(metrics)
        # wandb.log({"conf_mat": wandb.plot.confusion_matrix(probs=None,
        #                                                    y_true=ground_truth, preds=predictions,
        #                                                    class_names=label_names)})
