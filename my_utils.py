import torch
from torch import nn

def get_accuracy(preds, label):
    class_preds = torch.argmax(preds, dim=1)
    correct = torch.eq(class_preds, label).float()
    acc = correct.sum()
    return acc

def classifier_inference(model, dataloader, device, testing=False):
    model.eval()

    if testing: 
        result = []
    else:
        inference_acc = 0.0

    num_data = 0.0
    with torch.no_grad():
        for key, (x_batch, y_batch) in enumerate(dataloader):
            num_data += y_batch.shape[0]
            x_batch = x_batch.to(device)
            y_batch = y_batch.long().to(device)
            preds = model(x_batch)
            if testing:
                result.extend(torch.argmax(nn.functional.softmax(preds, dim=1), dim=1).cpu().numpy())
            else:
                inference_acc += get_accuracy(preds, y_batch.squeeze_()).item()
    

    if testing:
        return result
    else:
        return inference_acc / num_data
    
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.00)
    