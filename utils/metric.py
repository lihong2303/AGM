import torch

def Accuracy(logits,target):
    logits = logits.detach().cpu()
    target = target.detach().cpu()

    preds = logits.argmax(dim=-1)
    assert preds.shape == target.shape
    correct = torch.sum(preds==target)
    total = torch.numel(target)
    if total == 0:
        return 1
    else:
        return correct / total