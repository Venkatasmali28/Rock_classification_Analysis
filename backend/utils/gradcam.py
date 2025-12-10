import torch
import torch.nn.functional as F

def generate_gradcam(model, input_tensor, target_class):
    # Find last Conv2d layer
    conv_layers = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    last_conv = list(model.named_modules())[-1][1]
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)
    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    fh = last_conv.register_forward_hook(forward_hook)
    bh = last_conv.register_full_backward_hook(backward_hook)

    output = model(input_tensor)
    score = output[0, target_class]
    model.zero_grad()
    score.backward(retain_graph=True)

    fh.remove()
    bh.remove()

    grad = gradients[0].cpu().data
    act = activations[0].cpu().data
    weights = grad.mean(dim=[2, 3], keepdim=True)
    cam = F.relu((weights * act).sum(1)).squeeze()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam
