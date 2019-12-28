import torch

def gradient_quotient(loss, params, eps=1e-5):
    # Calculate g(theta) = gradient of loss
    grad = torch.autograd.grad(
            loss,
            params,
            retain_graph=True, create_graph=True) # <= required for higher-order gradients

    # Calculate H(theta) * g(theta)
    prod = torch.autograd.grad(
            sum([(g**2).sum() / 2 for g in grad]),
            params,
            retain_graph=True, create_graph=True)

    # Form the gradient quotient (GQ) loss as in eq (1)
    out = sum([((g - p) / (g + eps * (2*(g >= 0).float() - 1).detach())- 1).abs().sum() \
                for g, p in zip(grad, prod)])

    return out / sum([p.data.nelement() for p in params])

def metainit(model, criterion, x_size, y_size, lr=0.1, momentum=0.9, steps=500, eps=1e-5):
    model.eval()

    # Only perform gradient on matrix and tensor parameters, not vector biases
    params = [p for p in model.parameters() if p.requires_grad and len(p.size()) >= 2]
    
    # Exponential moving average of first moment to implement momentum
    memory = [0] * len(params)

    for i in range(steps):
        # Draw input i.i.d. from N[0,1]
        input = torch.Tensor(*x_size).normal_(0, 1) #.cuda()

        # Draw output i.i.d. from uniform over integers
        target = torch.randint(0, y_size, (x_size[0],)) #.cuda()

        # Calculate learning and meta-learning losses
        loss = criterion(model(input), target)
        gq = gradient_quotient(loss, list(model.parameters()), eps)
        
        # For each parameter matrix/tensor, perform step of signSGD
        grad = torch.autograd.grad(gq, params)
        for j, (p, g_all) in enumerate(zip(params, grad)):
            # Calculate L_2/Frobenius norm of parameter
            norm = p.data.norm().item()

            # d(GQ)/d(norm of parameter)
            g = torch.sign((p.data * g_all).sum() / norm)

            # signSGD update rule
            memory[j] = momentum * memory[j] - lr * g.item()
            new_norm = norm + memory[j]
            
            # Update norm of parameter
            p.data.mul_(new_norm / norm)
        
        print("%d/GQ = %.2f" % (i, gq.item()))
