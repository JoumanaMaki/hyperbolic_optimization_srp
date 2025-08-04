from geoopt.optim import RiemannianAdam

def apply_riemannian_optimizer(model, lr=0.01, weight_decay=5e-4):
    """
    Returns a RiemannianAdam optimizer from Geoopt for the given model parameters.

    Args:
        model (torch.nn.Module): The model whose parameters will be optimized.
        lr (float): Learning rate.
        weight_decay (float): Weight decay (L2 penalty).

    Returns:
        optimizer (RiemannianAdam): The Riemannian optimizer instance.
    """
    return RiemannianAdam(model.parameters(), lr=lr, weight_decay=weight_decay)