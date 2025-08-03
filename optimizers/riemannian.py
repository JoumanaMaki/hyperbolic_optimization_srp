from geoopt.optim import RiemannianAdam

def apply_riemannian_optimizer(model, lr=0.01, weight_decay=5e-4):
    return RiemannianAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
