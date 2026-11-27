# Adapted for the PricePTQ paper release.
def get_predictor(model, inputs, targets, device='cpu', **kwargs):

    if model == 'rbf':
        from priceptq.vendor.amq.predictor.rbf import RBF
        predictor = RBF(**kwargs)
        predictor.fit(inputs, targets)

    else:
        raise ValueError('The paper uses the RBF predictor.')
    return predictor
