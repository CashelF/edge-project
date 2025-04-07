from models.conv5 import Conv5, Conv5_small, Conv5_small_BN


def get_model(model_name, loss_type='fedavg'):
    if model_name == "conv5":
        return Conv5(loss_type=loss_type)
    elif model_name == "conv5small":
        return Conv5_small(loss_type=loss_type)
    elif model_name == "conv5smallBN":
        return Conv5_small_BN(loss_type=loss_type)
    else:
        raise NotImplementedError(f'[!] ERROR: Model {model_name} not implemented yet')
