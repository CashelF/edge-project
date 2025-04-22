from models.conv5 import (
    Conv1_small_BN,
    Conv2_small_BN,
    Conv3_small_BN,
    Conv5,
    Conv5_small,
    Conv5_small_BN,
)
from models.dws_conv5 import Conv3_small_BN_DWS, Conv5_small_BN_DWS


def get_model(model_name, loss_type="fedavg"):
    if model_name == "conv5":
        return Conv5(loss_type=loss_type)
    elif model_name == "conv5small":
        return Conv5_small(loss_type=loss_type)
    elif model_name == "conv5smallBN":
        return Conv5_small_BN(loss_type=loss_type)
    elif model_name == "conv3smallBN":
        return Conv3_small_BN(loss_type=loss_type)
    elif model_name == "conv2smallBN":
        return Conv2_small_BN(loss_type=loss_type)
    elif model_name == "conv1smallBN":
        return Conv1_small_BN(loss_type=loss_type)
    elif model_name == "conv5smallBN_DWS":
        return Conv5_small_BN_DWS(loss_type=loss_type)
    elif model_name == "conv3smallBN_DWS":
        return Conv3_small_BN_DWS(loss_type=loss_type)
    else:
        raise NotImplementedError(f"[!] ERROR: Model {model_name} not implemented yet")
