from .base_indicator import BaseIndicator


class LossIndicator(BaseIndicator):
    """
    Based on the layer-wise loss
    """

    def get_layer_indicator(self) -> float:
        return 0
