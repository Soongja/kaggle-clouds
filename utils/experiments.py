"""
Reference:
- https://github.com/khornlund/severstal-steel-defect-detection/blob/master/sever/model/loss.py
"""


class LabelSmoother:
    """
    Maps binary labels (0, 1) to (eps, 1 - eps)
    """
    def __init__(self, eps=1e-4):
        self.eps = eps
        self.scale = 1 - 2 * self.eps
        self.bias = self.eps / self.scale

    def __call__(self, t):
        return (t + self.bias) * self.scale
