class LossMeter:
    """
    Computes and stores the average and current loss value
    """

    def __init__(self):
        """
        Create loss tracker
        """
        self.reset()

    def reset(self):
        """
        Reset loss tracker
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        """
        Update loss tracker
        :param val: Loss value to add to tracker
        :param n: Number of elements contributing to val
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
