from torch.cuda import is_available, device_count, get_device_name, current_device


class Cudafy(object):

    def __init__(self, device=None):
        if is_available() and device:
            self.device = device
        else:
            self.device = 0

    @staticmethod
    def check_devices():
        for i in range(device_count()):
            print("Found device {}:".format(i), get_device_name(i))
        if device_count() == 0:
            print("No GPU device found")
        else:
            print("Current cuda device is", get_device_name(current_device()))

    def name(self):
        if is_available():
            return get_device_name(self.device)
        return 'Cuda is not available.'

    def put(self, x):
        """Put x on the default cuda device."""
        if is_available():
            return x.to(device=self.device)
        return x

    def __call__(self, x):
        return self.put(x)

    def get(self, x):
        """Get from cpu."""
        if x.is_cuda:
            return x.to(device='cpu')
        return x