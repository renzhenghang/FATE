from ..secureprotol import gmpy_math
import random

class engine:

    pass

class _CPUEngine(engine):
    """
    common execution of operations on CPU.
    """
    def accept(self, task):
        task.visitCPU(self)


class _GPUEngine(engine):
    """
    operations on GPU
    """
    def __init__(self):
        """
        check the status of gpu function
        """
        pass


    def accept(self, task):
        task.visitGPU(self)


class _FPGAEngine(engine):
    """
    operations on FPGA
    """
    def accept(self, task):
        task.visitFPGA(self)

cpu_engine = _CPUEngine()
gpu_engine = _GPUEngine()
fpga_engine = _FPGAEngine()