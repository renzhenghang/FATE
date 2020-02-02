from ..secureprotol import gmpy_math
import random

class engine:

    pass

class CPUEngine(engine):
    """
    common execution of operations on CPU.
    """
    def accept(self, task):
        task.visitCPU(self)


class GPUEngine(engine):
    """
    operations on GPU
    """
    def accept(self, task):
        task.visitGPU(self)


class FPGAEngine(engine):
    """
    operations on FPGA
    """
    def accept(self, task):
        task.visitFPGA(self)

cpu_engine = CPUEngine()
gpu_engine = GPUEngine()
fpga_engine = FPGAEngine()