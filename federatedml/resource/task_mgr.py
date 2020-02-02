import random
random.seed(0)
from ..secureprotol import gmpy_math
from .compute_engine import CPUEngine, GPUEngine, FPGAEngine

class Task:
    """
    describe compute task submitted to the task manager
    """
    def visit(self, engine):
        pass


class EncryptTask(Task):
    def __init__(self, pub_key, plaintext, obfuscate=True):
        self.pub_key = pub_key
        self.plaintext = plaintext
        self.obfuscate = obfuscate

    def visitCPU(self, engine:CPUEngine):
        if not isinstance(self.plaintext, int):
            raise TypeError("plaintext should be int, but got: %s" %
                            type(self.plaintext))

        if self.plaintext >= (self.pub_key.n - self.pub_key.max_int) and self.plaintext < self.pub_key.n:
            # Very large plaintext, take a sneaky shortcut using inverses
            neg_plaintext = self.pub_key.n - self.plaintext  # = abs(plaintext - nsquare)
            neg_ciphertext = (self.pub_key.n * neg_plaintext + 1) % self.pub_key.nsquare
            ciphertext = gmpy_math.invert(neg_ciphertext, self.pub_key.nsquare)
        else:
            ciphertext = (self.pub_key.n * self.plaintext + 1) % self.pub_key.nsquare

        if self.obfuscate:
            # r = random.SystemRandom().randrange(1, self.pub_key.n)
            r = random.randrange(1, self.pub_key.n)
            obfuscator = gmpy_math.powmod(r, self.pub_key.n, self.pub_key.nsquare)

            ciphertext = ( ciphertext * obfuscator ) % self.pub_key.nsquare

        return ciphertext
    
    def visitGPU(self, engine: GPUEngine):
        print('enc task on GPU not implemented')

    def visitFPGA(self, engine: FPGAEngine):
        print('enc task on FPGA not implemented')

class DecryptTask(Task):
    def __init__(self, priv_key, pub_key, cipher):
        self.priv_key = priv_key
        self.pub_key = pub_key
        self.cipher = cipher
        self.l_func = lambda x, p : (x - 1) // p 

    def visitCPU(self, engine:CPUEngine):
        if not isinstance(self.cipher, int):
            raise TypeError("ciphertext should be an int, not: %s" %
                type(self.cipher))
        
        mp = self.l_func(gmpy_math.powmod(self.cipher,
                                              self.priv_key.p-1, self.priv_key.psquare),
                                              self.priv_key.p) * self.priv_key.hp % self.priv_key.p

        mq = self.l_func(gmpy_math.powmod(self.cipher,
                                              self.priv_key.q-1, self.priv_key.qsquare),
                                              self.priv_key.q) * self.priv_key.hq % self.priv_key.q

        u = (mp - mq) * self.priv_key.q_inverse % self.priv_key.p
        x = (mq + (u * self.priv_key.q)) % self.pub_key.n
        return x

    def visitGPU(self, engine:GPUEngine):
        pass

    def visitFPGA(self, engine:FPGAEngine):
        pass

class AddTask(Task):
    def __init__(self, cipher_a, cipher_b, pub_key):
        self.cipher_a = cipher_a
        self.cipher_b = cipher_b
        self.pub_key = pub_key
    
    def visitCPU(self, engine:CPUEngine):
        pass

    def visitGPU(self, engine:GPUEngine):
        pass

    def visitFPGA(self, engine:FPGAEngine):
        pass

class MulTask(Task):
    def __init__(self, cipher_a, cipher_b, pub_key):
        self.cipher_a = cipher_a
        self.cipher_b = cipher_b
        self.pub_key = pub_key
    
    def visitCPU(self, engine:CPUEngine):
        pass

    def visitGPU(self, engine:GPUEngine):
        pass

    def visitFPGA(self, engine:FPGAEngine):
        pass

class Scheduler:
    """
    scheduler is responsible for picking up a task from task queue according to 
    some kind of priority.
    """
    def __init__(self, task_manager):
        self.exec_thread = None
        self.state = 'Sleeping'
        self.task_manager = task_manager
        self.task_pool = []

class FifoScheduler(Scheduler):

    def prepare(self):
        pass

    def pick_one(self, task_pool):
        pass

    def run(self, task_pool):
        pass

    def stop(self):
        pass

class TaskManager:
    """
    task manager is for submitting the task and checking the status of task, e.g. success or error.
    """

    def __init__(self):
        self.strategy = {
            'fifo': FifoScheduler(self)
        }
        self.res_queue = []

    def submit_task(self, task):
        pass

    def set_scheduler(self, strategy='fifo'):
        pass