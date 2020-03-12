import random
random.seed(0)
from ..secureprotol import gmpy_math
from .compute_engine import cpu_engine, gpu_engine, fpga_engine
from .paillier_cuda import raw_encrypt_gpu, raw_encrypt_obfs_gpu, raw_decrypt_gpu
from ..secureprotol.fixedpoint import FixedPointNumber
from ..secureprotol.fate_paillier import PaillierEncryptedNumber
from ..util import consts 
import numpy as np

class Task:
    """
    describe compute task submitted to the task manager
    """
    def visit(self, engine):
        return engine.accept(self)


class EncryptTask(Task):
    def __init__(self, pub_key, plaintexts, obfuscate=True, device=consts.CPU, precision=None):
        self.pub_key = pub_key
        self.plaintexts = plaintexts
        self.obfuscate = obfuscate
        self.device = device
        self.precision = precision

    def encode(self):
        for plaintext in self.plaintexts:
            if type(plaintext) not in [int, float, np.int16, np.int32, np.int64, np.float16, \
                np.float32, np.float64]:
                raise TypeError("plaintext should be int or float, but got: %s" %
                                type(plaintext))
        return [FixedPointNumber.encode(v, self.pub_key.n, self.pub_key.max_int, self.precision) \
            for v in self.plaintexts]

    def visitCPU(self, engine):
        """
        encryption task on CPU
        """
        encoded_texts = self.encode()
        res = []
        for encoded_elem in encoded_texts:
            plaintext = encoded_elem.encoding
            if plaintext >= (self.pub_key.n - self.pub_key.max_int) and plaintext < self.pub_key.n:
                # Very large plaintext, take a sneaky shortcut using inverses
                neg_plaintext = self.pub_key.n - plaintext  # = abs(plaintext - nsquare)
                neg_ciphertext = (self.pub_key.n * neg_plaintext + 1) % self.pub_key.nsquare
                ciphertext = gmpy_math.invert(neg_ciphertext, self.pub_key.nsquare)
            else:
                ciphertext = (self.pub_key.n * plaintext + 1) % self.pub_key.nsquare

            if self.obfuscate:
                # r = random.SystemRandom().randrange(1, self.pub_key.n)
                r = random.randrange(1, self.pub_key.n)
                obfuscator = gmpy_math.powmod(r, self.pub_key.n, self.pub_key.nsquare)
                ciphertext = ( ciphertext * obfuscator ) % self.pub_key.nsquare
            
            encrypted_text = PaillierEncryptedNumber(self.pub_key, ciphertext, encoded_elem.exponent)
            res.append(encrypted_text)

        return res
    
    def visitGPU(self, engine):
        """
        encryption task on GPU
        """
        encoded_texts = self.encode()

        encoded_list = [v.encoding for v in encoded_texts]
        if self.obfuscate:
            rand_vals = [random.randint(1, 2 ** 32 - 1) for _ in range(len(encoded_list))]
            encrypted_list = raw_encrypt_obfs_gpu(encoded_list, rand_vals)
        else:
            encrypted_list = raw_encrypt_gpu(encoded_list)
        
        paillier_nums = [PaillierEncryptedNumber(self.pub_key, encrypted_list[i], encoded_texts[i].exponent) \
            for i in range(len(encoded_texts))]
        
        return paillier_nums


    def visitFPGA(self, engine):
        raise NotImplementedError("enc task on FPGA not implemented.")

class DecryptTask(Task):
    def __init__(self, priv_key, pub_key, cipher, device=consts.CPU):
        self.priv_key = priv_key
        self.pub_key = pub_key
        self.cipher = cipher
        self.l_func = lambda x, p : (x - 1) // p 
        self.device = device

    def visitCPU(self, engine):
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

    def visitGPU(self, engine):
        """
        decrypt on GPU
        """
        for c in self.cipher:
            if type(c) is not PaillierEncryptedNumber:
                raise TypeError('cipher text type not PaillierEncryptedNumber')
        
        # get raw coding
        # TODO: test if pub key is the same.
        raw_ciphers = [v.ciphertext(be_secure=False) for v in self.cipher]
        raw_dec = raw_decrypt_gpu(raw_ciphers)
        decoder = lambda dec, cipher : FixedPointNumber(dec, cipher.exponent).decode()
        decoded = map(decoder, raw_dec, self.cipher)

        return list(decoded)

    def visitFPGA(self, engine):
        pass

class AddTask(Task):
    def __init__(self, cipher_a, cipher_b, pub_key, device=consts.CPU):
        self.cipher_a = cipher_a
        self.cipher_b = cipher_b
        self.pub_key = pub_key
        self.device = device
    
    def visitCPU(self, engine):
        pass

    def visitGPU(self, engine):
        pass

    def visitFPGA(self, engine):
        pass

class MulTask(Task):
    def __init__(self, cipher_a, cipher_b, pub_key, device=consts.CPU):
        self.cipher_a = cipher_a
        self.cipher_b = cipher_b
        self.pub_key = pub_key
        self.device=device
    
    def visitCPU(self, engine):
        pass

    def visitGPU(self, engine):
        pass

    def visitFPGA(self, engine):
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

class _TaskManager:
    """
    task manager is for submitting the task and checking the status of task, e.g. success or error.
    """

    def __init__(self):
        self.strategy = {
            'fifo': FifoScheduler(self)
        }
        self.res_queue = []
        self.devices = {
            consts.CPU: cpu_engine,
            consts.GPU: gpu_engine,
            consts.FPGA: fpga_engine
        }

    def submit_task(self, task):
        pass

    def set_scheduler(self, strategy='fifo'):
        pass

    def run_task(self, task):
        return task.visit(self.devices[task.device])

TaskManager = _TaskManager()