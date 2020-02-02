import unittest

from federatedml.secureprotol.fate_paillier import PaillierKeypair
from ..task_mgr import EncryptTask, DecryptTask
from ..compute_engine import cpu_engine as CE, gpu_engine as GE
import random
random.seed(0)

class TestRawEncrypt(unittest.TestCase):
    
    def setUp(self):
        self.public_key, self.private_key = PaillierKeypair.generate_keypair()
    
    def test_cpu(self):
        # test_num = random.SystemRandom(0).randint(0, 2**32)
        test_num = random.randint(0, 2**32)
        enc_task = EncryptTask(self.public_key, test_num)
        res = enc_task.visitCPU(CE)
        res_4 = self.public_key.raw_encrypt(test_num)
        print('debug:',res_4)
        # enc_task.visitGPU(GE)
        print(test_num, res)
        dec_task = DecryptTask(self.private_key, self.public_key, res)
        res_2 = dec_task.visitCPU(CE)
        print(res_2)
        res_3 = self.private_key.raw_decrypt(res)
        print(res_3)

if __name__ == '__main__':
    unittest.main()