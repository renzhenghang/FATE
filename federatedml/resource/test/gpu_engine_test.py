import unittest

from federatedml.secureprotol.fate_paillier import PaillierKeypair
from federatedml.resource.task_mgr import EncryptTask, DecryptTask, TaskManager
from federatedml.util import consts
import random
random.seed(0)

def gen_instance(ins_num):
    return [random.randint(1, 2 ** 16 - 1) for i in range(ins_num)]

class TestGPU(unittest.TestCase):
    def setUp(self):
        self.pub_key, self.priv_key = PaillierKeypair.generate_keypair()
    
    def test_enc(self):
        test_inst = gen_instance(10)
        enc_task = EncryptTask(self.pub_key, test_inst, obfuscate=True, device=consts.GPU)
        enc_res = TaskManager.run_task(enc_task)

        for res in enc_res:
            print(res.ciphertext(be_secure=False))
    
    def test_dec(self):
        test_inst = gen_instance(10)
        enc_task = EncryptTask(self.pub_key, test_inst, obfuscate=True, device=consts.GPU)
        enc_res = TaskManager.run_task(enc_task)

        dec_task = DecryptTask(self.priv_key, self.pub_key, enc_res, device=consts.GPU)
        dec_res = TaskManager.run_task(dec_task)
        print(test_inst)
        print(dec_res)
        assert(test_inst == dec_res)


if __name__ == '__main__':
    unittest.main()