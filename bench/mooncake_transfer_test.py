import unittest
import socket

import torch

from mooncake.engine import TransferEngine


def local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return socket.gethostbyname(socket.gethostname())


def split_and_get(value: int, size: int, splits: int = 4) -> list[int]:
    return [value if i < 2 else value + 1 for i in range(splits)]


class TestMooncakeTransfer(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("No CUDA device")

        cls.protocol = "nvlink_intra"
        cls.device = torch.device("cuda:0")
        cls.size = 64 * 1024 * 1024

        cls.engine = TransferEngine()
        local_host = local_ip()
        cls.ret = cls.engine.initialize(local_host, "P2PHANDSHAKE", cls.protocol, "")
        if cls.ret != 0:
            raise unittest.SkipTest(f"Mooncake initialize failed: {cls.ret}")
        cls.target_name = f"{local_host}:{cls.engine.get_rpc_port()}"

        cls.src = torch.empty(cls.size, dtype=torch.uint8, device=cls.device)
        cls.dst = torch.zeros(cls.size, dtype=torch.uint8, device=cls.device)

        registrations = [
            cls.engine.register_memory(cls.src.data_ptr(), cls.src.nbytes),
            cls.engine.register_memory(cls.dst.data_ptr(), cls.dst.nbytes),
        ]
        if any(ret != 0 for ret in registrations):
            raise RuntimeError("Mooncake memory registration failed")
        cls.remote_addr = cls.dst.data_ptr()

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "engine") and hasattr(cls, "engine"):
            cls.engine.unregister_memory(cls.src.data_ptr())
            cls.engine.unregister_memory(cls.dst.data_ptr())

    def test_transfer_write(self) -> None:
        for value in (17, 19):
            self.src.fill_(value)
            self.dst.zero_()
            ret = self.engine.transfer_sync_write(
                self.target_name,
                self.src.data_ptr(),
                self.remote_addr,
                self.size,
            )
            self.assertEqual(ret, 0)
            self.assertTrue(torch.all(self.dst == value))

    def test_transfer_read(self) -> None:
        for value in (23, 29):
            self.dst.fill_(value)
            self.src.zero_()
            ret = self.engine.transfer_sync_read(
                self.target_name,
                self.src.data_ptr(),
                self.remote_addr,
                self.size,
            )
            self.assertEqual(ret, 0)
            self.assertTrue(torch.all(self.src == value))

    def test_batch_transfer_write(self) -> None:
        ret = self.engine.batch_transfer_sync_write(
            self.target_name,
            [self.src.data_ptr(), self.src.data_ptr() + self.size // 2],
            [self.remote_addr, self.remote_addr + self.size // 2],
            [self.size // 2, self.size // 2],
        )
        self.assertEqual(ret, 0)
        torch.cuda.synchronize(self.device)

    def test_batch_transfer_read(self) -> None:
        ret = self.engine.batch_transfer_sync_read(
            self.target_name,
            [self.src.data_ptr(), self.src.data_ptr() + self.size // 2],
            [self.remote_addr, self.remote_addr + self.size // 2],
            [self.size // 2, self.size // 2],
        )
        self.assertEqual(ret, 0)
        torch.cuda.synchronize(self.device)


if __name__ == "__main__":
    unittest.main()
