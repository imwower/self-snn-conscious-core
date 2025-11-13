# -*- coding: utf-8 -*-
"""
E/I 平衡与发放分布
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
import unittest


class TestEI(unittest.TestCase):
    def test_ei_probe(self):
        ts = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        run = Path("runs") / ts
        subprocess.check_call([sys.executable, "scripts/run_freeplay.py", "--config", "config/emergence.toml", "--run", str(run)])
        subprocess.check_call([sys.executable, "scripts/eibalance_probe.py", "--run", str(run)])
        ei = json.loads((run / "ei.json").read_text(encoding="utf-8"))
        self.assertIn("e_over_i", ei)
        self.assertIn("rate_hist", ei)


if __name__ == "__main__":
    unittest.main()

