# -*- coding: utf-8 -*-
"""
临界性统计与谱半径代理
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
import unittest


class TestCriticality(unittest.TestCase):
    def test_freeplay_criticality(self):
        ts = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        run = Path("runs") / ts
        subprocess.check_call([sys.executable, "scripts/run_freeplay.py", "--config", "config/emergence.toml", "--run", str(run)])
        subprocess.check_call([sys.executable, "scripts/crit_test.py", "--run", str(run)])
        subprocess.check_call([sys.executable, "scripts/eig_monitor.py", "--run", str(run)])
        arr = [json.loads(x) for x in (run / "freeplay.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertTrue(arr)
        bh = arr[-1]["branch_hat"]
        self.assertTrue(0.8 <= bh <= 1.2)
        eig = json.loads((run / "eig.json").read_text(encoding="utf-8"))
        self.assertIn("lambda_max", eig)


if __name__ == "__main__":
    unittest.main()

