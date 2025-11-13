# -*- coding: utf-8 -*-
"""
贯通小用例：
- 生成数据 -> 构建数据集 -> 训练若干步 -> 评测
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
import unittest


class TestMinimal(unittest.TestCase):
    def test_pipeline(self):
        # 生成
        subprocess.check_call([sys.executable, "scripts/gen_text_images.py",
                               "--concepts", "examples/concepts_small.jsonl",
                               "--out", "data/raw/text"])
        subprocess.check_call([sys.executable, "scripts/gen_semantic_images.py",
                               "--concepts", "examples/concepts_small.jsonl",
                               "--out", "data/raw/semantic"])
        subprocess.check_call([sys.executable, "scripts/build_dataset.py",
                               "--in", "data/raw", "--out", "data/processed",
                               "--val_ratio", "0.1", "--test_ratio", "0.1"])
        # 训练
        ts = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        run = Path("runs") / ts
        subprocess.check_call([sys.executable, "scripts/train.py", "--config", "config/default.toml", "--run", str(run)])
        # 评测
        subprocess.check_call([sys.executable, "scripts/eval.py", "--run", str(run), "--config", "config/default.toml"])
        # 断言
        self.assertTrue((run / "logs.jsonl").exists())
        self.assertTrue((run / "eval.json").exists())
        m = json.loads((run / "eval.json").read_text(encoding="utf-8"))
        self.assertIn("retrieval", m)
        self.assertGreaterEqual(m["retrieval"]["R@1"], 0.0)
        self.assertGreaterEqual(m["energy"]["steps"], 0)


if __name__ == "__main__":
    unittest.main()

