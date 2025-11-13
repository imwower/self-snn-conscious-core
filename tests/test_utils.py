# -*- coding: utf-8 -*-
"""
工具层单测：PGM/PPM 写入、ASCII 绘图、配置加载
"""
from __future__ import annotations

import os
from pathlib import Path
import unittest

from self_core.utils.io import write_pgm, write_ppm, read_config
from self_core.utils.ascii_plot import plot_line, plot_bar, plot_heat


class TestUtils(unittest.TestCase):
    def test_images_and_plots(self):
        p1 = Path("tmp_a.pgm")
        p2 = Path("tmp_b.ppm")
        img = [[0 if (i+j)%2==0 else 255 for i in range(16)] for j in range(16)]
        write_pgm(p1, img)
        rgb = [[[i%256, j%256, (i*j)%256] for i in range(16)] for j in range(16)]
        write_ppm(p2, rgb)
        self.assertTrue(p1.exists() and p1.stat().st_size > 0)
        self.assertTrue(p2.exists() and p2.stat().st_size > 0)
        os.remove(p1)
        os.remove(p2)

        self.assertTrue(plot_line([0,1,2,1,0]))
        self.assertTrue(plot_bar([1,2,3]))
        self.assertTrue(plot_heat([[0,1],[1,0]]))

    def test_config(self):
        cfg = read_config(Path("config/default.toml"))
        self.assertIn("core", cfg)
        self.assertIn("train", cfg)


if __name__ == "__main__":
    unittest.main()

