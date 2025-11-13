# -*- coding: utf-8 -*-
"""
系统中文字体渲染测试（可选）：
- 需要本地安装 Pillow 且系统存在中文字体文件
- 若缺少依赖或字体，测试将跳过
"""
from __future__ import annotations

import os
from pathlib import Path
import unittest


class TestSystemFont(unittest.TestCase):
    def test_render_cn_with_system_font(self):
        try:
            from self_core.encoding.system_font import render_text_to_bitmap, find_chinese_font
        except Exception as e:  # 模块导入失败（极少见）
            self.skipTest(f"system_font module not available: {e}")

        # 尝试导入 Pillow
        try:
            import PIL  # type: ignore
        except Exception:
            self.skipTest("Pillow not installed; skip system-font test. Install via 'pip install pillow'.")

        font_path = find_chinese_font()
        if not font_path or not os.path.exists(font_path):
            self.skipTest("No Chinese system font found on this machine; skipping.")

        bm = render_text_to_bitmap("中文测试", font_path=font_path, size=28, padding=2)
        # 粗检查：非空且存在黑色像素
        self.assertTrue(len(bm) > 0 and len(bm[0]) > 0)
        has_ink = any(px < 255 for row in bm for px in row)
        self.assertTrue(has_ink)

    def test_render_en_with_system_font(self):
        try:
            from self_core.encoding.system_font import render_text_to_bitmap, find_chinese_font
        except Exception as e:
            self.skipTest(f"system_font module not available: {e}")
        try:
            import PIL  # type: ignore
        except Exception:
            self.skipTest("Pillow not installed; skip system-font test. Install via 'pip install pillow'.")
        font_path = find_chinese_font()
        if not font_path or not os.path.exists(font_path):
            self.skipTest("No Chinese system font found on this machine; skipping.")
        bm = render_text_to_bitmap("Hello World", font_path=font_path, size=24, padding=2)
        self.assertTrue(len(bm) > 0 and len(bm[0]) > 0)
        has_ink = any(px < 255 for row in bm for px in row)
        self.assertTrue(has_ink)


if __name__ == "__main__":
    unittest.main()
