#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

if __name__ == "__main__":
    setup(
        entry_points={
            "console_scripts": [
                "qwen25-grpo=qwen25_grpo.main:main",
            ],
        },
    ) 