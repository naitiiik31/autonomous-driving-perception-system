"""
Autonomous Driving — Car Detection using YOLO
Setup Configuration
"""

from setuptools import setup, find_packages

setup(
    name="yolo-car-detection",
    version="2.0.0",
    author="Deep Learning Project",
    description="Production-grade vehicle detection for autonomous driving using YOLO v2",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "tensorflow>=2.10.0",
        "numpy>=1.21.0",
        "Pillow>=9.0.0",
        "opencv-python>=4.6.0",
        "scipy>=1.7.0",
        "tqdm>=4.64.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.12.0",
        "colorama>=0.4.6",
        "tabulate>=0.9.0",
    ],
    entry_points={
        "console_scripts": [
            "yolo-detect=detect_image:main",
            "yolo-video=detect_video:main",
            "yolo-eval=evaluate:main",
            "yolo-demo=run_demo:main",
            "yolo-train=train:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    keywords="yolo object-detection autonomous-driving deep-learning tensorflow",
)
