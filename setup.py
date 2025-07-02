from setuptools import setup

setup(
    name="contour_diffusion",
    py_modules=["contour_diffusion"],
    install_requires=["blobfile>=1.0.5", "torch", "tqdm"],
)
