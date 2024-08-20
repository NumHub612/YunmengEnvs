from setuptools import setup, find_packages


with open("VERSION", "r", encoding="utf8") as info:
    version = info.readline().strip()

with open("requirements.txt", "r", encoding="utf8") as req:
    requirements = [l.strip() for l in req.readlines()]

Packages = find_packages()


setup(
    name="yunmengenvs",
    version=version,
    description="云梦环境流体力学解决方案",
    url="https://github.com/NumHub612/YunmengEnvs",
    license="Apache-2.0",
    python_requires=">=3.10",
    packages=Packages,
    install_requires=requirements,
    include_package_data=True,
)
