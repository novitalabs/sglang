# 修复 pybind11 版本问题

## 问题说明

如果在编译时遇到如下错误:
```
/usr/include/pybind11/detail/type_caster_base.h:482:26: error: invalid use of incomplete type 'PyFrameObject'
```

这是因为系统安装的 pybind11 版本太老 (如 2.9.1),不兼容 Python 3.12。

## 解决方案

### 方法 1: 运行依赖安装脚本 (推荐)

```bash
./install_deps.sh
./build.sh
```

### 方法 2: 手动升级 pybind11

```bash
pip install --upgrade "pybind11>=2.10.0"
./build.sh
```

### 方法 3: 使用虚拟环境 (最佳实践)

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install torch "pybind11>=2.10.0" ninja

# 编译
./build.sh
```

## 版本要求

| 组件 | 最低版本 | 推荐版本 | 说明 |
|------|---------|---------|------|
| Python | 3.8 | 3.12 | Python 3.12 需要 pybind11 >= 2.10 |
| pybind11 | 2.6.0 | 2.10.0+ | **Python 3.12 必须 >= 2.10.0** |
| CMake | 3.14 | 3.20+ | - |
| PyTorch | 2.0 | Latest | - |
| ninja | 1.10 | Latest | 可选,加速编译 |

## 检查当前版本

```bash
python --version
python -c "import pybind11; print('pybind11:', pybind11.__version__)"
python -c "import torch; print('PyTorch:', torch.__version__)"
cmake --version
```

## 为什么会出现这个问题?

Python 3.12 改变了内部 API,将 `PyFrameObject` 结构体变为不透明类型。旧版本的 pybind11 (< 2.10.0) 直接访问了这个结构体的内部字段,导致编译失败。

pybind11 2.10.0+ 版本使用了新的 Python API,兼容 Python 3.12。

## 检查系统 pybind11

如果系统同时安装了多个版本的 pybind11,可能会优先使用系统版本而不是 pip 安装的版本。

检查方法:
```bash
# 检查 pip 安装的版本
pip show pybind11

# 检查 CMake 找到的版本
cmake --find-package -DNAME=pybind11 -DCOMPILER_ID=GNU -DLANGUAGE=CXX -DMODE=VERSION 2>/dev/null || echo "Not found via CMake"

# 检查系统包管理器安装的版本 (Ubuntu/Debian)
dpkg -l | grep pybind11
```

### Ubuntu/Debian 系统

如果系统通过 apt 安装了旧版本的 pybind11:
```bash
# 卸载系统版本
sudo apt remove python3-pybind11

# 安装 pip 版本
pip install "pybind11>=2.10.0"
```

### 强制使用 pip 版本

修改 `setup.py` 或设置环境变量:
```bash
export pybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")
./build.sh
```

## 完整的干净安装流程

```bash
# 1. 创建干净的虚拟环境
python3.12 -m venv ~/venv_sglang
source ~/venv_sglang/bin/activate

# 2. 升级 pip
pip install --upgrade pip

# 3. 安装依赖 (正确的顺序)
pip install torch  # 先安装 PyTorch
pip install "pybind11>=2.10.0"  # 明确要求版本
pip install ninja packaging

# 4. 验证版本
python -c "import pybind11; print('pybind11:', pybind11.__version__); assert pybind11.__version__ >= '2.10.0'"

# 5. 清理旧的编译产物
cd /path/to/suffix_cache
rm -rf build *.so

# 6. 编译
./build.sh
```

## 还是不行?

如果上述方法都不行,尝试:

### 1. 完全卸载 pybind11
```bash
pip uninstall pybind11 -y
sudo apt remove python3-pybind11 -y  # Ubuntu/Debian
pip install "pybind11>=2.10.0"
```

### 2. 显式指定 pybind11 路径
```bash
export pybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")
echo "Using pybind11 from: $pybind11_DIR"
python setup.py build_ext --inplace
```

### 3. 手动编译
```bash
mkdir -p build && cd build
cmake ../csrc/suffix_cache \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE=$(which python) \
    -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")
make -j$(nproc)
cp _C.cpython-*.so ..
cd ..
```

## 联系支持

如果问题仍然存在,请提供以下信息:
```bash
python --version
pip show pybind11
cmake --version
echo $pybind11_DIR
dpkg -l | grep pybind11  # Linux
```
