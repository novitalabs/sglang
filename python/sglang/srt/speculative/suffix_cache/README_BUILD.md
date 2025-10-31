# suffix_cache C++ Extension - 编译指南

本目录包含 suffix_cache C++ 扩展的编译脚本和文档。

## 快速开始 (Quick Start)

```bash
cd /path/to/sglang/python/sglang/srt/speculative/suffix_cache

# 方法1: 使用 build.sh (推荐)
./build.sh

# 方法2: 使用 Makefile
make build

# 方法3: 直接使用 setup.py
python setup.py build_ext --inplace
```

## 编译输出 (Build Output)

编译成功后会生成:
```
_C.cpython-312-x86_64-linux-gnu.so
```

文件名会根据你的 Python 版本和平台而变化:
- **Linux**: `_C.cpython-312-x86_64-linux-gnu.so`
- **macOS**: `_C.cpython-312-darwin.so`
- **Windows**: `_C.cp312-win_amd64.pyd`

## 提供的脚本和工具 (Available Scripts)

### 1. build.sh - 自动化编译脚本
```bash
./build.sh
```
- 自动检查依赖
- 清理旧的编译产物
- 执行编译
- 显示生成的 .so 文件位置

### 2. Makefile - Make 构建系统
```bash
make help        # 显示帮助信息
make build       # 编译扩展
make clean       # 清理编译产物
make test        # 测试扩展
make install     # 开发模式安装
make deps        # 安装依赖
make rebuild     # 清理后重新编译
```

### 3. test_extension.py - 测试脚本
```bash
python test_extension.py
```
运行完整的测试套件,验证扩展功能是否正常。

## 依赖要求 (Prerequisites)

编译前需要安装:

1. **Python 3.12** (或兼容版本)
2. **PyTorch**
3. **pybind11**
4. **CMake** >= 3.14
5. **C++ 编译器** 支持 C++17 (g++, clang, 或 MSVC)
6. **ninja** (可选，用于加速编译)

安装依赖:
```bash
pip install torch pybind11 ninja
# 或
make deps
```

## 验证编译结果 (Verification)

### 快速测试
```bash
python -c "from sglang.srt.speculative.suffix_cache._C import SuffixTree; print('Import successful!')"
```

### 完整测试
```bash
python test_extension.py
# 或
make test
```

### 功能测试示例
```python
from sglang.srt.speculative.suffix_cache._C import SuffixTree, Candidate

# 创建 suffix tree,最大深度为 64
tree = SuffixTree(64)

# 向序列 0 添加 tokens
tree.extend(0, [1, 2, 3, 4, 5])

# 基于模式 [3, 4] 进行推测
candidate = tree.speculate(
    pattern=[3, 4],
    max_spec_tokens=8,
    max_spec_factor=1.0,
    max_spec_offset=0.0,
    min_token_prob=0.1,
    use_tree_spec=False
)

print(f"推测的 tokens: {candidate.token_ids}")
print(f"匹配长度: {candidate.match_len}")
print(f"得分: {candidate.score}")
```

## 故障排查 (Troubleshooting)

### CMake 未找到
```bash
pip install cmake
# 或在 Ubuntu/Debian:
sudo apt-get install cmake
```

### pybind11 未找到
```bash
pip install pybind11
```

### C++17 编译错误
确保你的编译器支持 C++17:
- GCC >= 7.0
- Clang >= 5.0
- MSVC >= 19.14 (Visual Studio 2017 15.7)

### .so 文件名中的 Python 版本不对
脚本使用当前激活的 Python。确保你在正确的虚拟环境中:
```bash
which python
python --version
```

### 清理编译产物
```bash
make clean
# 或
rm -rf build *.so
./build.sh
```

## 项目结构 (Project Structure)

```
suffix_cache/
├── csrc/suffix_cache/          # C++ 源代码
│   ├── CMakeLists.txt         # CMake 配置
│   ├── pybind.cc              # Python 绑定
│   ├── suffix_tree.cc         # 核心实现
│   └── suffix_tree.h          # 头文件
├── setup.py                    # Python 构建配置
├── build.sh                    # 自动编译脚本 (新增)
├── Makefile                    # Make 构建文件 (新增)
├── test_extension.py           # 测试脚本 (新增)
├── BUILD.md                    # 详细构建文档 (新增)
└── README_BUILD.md            # 本文件 (新增)
```

## 构建系统说明 (Build System Overview)

构建系统使用:
- **setup.py**: 使用 setuptools 的 Python 构建协调器
- **CMakeLists.txt**: CMake 构建配置
- **pybind11**: C++/Python 绑定库
- **ninja**: 快速并行构建系统 (可选)

## CI/CD 集成示例

```bash
#!/bin/bash
# 自动化构建脚本示例

# 安装依赖
pip install pybind11 ninja torch

# 构建
cd python/sglang/srt/speculative/suffix_cache
python setup.py build_ext --inplace

# 测试
python test_extension.py
```

## 更多信息 (More Information)

详细的构建说明请参考 [BUILD.md](BUILD.md)

## 常见问题 (FAQ)

**Q: 为什么有三种构建方法?**
A:
- `build.sh`: 最简单,自动化程度最高,推荐给大多数用户
- `Makefile`: 适合喜欢使用 make 的用户
- `setup.py`: 最灵活,适合需要自定义构建参数的场景

**Q: 需要每次修改 Python 代码后重新编译吗?**
A: 不需要。只有 C++ 代码 (csrc/suffix_cache/ 目录下) 修改后才需要重新编译。

**Q: 编译需要多长时间?**
A: 通常 10-30 秒,取决于你的系统性能和是否使用 ninja。

**Q: 可以在虚拟环境中编译吗?**
A: 可以,推荐在虚拟环境中编译以避免污染系统 Python。

**Q: 支持哪些操作系统?**
A: Linux, macOS, Windows (需要 Visual Studio 或 MinGW)

## 许可证 (License)

Copyright 2025 Snowflake Inc.
SPDX-License-Identifier: Apache-2.0
