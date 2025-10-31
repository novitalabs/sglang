# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py as _build_py

# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}


class CMakeExtension(Extension):

    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):

    def copy_extensions_to_source(self):
        """Override to prevent the default copy behavior.

        We handle the copy manually in build_extension() to ensure correct paths.
        """
        # Skip the default copy behavior - we do it manually
        pass

    def build_extension(self, ext: CMakeExtension) -> None:
        # Must be in this form due to bug in .resolve() only fixed in Python 3.10+
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        # When building in-place, copy to current directory instead of nested path
        if self.inplace:
            # For --inplace builds, output directly to the source directory
            extdir = Path(ext.sourcedir).resolve().parent

        # Using this requires trailing slash for auto-detection & inclusion of
        # auxiliary "native" libs

        debug = int(os.environ.get("DEBUG",
                                   0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
        # from Python.

        # Force CMake to use pybind11 from pip (not system installation)
        # This is critical for Python 3.12 compatibility (requires pybind11 >= 2.10.0)
        try:
            import pybind11
            pybind11_cmake_dir = pybind11.get_cmake_dir()
            os.environ["pybind11_DIR"] = pybind11_cmake_dir
            print(f"Using pybind11 from pip: {pybind11_cmake_dir} (version {pybind11.__version__})")
        except Exception as e:
            print(f"Warning: Could not set pybind11_DIR: {e}")

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",  # not used on MSVC, but no harm
            f"-DTORCH_CMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}",
        ]
        build_args = []
        # Adding CMake arguments set as environment variable
        # (needed e.g. to build for ARM OSx on conda-forge)
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [
                item for item in os.environ["CMAKE_ARGS"].split(" ") if item
            ]

        # In this example, we pass in the version to C++. You might not need to.
        cmake_args += [
            f"-DEXAMPLE_VERSION_INFO={self.distribution.get_version()}"
        ]

        if self.compiler.compiler_type != "msvc":
            # Using Ninja-build since it a) is available as a wheel and b)
            # multithreads automatically. MSVC would require all variables be
            # exported for Ninja to pick it up, which is a little tricky to do.
            # Users can override the generator with CMAKE_GENERATOR in CMake
            # 3.15+.
            if not cmake_generator or cmake_generator == "Ninja":
                try:
                    import ninja

                    ninja_executable_path = Path(ninja.BIN_DIR) / "ninja"
                    cmake_args += [
                        "-GNinja",
                        f"-DCMAKE_MAKE_PROGRAM:FILEPATH={ninja_executable_path}",
                    ]
                except ImportError:
                    pass

        else:
            # Single config generators are handled "normally"
            single_config = any(x in cmake_generator
                                for x in {"NMake", "Ninja"})

            # CMake allows an arch-in-generator style for backward compatibility
            contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})

            # Specify the arch if using MSVC generator, but only if it doesn't
            # contain a backward-compatibility arch spec already in the
            # generator name.
            if not single_config and not contains_arch:
                cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]

            # Multi-config generators have a different way to specify configs
            if not single_config:
                cmake_args += [
                    f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"
                ]
                build_args += ["--config", cfg]

        if sys.platform.startswith("darwin"):
            # Cross-compile support for macOS - respect ARCHFLAGS if set
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += [
                    "-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))
                ]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += [f"-j{self.parallel}"]

        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        subprocess.run(["cmake", ext.sourcedir, *cmake_args],
                       cwd=build_temp,
                       check=True)
        subprocess.run(["cmake", "--build", ".", *build_args],
                       cwd=build_temp,
                       check=True)

        # Always manually copy the .so file to the source directory for inplace builds
        # The built file is in build/lib.*/sglang/srt/speculative/suffix_cache/_C.*.so
        print(f"[DEBUG] Looking for built .so file...")
        print(f"[DEBUG] build_lib: {self.build_lib}")
        print(f"[DEBUG] build_temp: {self.build_temp}")
        print(f"[DEBUG] inplace: {self.inplace}")

        # Search for the built .so file
        build_dir = Path(self.build_temp).parent.parent
        built_so_files = list(build_dir.rglob("_C.cpython-*.so"))
        print(f"[DEBUG] Found .so files: {built_so_files}")

        if built_so_files:
            src_file = built_so_files[0]
            # Copy to current directory (where setup.py is)
            dst_file = Path.cwd() / src_file.name
            print(f"[INFO] Copying {src_file} -> {dst_file}")
            shutil.copy2(src_file, dst_file)
            print(f"[INFO] ✓ Copy completed! File exists: {dst_file.exists()}")
        else:
            print(f"[ERROR] Could not find built .so file")
            print(f"[DEBUG] Searched in: {build_dir}")


class CompileGrpc(_build_py):
    """Custom build command to compile .proto files before building."""

    def run(self):
        import os
        import sys
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from sglang.srt.speculative.suffix_cache.embedding.generate_proto import generate_grpc_code
        generate_grpc_code()
        # Run the original build_py command
        _build_py.run(self)

setup(
    ext_modules=[
        CMakeExtension("sglang.srt.speculative.suffix_cache._C",
                       "csrc/suffix_cache"),
    ],
    cmdclass={"build_ext": CMakeBuild, 'build_py': CompileGrpc},
)
