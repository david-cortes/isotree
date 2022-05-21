try:
    from setuptools import setup
    from setuptools.extension import Extension
except:
    from distutils.core import setup
    from distutils.extension import Extension

import numpy as np
import ctypes
from Cython.Distutils import build_ext
from sys import platform
import sys, os, subprocess, warnings, re
from os import environ

found_omp = True
def set_omp_false():
    global found_omp
    found_omp = False

## https://stackoverflow.com/questions/724664/python-distutils-how-to-get-a-compiler-that-is-going-to-be-used
class build_ext_subclass( build_ext ):
    def build_extensions(self):
        is_msvc = self.compiler.compiler_type == "msvc"
        is_clang = hasattr(self.compiler, 'compiler_cxx') and ("clang++" in self.compiler.compiler_cxx)
        is_windows = sys.platform[:3] == "win"
        is_mingw = (is_windows and
                    (self.compiler.compiler_type.lower()
                     in ["mingw32", "mingw64", "mingw", "msys", "msys2", "gcc", "g++"]))
        no_ld = "NO_LONG_DOUBLE" in os.environ
        has_robinmap = os.path.exists("src/robinmap/include/tsl")

        if is_msvc:
            for e in self.extensions:
                e.extra_compile_args = ['/openmp', '/O2', '/std:c++14', '/fp:except-', '/wd4244', '/wd4267', '/wd4018', '/wd5030']
                e.define_macros += [("NO_LONG_DOUBLE", None)]
                if has_robinmap:
                    e.define_macros += [("_USE_ROBIN_MAP", None)]
        
        else:
            if not self.check_for_variable_dont_set_march() and not self.is_arch_in_cflags():
                self.add_march_native()
            self.add_openmp_linkage()
            self.add_restrict_qualifier()
            self.add_O3()
            self.add_no_math_errno()
            self.add_no_trapping_math()
            if not is_mingw:
                self.add_highest_supported_cxx_standard()
            if not is_windows:
                self.add_link_time_optimization()

            for e in self.extensions:

                if is_mingw:
                    e.extra_compile_args += ['-std=gnu++14']
                    if np.iinfo(ctypes.c_size_t).max >= (2**64 - 1):
                        e.define_macros += [("_FILE_OFFSET_BITS", 64)]
                
                # ## for testing
                # e.extra_compile_args += ['-ggdb']
                
                if has_robinmap:
                    e.define_macros += [("_USE_ROBIN_MAP", None)]

                if is_windows or no_ld:
                    e.define_macros += [("NO_LONG_DOUBLE", None)]
                
                # ## for testing
                # e.extra_compile_args = ["-std=c++11", "-ggdb"]

                # e.extra_compile_args = ['-fopenmp', '-O3', '-march=native', '-std=c++11']
                # e.extra_link_args    = ['-fopenmp']

                # ## when testing with clang:
                # e.extra_compile_args = ['-fopenmp=libiomp5', '-O3', '-march=native', '-std=c++17']
                # e.extra_link_args    = ['-fopenmp']

                # e.extra_compile_args = ['-O2', '-march=native', '-std=c++11']
                # e.extra_compile_args = ['-O0', '-march=native', '-std=c++11']

                # ## for testing (run with `LD_PRELOAD=libasan.so python script.py`)
                # e.extra_compile_args = ["-std=c++11", "-fsanitize=address", "-static-libasan", "-ggdb"]
                # e.extra_link_args    = ["-fsanitize=address", "-static-libasan"]

                # ## for testing with clang (run with `LD_PRELOAD=libasan.so python script.py`)
                # e.extra_compile_args = ["-std=c++11", "-fsanitize=address", "-static-libsan"]
                # e.extra_link_args    = ["-fsanitize=address", "-static-libsan"]

                # if is_clang:
                #     e.extra_compile_args += ['-fopenmp=libomp']
                #     e.extra_link_args += ['-fopenmp']

        build_ext.build_extensions(self)

    def check_for_variable_dont_set_march(self):
        return "DONT_SET_MARCH" in os.environ

    def add_march_native(self):
        arg_march_native = "-march=native"
        arg_mcpu_native = "-mcpu=native"
        if self.test_supports_compile_arg(arg_march_native):
            for e in self.extensions:
                e.extra_compile_args.append(arg_march_native)
        elif self.test_supports_compile_arg(arg_mcpu_native):
            for e in self.extensions:
                e.extra_compile_args.append(arg_mcpu_native)

    def add_link_time_optimization(self):
        arg_lto = "-flto"
        if self.test_supports_compile_arg(arg_lto):
            for e in self.extensions:
                e.extra_compile_args.append(arg_lto)
                e.extra_link_args.append(arg_lto)

    def add_no_math_errno(self):
        arg_fnme = "-fno-math-errno"
        if self.test_supports_compile_arg(arg_fnme):
            for e in self.extensions:
                e.extra_compile_args.append(arg_fnme)
                e.extra_link_args.append(arg_fnme)

    def add_no_trapping_math(self):
        arg_fntm = "-fno-trapping-math"
        if self.test_supports_compile_arg(arg_fntm):
            for e in self.extensions:
                e.extra_compile_args.append(arg_fntm)
                e.extra_link_args.append(arg_fntm)

    def add_openmp_linkage(self):
        arg_omp1 = "-fopenmp"
        arg_omp2 = "-qopenmp"
        arg_omp3 = "-xopenmp"
        arg_omp4 = "-fiopenmp"
        args_apple_omp = ["-Xclang", "-fopenmp", "-lomp"]
        args_apple_omp2 = ["-Xclang", "-fopenmp", "-L/usr/local/lib", "-lomp", "-I/usr/local/include"]
        if self.test_supports_compile_arg(arg_omp1, with_omp=True):
            for e in self.extensions:
                e.extra_compile_args.append(arg_omp1)
                e.extra_link_args.append(arg_omp1)
        elif (sys.platform[:3].lower() == "dar") and self.test_supports_compile_arg(args_apple_omp, with_omp=True):
            for e in self.extensions:
                e.extra_compile_args += ["-Xclang", "-fopenmp"]
                e.extra_link_args += ["-lomp"]
        elif (sys.platform[:3].lower() == "dar") and self.test_supports_compile_arg(args_apple_omp2, with_omp=True):
            for e in self.extensions:
                e.extra_compile_args += ["-Xclang", "-fopenmp"]
                e.extra_link_args += ["-L/usr/local/lib", "-lomp"]
                e.include_dirs += ["/usr/local/include"]
        elif self.test_supports_compile_arg(arg_omp2, with_omp=True):
            for e in self.extensions:
                e.extra_compile_args.append(arg_omp2)
                e.extra_link_args.append(arg_omp2)
        elif self.test_supports_compile_arg(arg_omp3, with_omp=True):
            for e in self.extensions:
                e.extra_compile_args.append(arg_omp3)
                e.extra_link_args.append(arg_omp3)
        elif self.test_supports_compile_arg(arg_omp4, with_omp=True):
            for e in self.extensions:
                e.extra_compile_args.append(arg_omp4)
                e.extra_link_args.append(arg_omp4)
        else:
            set_omp_false()

    def add_O3(self):
        O3 = "-O3"
        if self.test_supports_compile_arg(O3):
            for e in self.extensions:
                e.extra_compile_args.append(O3)

    def add_highest_supported_cxx_standard(self):
        cxx17 = "-std=c++17"
        cxx14 = "-std=gnu++14"
        cxx11 = "-std=c++11"
        if self.test_supports_compile_arg(cxx17):
            for e in self.extensions:
                e.extra_compile_args.append(cxx17)
        elif self.test_supports_compile_arg(cxx14):
            for e in self.extensions:
                e.extra_compile_args.append(cxx14)
        elif self.test_supports_compile_arg(cxx11):
            for e in self.extensions:
                e.extra_compile_args.append(cxx11)
        else:
            msg  = "\n\n\nWarning: compiler does not support C++11, compilation/installation of "
            msg += "'isotree' might fail without it.\n\n\n"
            warnings.warn(msg)

    def test_supports_compile_arg(self, comm, with_omp=False):
        is_supported = False
        try:
            if not hasattr(self.compiler, "compiler_cxx"):
                return False
            if not isinstance(comm, list):
                comm = [comm]
            print("--- Checking compiler support for option '%s'" % " ".join(comm))
            fname = "isotree_compiler_testing.cpp"
            with open(fname, "w") as ftest:
                ftest.write(u"int main(int argc, char**argv) {return 0;}\n")
            try:
                if not isinstance(self.compiler.compiler_cxx, list):
                    cmd = list(self.compiler.compiler_cxx)
                else:
                    cmd = self.compiler.compiler_cxx
            except:
                cmd = self.compiler.compiler_cxx
            val_good = subprocess.call(cmd + [fname])
            if with_omp:
                with open(fname, "w") as ftest:
                    ftest.write(u"#include <omp.h>\nint main(int argc, char**argv) {return 0;}\n")
            try:
                val = subprocess.call(cmd + comm + [fname])
                is_supported = (val == val_good)
            except:
                is_supported = False
        except:
            pass
        try:
            os.remove(fname)
        except:
            pass
        return is_supported

    def is_arch_in_cflags(self):
        arch_flags = '-march -mtune -msse -msse2 -msse3 -mssse3 -msse4 -msse4a -msse4.1 -msse4.2 -mavx -mavx2 -mcpu'.split()
        for env_var in ("CFLAGS", "CXXFLAGS"):
            if env_var in os.environ:
                for flag in arch_flags:
                    if flag in os.environ[env_var]:
                        return True

        return False

    def add_restrict_qualifier(self):
        supports_restrict = False
        try:
            if not hasattr(self.compiler, "compiler_cxx"):
                return None
            print("--- Checking compiler support for '__restrict' qualifier")
            fname = "isotree_compiler_testing.cpp"
            with open(fname, "w") as ftest:
                ftest.write(u"int main(int argc, char**argv) {return 0;}\n")
            try:
                if not isinstance(self.compiler.compiler_cxx, list):
                    cmd = list(self.compiler.compiler_cxx)
                else:
                    cmd = self.compiler.compiler_cxx
            except:
                cmd = self.compiler.compiler_cxx
            val_good = subprocess.call(cmd + [fname])
            try:
                with open(fname, "w") as ftest:
                    ftest.write(u"int main(int argc, char**argv) {double *__restrict x = 0; return 0;}\n")
                val = subprocess.call(cmd + [fname])
                supports_restrict = (val == val_good)
            except:
                return None
        except:
            pass
        try:
            os.remove(fname)
        except:
            pass
        
        if supports_restrict:
            for e in self.extensions:
                e.define_macros += [("SUPPORTS_RESTRICT", "1")]


setup(
    name  = "isotree",
    packages = ["isotree"],
    version = '0.5.15-1',
    description = 'Isolation-Based Outlier Detection, Distance, and NA imputation',
    author = 'David Cortes',
    author_email = 'david.cortes.rivera@gmail.com',
    url = 'https://github.com/david-cortes/isotree',
    keywords = ['isolation-forest', 'anomaly', 'outlier'],
    cmdclass = {'build_ext': build_ext_subclass},
    ext_modules = [Extension(
                                "isotree._cpp_interface",
                                sources=["isotree/cpp_interface.pyx",
                                         "src/indexer.cpp",
                                         "src/merge_models.cpp", "src/subset_models.cpp",
                                         "src/serialize.cpp", "src/sql.cpp"],
                                include_dirs=[np.get_include(), ".", "./src"],
                                language="c++",
                                install_requires = ["numpy", "pandas>=0.24.0", "cython", "scipy"],
                                define_macros = [("_USE_XOSHIRO", None),
                                                 ("_FOR_PYTHON", None)]
                            )]
    )

if not found_omp:
    omp_msg  = "\n\n\nCould not detect OpenMP. Package will be built without multi-threading capabilities. "
    omp_msg += " To enable multi-threading, first install OpenMP"
    if (sys.platform[:3] == "dar"):
        omp_msg += " - for macOS: 'brew install libomp'\n"
    else:
        omp_msg += " modules for your compiler. "
    
    omp_msg += "Then reinstall this package from scratch: 'pip install --force-reinstall isotree'.\n"
    warnings.warn(omp_msg)
