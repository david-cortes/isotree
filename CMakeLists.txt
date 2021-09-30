cmake_minimum_required (VERSION 3.12.4)
project (isotree VERSION 0.3.4)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(SRC_FILES ${PROJECT_SOURCE_DIR}/src/instantiate_model.cpp
              ${PROJECT_SOURCE_DIR}/src/oop_interface.cpp
              ${PROJECT_SOURCE_DIR}/src/c_interface.cpp
              ${PROJECT_SOURCE_DIR}/src/merge_models.cpp
              ${PROJECT_SOURCE_DIR}/src/subset_models.cpp
              ${PROJECT_SOURCE_DIR}/src/serialize.cpp
              ${PROJECT_SOURCE_DIR}/src/sql.cpp)
set(BUILD_SHARED_LIBS True)
add_library(isotree SHARED ${SRC_FILES})
target_include_directories(isotree PRIVATE ${PROJECT_SOURCE_DIR}/src)
set(HEADER_FILES "${PROJECT_SOURCE_DIR}/include/isotree.hpp"
                 "${PROJECT_SOURCE_DIR}/include/isotree_oop.hpp"
                 "${PROJECT_SOURCE_DIR}/include/isotree_c.h")
set_target_properties(isotree PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")
add_compile_definitions(ISOTREE_COMPILE_TIME)

## omit this line to use the system's default RNG engine
option(USE_XOSHIRO "Use the xoshiro RNG family" ON)
if (USE_XOSHIRO)
    message(STATUS "Using the Xoshiro random engine family")
    add_compile_definitions(_USE_XOSHIRO)
else()
    message(STATUS "Using the compiler's default random engine")
endif()

## https://cliutils.gitlab.io/modern-cmake/chapters/packages/OpenMP.html
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(isotree PUBLIC OpenMP::OpenMP_CXX)
else()
    message(STATUS "OpenMP not found - will compile without multi-threading support")
endif()

# check if the robin-map library is available under /src
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_INCLUDES ${PROJECT_SOURCE_DIR}/src)
check_cxx_source_compiles(
    "
    #include \"robinmap/include/tsl/robin_growth_policy.h\"
    int main(int argc, char **argv)
    {
        return 0;
    }
    "
    HAS_ROBIN_MAP
)
if (HAS_ROBIN_MAP)
    add_compile_definitions(_USE_ROBIN_MAP)
else()
    # if it's not under /src, check if there's a system install
    find_package(tsl-robin-map)
    if (tsl-robin-map_FOUND)
        target_link_libraries(isotree PRIVATE tsl::robin_map)
        add_compile_definitions(_USE_SYSTEM_ROBIN)
    else()
        message(WARNING "Robin-Map Headers not found - will use STL hashmaps which are slower.")
    endif()
endif()

# For handling large files with MinGW
if (WIN32)
    if (CMAKE_SIZEOF_VOID_P GREATER_EQUAL 8 AND (MSYS OR MINGW OR GCC))
        add_compile_definitions(_FILE_OFFSET_BITS=64)
    endif()
endif()

# Compiler optimizations
check_cxx_source_compiles(
    "
    #include <cstddef>
    int main(int argc, char **argv)
    {
        double *__restrict x = nullptr;
        return 0;
    }
    "
    SUPPORTS_RESTRICT
)
if (SUPPORTS_RESTRICT)
    add_compile_definitions(SUPPORTS_RESTRICT=1)
endif()

if (MSVC)
    if (NOT (${CMAKE_CXX_FLAGS_RELEASE} MATCHES "/O2"))
        set(OLD_FLAGS ${CMAKE_REQUIRED_FLAGS})
        set(CMAKE_REQUIRED_FLAGS "/O2")
        check_cxx_source_compiles(
            "
            int main(int argc, char **argv)
            {
                return 0;
            }
            "
            SUPPORTS_O2
        )
        set(CMAKE_REQUIRED_FLAGS ${OLD_FLAGS})
        if (SUPPORTS_O2)
            set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
        endif()
    endif()
else()
    if (NOT (${CMAKE_CXX_FLAGS_RELEASE} MATCHES "-O3 "))
        set(OLD_FLAGS ${CMAKE_REQUIRED_FLAGS})
        set(CMAKE_REQUIRED_FLAGS "-O3")
        check_cxx_source_compiles(
            "
            int main(int argc, char **argv)
            {
                return 0;
            }
            "
            SUPPORTS_O3
        )
        set(CMAKE_REQUIRED_FLAGS ${OLD_FLAGS})
        if (SUPPORTS_O3)
            set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
        else()
            set(CMAKE_REQUIRED_FLAGS "-O2")
            check_cxx_source_compiles(
                "
                int main(int argc, char **argv)
                {
                    return 0;
                }
                "
                SUPPORTS_O2
            )
            set(CMAKE_REQUIRED_FLAGS ${OLD_FLAGS})
            if (SUPPORTS_O2)
                set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")
            endif()
        endif()
    endif()
endif()

option(USE_MARCH_NATIVE "Build with -march=native" OFF)
if (USE_MARCH_NATIVE AND NOT MSVC)
    set(OLD_FLAGS ${CMAKE_REQUIRED_FLAGS})
    set(CMAKE_REQUIRED_FLAGS "-march=native")
    check_cxx_source_compiles(
        "
        int main(int argc, char **argv)
        {
            return 0;
        }
        "
        SUPPORTS_MARCH_NATIVE
    )
    set(CMAKE_REQUIRED_FLAGS ${OLD_FLAGS})
    if (SUPPORTS_MARCH_NATIVE)
        message(STATUS "Adding flag -march=native.")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
    else()
        set(CMAKE_REQUIRED_FLAGS "-mcpu=native")
        check_cxx_source_compiles(
            "
            int main(int argc, char **argv)
            {
                return 0;
            }
            "
            SUPPORTS_MCPU_NATIVE
        )
        set(CMAKE_REQUIRED_FLAGS ${OLD_FLAGS})
        if (SUPPORTS_MCPU_NATIVE)
            message(STATUS "Adding flag -mcpu=native.")
            set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mcpu=native")
        else()
            message(WARNING "Flags -march=native and -mcpu=native not supported by the compiler.")
        endif()
    endif()
endif()

# Link-time optimization if supported
# https://stackoverflow.com/questions/31355692/how-do-i-enable-link-time-optimization-lto-with-cmake
include(CheckIPOSupported)
check_ipo_supported(RESULT LTO_SUPPORTED)
if (LTO_SUPPORTED)
    set_property(TARGET isotree PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()


include(GNUInstallDirs)

install(TARGETS isotree
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

configure_file(isotree.pc.in isotree.pc @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/isotree.pc DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/pkgconfig)

# uninstall target
if(NOT TARGET uninstall)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
        IMMEDIATE @ONLY)

    add_custom_target(uninstall
        COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif()
