AC_INIT([isotree], 0.5.17)
AC_LANG(C++)
AC_COMPILE_IFELSE(
    [AC_LANG_PROGRAM([[]], [[double *__restrict x;]])],
    [SUPPORTS_RESTRICT="-DSUPPORTS_RESTRICT=1"],
    [SUPPORTS_RESTRICT=""]
)
AC_SUBST(SUPPORTS_RESTRICT)

AC_LANG(C++)
OLD_CXXFLAGS="$CXXFLAGS"
CXXFLAGS="$CXXFLAGS -fno-math-errno"
AC_COMPILE_IFELSE(
    [AC_LANG_PROGRAM([[]])],
    [FNE_FLAG="-fno-math-errno"],
    [FNE_FLAG=""]
)
CXXFLAGS="$OLD_CXXFLAGS"
AC_SUBST(FNE_FLAG)

OLD_CXXFLAGS="$CXXFLAGS"
CXXFLAGS="$CXXFLAGS -fno-trapping-math"
AC_COMPILE_IFELSE(
    [AC_LANG_PROGRAM([[]])],
    [FNTP_FLAG="-fno-trapping-math"],
    [FNTP_FLAG=""]
)
CXXFLAGS="$OLD_CXXFLAGS"
AC_SUBST(FNTP_FLAG)

OLD_CXXFLAGS="$CXXFLAGS"
CXXFLAGS="$CXXFLAGS -std=c++17"
AC_COMPILE_IFELSE(
    [AC_LANG_PROGRAM([[]])],
    [CXX_SUPPORTED="CXX17"],
    [CXX_SUPPORTED="CXX11"]
)
CXXFLAGS="$OLD_CXXFLAGS"

if test "$CXX_SUPPORTED" = "CXX11"; then
    OLD_CXXFLAGS="$CXXFLAGS"
    CXXFLAGS="$CXXFLAGS -std=gnu++14"
    AC_COMPILE_IFELSE(
        [AC_LANG_PROGRAM([[]])],
        [CXX_SUPPORTED="CXX14"],
        [CXX_SUPPORTED="CXX11"]
    )
    CXXFLAGS="$OLD_CXXFLAGS"
fi

AC_SUBST(CXX_SUPPORTED)

LD_SUPPORT=""
if test -n "${NO_LONG_DOUBLE}"; then
    LD_SUPPORT="-DNO_LONG_DOUBLE"
fi
if test "$LD_SUPPORT" = ""; then
    if test `${R_HOME}/bin${R_ARCH_BIN}/Rscript -e "cat(unname(as.integer(capabilities('long.double'))))"` = "0"; then
        LD_SUPPORT="-DNO_LONG_DOUBLE"
    fi
fi
AC_SUBST(LD_SUPPORT)

AC_CONFIG_FILES([src/Makevars])
AC_OUTPUT
