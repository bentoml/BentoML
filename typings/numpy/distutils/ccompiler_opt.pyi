"""Provides the `CCompilerOpt` class, used for handling the CPU/hardware
optimization, starting from parsing the command arguments, to managing the
relation between the CPU baseline and dispatch-able features,
also generating the required C headers and ending with compiling
the sources with proper compiler's flags.

`CCompilerOpt` doesn't provide runtime detection for the CPU features,
instead only focuses on the compiler side, but it creates abstract C headers
that can be used later for the final runtime dispatching process."""

class _Config:
    """An abstract class holds all configurable attributes of `CCompilerOpt`,
    these class attributes can be used to change the default behavior
    of `CCompilerOpt` in order to fit other requirements.

    Attributes
    ----------
    conf_nocache : bool
        Set True to disable memory and file cache.
        Default is False.

    conf_noopt : bool
        Set True to forces the optimization to be disabled,
        in this case `CCompilerOpt` tends to generate all
        expected headers in order to 'not' break the build.
        Default is False.

    conf_cache_factors : list
        Add extra factors to the primary caching factors. The caching factors
        are utilized to determine if there are changes had happened that
        requires to discard the cache and re-updating it. The primary factors
        are the arguments of `CCompilerOpt` and `CCompiler`'s properties(type, flags, etc).
        Default is list of two items, containing the time of last modification
        of `ccompiler_opt` and value of attribute "conf_noopt"

    conf_tmp_path : str,
        The path of temporary directory. Default is auto-created
        temporary directory via ``tempfile.mkdtemp()``.

    conf_check_path : str
        The path of testing files. Each added CPU feature must have a
        **C** source file contains at least one intrinsic or instruction that
        related to this feature, so it can be tested against the compiler.
        Default is ``./distutils/checks``.

    conf_target_groups : dict
        Extra tokens that can be reached from dispatch-able sources through
        the special mark ``@targets``. Default is an empty dictionary.

        **Notes**:
            - case-insensitive for tokens and group names
            - sign '#' must stick in the begin of group name and only within ``@targets``

        **Example**:
            .. code-block:: console

                $ "@targets #avx_group other_tokens" > group_inside.c

            >>> CCompilerOpt.conf_target_groups["avx_group"] = \\
            "$werror $maxopt avx2 avx512f avx512_skx"
            >>> cco = CCompilerOpt(cc_instance)
            >>> cco.try_dispatch(["group_inside.c"])

    conf_c_prefix : str
        The prefix of public C definitions. Default is ``"NPY_"``.

    conf_c_prefix_ : str
        The prefix of internal C definitions. Default is ``"NPY__"``.

    conf_cc_flags : dict
        Nested dictionaries defining several compiler flags
        that linked to some major functions, the main key
        represent the compiler name and sub-keys represent
        flags names. Default is already covers all supported
        **C** compilers.

        Sub-keys explained as follows:

        "native": str or None
            used by argument option `native`, to detect the current
            machine support via the compiler.
        "werror": str or None
            utilized to treat warning as errors during testing CPU features
            against the compiler and also for target's policy `$werror`
            via dispatch-able sources.
        "maxopt": str or None
            utilized for target's policy '$maxopt' and the value should
            contains the maximum acceptable optimization by the compiler.
            e.g. in gcc `'-O3'`

        **Notes**:
            * case-sensitive for compiler names and flags
            * use space to separate multiple flags
            * any flag will tested against the compiler and it will skipped
              if it's not applicable.

    conf_min_features : dict
        A dictionary defines the used CPU features for
        argument option `'min'`, the key represent the CPU architecture
        name e.g. `'x86'`. Default values provide the best effort
        on wide range of users platforms.

        **Note**: case-sensitive for architecture names.

    conf_features : dict
        Nested dictionaries used for identifying the CPU features.
        the primary key is represented as a feature name or group name
        that gathers several features. Default values covers all
        supported features but without the major options like "flags",
        these undefined options handle it by method `conf_features_partial()`.
        Default value is covers almost all CPU features for *X86*, *IBM/Power64*
        and *ARM 7/8*.

        Sub-keys explained as follows:

        "implies" : str or list, optional,
            List of CPU feature names to be implied by it,
            the feature name must be defined within `conf_features`.
            Default is None.

        "flags": str or list, optional
            List of compiler flags. Default is None.

        "detect": str or list, optional
            List of CPU feature names that required to be detected
            in runtime. By default, its the feature name or features
            in "group" if its specified.

        "implies_detect": bool, optional
            If True, all "detect" of implied features will be combined.
            Default is True. see `feature_detect()`.

        "group": str or list, optional
            Same as "implies" but doesn't require the feature name to be
            defined within `conf_features`.

        "interest": int, required
            a key for sorting CPU features

        "headers": str or list, optional
            intrinsics C header file

        "disable": str, optional
            force disable feature, the string value should contains the
            reason of disabling.

        "autovec": bool or None, optional
            True or False to declare that CPU feature can be auto-vectorized
            by the compiler.
            By default(None), treated as True if the feature contains at
            least one applicable flag. see `feature_can_autovec()`

        "extra_checks": str or list, optional
            Extra test case names for the CPU feature that need to be tested
            against the compiler.

            Each test case must have a C file named ``extra_xxxx.c``, where
            ``xxxx`` is the case name in lower case, under 'conf_check_path'.
            It should contain at least one intrinsic or function related to the test case.

            If the compiler able to successfully compile the C file then `CCompilerOpt`
            will add a C ``#define`` for it into the main dispatch header, e.g.
            ```#define {conf_c_prefix}_XXXX`` where ``XXXX`` is the case name in upper case.

        **NOTES**:
            * space can be used as separator with options that supports "str or list"
            * case-sensitive for all values and feature name must be in upper-case.
            * if flags aren't applicable, its will skipped rather than disable the
              CPU feature
            * the CPU feature will disabled if the compiler fail to compile
              the test file
    """

    conf_nocache = ...
    conf_noopt = ...
    conf_cache_factors = ...
    conf_tmp_path = ...
    conf_check_path = ...
    conf_target_groups = ...
    conf_c_prefix = ...
    conf_c_prefix_ = ...
    conf_cc_flags = ...
    conf_min_features = ...
    conf_features = ...
    def conf_features_partial(self):
        """Return a dictionary of supported CPU features by the platform,
        and accumulate the rest of undefined options in `conf_features`,
        the returned dict has same rules and notes in
        class attribute `conf_features`, also its override
        any options that been set in 'conf_features'.
        """
        ...
    def __init__(self) -> None: ...

class _Distutils:
    """A helper class that provides a collection of fundamental methods
    implemented in a top of Python and NumPy Distutils.

    The idea behind this class is to gather all methods that it may
    need to override in case of reuse 'CCompilerOpt' in environment
    different than of what NumPy has.

    Parameters
    ----------
    ccompiler : `CCompiler`
        The generate instance that returned from `distutils.ccompiler.new_compiler()`.
    """

    def __init__(self, ccompiler) -> None: ...
    def dist_compile(self, sources, flags, ccompiler=..., **kwargs):
        """Wrap CCompiler.compile()"""
        ...
    def dist_test(self, source, flags, macros=...):
        """Return True if 'CCompiler.compile()' able to compile
        a source file with certain flags.
        """
        ...
    def dist_info(self):
        """
        Return a tuple containing info about (platform, compiler, extra_args),
        required by the abstract class '_CCompiler' for discovering the
        platform environment. This is also used as a cache factor in order
        to detect any changes happening from outside.
        """
        ...
    @staticmethod
    def dist_error(*args):
        """Raise a compiler error"""
        ...
    @staticmethod
    def dist_fatal(*args):
        """Raise a distutils error"""
        ...
    @staticmethod
    def dist_log(*args, stderr=...):
        """Print a console message"""
        ...
    @staticmethod
    def dist_load_module(name, path):
        """Load a module from file, required by the abstract class '_Cache'."""
        ...
    _dist_warn_regex = ...

_share_cache = ...

class _Cache:
    """An abstract class handles caching functionality, provides two
    levels of caching, in-memory by share instances attributes among
    each other and by store attributes into files.

    **Note**:
        any attributes that start with ``_`` or ``conf_`` will be ignored.

    Parameters
    ----------
    cache_path: str or None
        The path of cache file, if None then cache in file will disabled.

    *factors:
        The caching factors that need to utilize next to `conf_cache_factors`.

    Attributes
    ----------
    cache_private: set
        Hold the attributes that need be skipped from "in-memory cache".

    cache_infile: bool
        Utilized during initializing this class, to determine if the cache was able
        to loaded from the specified cache path in 'cache_path'.
    """

    _cache_ignore = ...
    def __init__(self, cache_path=..., *factors) -> None: ...
    def __del__(self): ...
    def cache_flush(self):
        """
        Force update the cache.
        """
        ...
    def cache_hash(self, *factors): ...
    @staticmethod
    def me(cb):
        """
        A static method that can be treated as a decorator to
        dynamically cache certain methods.
        """
        ...

class _CCompiler:
    """A helper class for `CCompilerOpt` containing all utilities that
    related to the fundamental compiler's functions.

    Attributes
    ----------
    cc_on_x86 : bool
        True when the target architecture is 32-bit x86
    cc_on_x64 : bool
        True when the target architecture is 64-bit x86
    cc_on_ppc64 : bool
        True when the target architecture is 64-bit big-endian PowerPC
    cc_on_armhf : bool
        True when the target architecture is 32-bit ARMv7+
    cc_on_aarch64 : bool
        True when the target architecture is 64-bit Armv8-a+
    cc_on_noarch : bool
        True when the target architecture is unknown or not supported
    cc_is_gcc : bool
        True if the compiler is GNU or
        if the compiler is unknown
    cc_is_clang : bool
        True if the compiler is Clang
    cc_is_icc : bool
        True if the compiler is Intel compiler (unix like)
    cc_is_iccw : bool
        True if the compiler is Intel compiler (msvc like)
    cc_is_nocc : bool
        True if the compiler isn't supported directly,
        Note: that cause a fail-back to gcc
    cc_has_debug : bool
        True if the compiler has debug flags
    cc_has_native : bool
        True if the compiler has native flags
    cc_noopt : bool
        True if the compiler has definition 'DISABLE_OPT*',
        or 'cc_on_noarch' is True
    cc_march : str
        The target architecture name, or "unknown" if
        the architecture isn't supported
    cc_name : str
        The compiler name, or "unknown" if the compiler isn't supported
    cc_flags : dict
        Dictionary containing the initialized flags of `_Config.conf_cc_flags`
    """

    def __init__(self) -> None: ...
    @_Cache.me
    def cc_test_flags(self, flags):
        """
        Returns True if the compiler supports 'flags'.
        """
        ...
    def cc_normalize_flags(self, flags):
        """
        Remove the conflicts that caused due gathering implied features flags.

        Parameters
        ----------
        'flags' list, compiler flags
            flags should be sorted from the lowest to the highest interest.

        Returns
        -------
        list, filtered from any conflicts.

        Examples
        --------
        >>> self.cc_normalize_flags(['-march=armv8.2-a+fp16', '-march=armv8.2-a+dotprod'])
        ['armv8.2-a+fp16+dotprod']

        >>> self.cc_normalize_flags(
            ['-msse', '-msse2', '-msse3', '-mssse3', '-msse4.1', '-msse4.2', '-mavx', '-march=core-avx2']
        )
        ['-march=core-avx2']
        """
        ...
    _cc_normalize_unix_mrgx = ...
    _cc_normalize_unix_frgx = ...
    _cc_normalize_unix_krgx = ...
    _cc_normalize_arch_ver = ...
    _cc_normalize_win_frgx = ...
    _cc_normalize_win_mrgx = ...

class _Feature:
    """A helper class for `CCompilerOpt` that managing CPU features.

    Attributes
    ----------
    feature_supported : dict
        Dictionary containing all CPU features that supported
        by the platform, according to the specified values in attribute
        `_Config.conf_features` and `_Config.conf_features_partial()`

    feature_min : set
        The minimum support of CPU features, according to
        the specified values in attribute `_Config.conf_min_features`.
    """

    def __init__(self) -> None: ...
    def feature_names(self, names=..., force_flags=..., macros=...):
        """
        Returns a set of CPU feature names that supported by platform and the **C** compiler.

        Parameters
        ----------
        names: sequence or None, optional
            Specify certain CPU features to test it against the **C** compiler.
            if None(default), it will test all current supported features.
            **Note**: feature names must be in upper-case.

        force_flags: list or None, optional
            If None(default), default compiler flags for every CPU feature will
            be used during the test.

        macros : list of tuples, optional
            A list of C macro definitions.
        """
        ...
    def feature_is_exist(self, name):
        """
        Returns True if a certain feature is exist and covered within
        `_Config.conf_features`.

        Parameters
        ----------
        'name': str
            feature name in uppercase.
        """
        ...
    def feature_sorted(self, names, reverse=...):
        """
        Sort a list of CPU features ordered by the lowest interest.

        Parameters
        ----------
        'names': sequence
            sequence of supported feature names in uppercase.
        'reverse': bool, optional
            If true, the sorted features is reversed. (highest interest)

        Returns
        -------
        list, sorted CPU features
        """
        ...
    def feature_implies(self, names, keep_origins=...):
        """
        Return a set of CPU features that implied by 'names'

        Parameters
        ----------
        names: str or sequence of str
            CPU feature name(s) in uppercase.

        keep_origins: bool
            if False(default) then the returned set will not contain any
            features from 'names'. This case happens only when two features
            imply each other.

        Examples
        --------
        >>> self.feature_implies("SSE3")
        {'SSE', 'SSE2'}
        >>> self.feature_implies("SSE2")
        {'SSE'}
        >>> self.feature_implies("SSE2", keep_origins=True)
        # 'SSE2' found here since 'SSE' and 'SSE2' imply each other
        {'SSE', 'SSE2'}
        """
        ...
    def feature_implies_c(self, names):
        """same as feature_implies() but combining 'names'"""
        ...
    def feature_ahead(self, names):
        """
        Return list of features in 'names' after remove any
        implied features and keep the origins.

        Parameters
        ----------
        'names': sequence
            sequence of CPU feature names in uppercase.

        Returns
        -------
        list of CPU features sorted as-is 'names'

        Examples
        --------
        >>> self.feature_ahead(["SSE2", "SSE3", "SSE41"])
        ["SSE41"]
        # assume AVX2 and FMA3 implies each other and AVX2
        # is the highest interest
        >>> self.feature_ahead(["SSE2", "SSE3", "SSE41", "AVX2", "FMA3"])
        ["AVX2"]
        # assume AVX2 and FMA3 don't implies each other
        >>> self.feature_ahead(["SSE2", "SSE3", "SSE41", "AVX2", "FMA3"])
        ["AVX2", "FMA3"]
        """
        ...
    def feature_untied(self, names):
        """
        same as 'feature_ahead()' but if both features implied each other
        and keep the highest interest.

        Parameters
        ----------
        'names': sequence
            sequence of CPU feature names in uppercase.

        Returns
        -------
        list of CPU features sorted as-is 'names'

        Examples
        --------
        >>> self.feature_untied(["SSE2", "SSE3", "SSE41"])
        ["SSE2", "SSE3", "SSE41"]
        # assume AVX2 and FMA3 implies each other
        >>> self.feature_untied(["SSE2", "SSE3", "SSE41", "FMA3", "AVX2"])
        ["SSE2", "SSE3", "SSE41", "AVX2"]
        """
        ...
    def feature_get_til(self, names, keyisfalse):
        """
        same as `feature_implies_c()` but stop collecting implied
        features when feature's option that provided through
        parameter 'keyisfalse' is False, also sorting the returned
        features.
        """
        ...
    def feature_detect(self, names):
        """
        Return a list of CPU features that required to be detected
        sorted from the lowest to highest interest.
        """
        ...
    @_Cache.me
    def feature_flags(self, names):
        """
        Return a list of CPU features flags sorted from the lowest
        to highest interest.
        """
        ...
    @_Cache.me
    def feature_test(self, name, force_flags=..., macros=...):
        """
        Test a certain CPU feature against the compiler through its own
        check file.

        Parameters
        ----------
        name: str
            Supported CPU feature name.

        force_flags: list or None, optional
            If None(default), the returned flags from `feature_flags()`
            will be used.

        macros : list of tuples, optional
            A list of C macro definitions.
        """
        ...
    @_Cache.me
    def feature_is_supported(self, name, force_flags=..., macros=...):
        """
        Check if a certain CPU feature is supported by the platform and compiler.

        Parameters
        ----------
        name: str
            CPU feature name in uppercase.

        force_flags: list or None, optional
            If None(default), default compiler flags for every CPU feature will
            be used during test.

        macros : list of tuples, optional
            A list of C macro definitions.
        """
        ...
    @_Cache.me
    def feature_can_autovec(self, name):
        """
        check if the feature can be auto-vectorized by the compiler
        """
        ...
    @_Cache.me
    def feature_extra_checks(self, name):
        """
        Return a list of supported extra checks after testing them against
        the compiler.

        Parameters
        ----------
        names: str
            CPU feature name in uppercase.
        """
        ...
    def feature_c_preprocessor(self, feature_name, tabs=...):
        """
        Generate C preprocessor definitions and include headers of a CPU feature.

        Parameters
        ----------
        'feature_name': str
            CPU feature name in uppercase.
        'tabs': int
            if > 0, align the generated strings to the right depend on number of tabs.

        Returns
        -------
        str, generated C preprocessor

        Examples
        --------
        >>> self.feature_c_preprocessor("SSE3")
        /** SSE3 **/
        #define NPY_HAVE_SSE3 1
        #include <pmmintrin.h>
        """
        ...

class _Parse:
    """A helper class that parsing main arguments of `CCompilerOpt`,
    also parsing configuration statements in dispatch-able sources.

    Parameters
    ----------
    cpu_baseline: str or None
        minimal set of required CPU features or special options.

    cpu_dispatch: str or None
        dispatched set of additional CPU features or special options.

    Special options can be:
        - **MIN**: Enables the minimum CPU features that utilized via `_Config.conf_min_features`
        - **MAX**: Enables all supported CPU features by the Compiler and platform.
        - **NATIVE**: Enables all CPU features that supported by the current machine.
        - **NONE**: Enables nothing
        - **Operand +/-**: remove or add features, useful with options **MAX**, **MIN** and **NATIVE**.
            NOTE: operand + is only added for nominal reason.

    NOTES:
        - Case-insensitive among all CPU features and special options.
        - Comma or space can be used as a separator.
        - If the CPU feature is not supported by the user platform or compiler,
          it will be skipped rather than raising a fatal error.
        - Any specified CPU features to 'cpu_dispatch' will be skipped if its part of CPU baseline features
        - 'cpu_baseline' force enables implied features.

    Attributes
    ----------
    parse_baseline_names : list
        Final CPU baseline's feature names(sorted from low to high)
    parse_baseline_flags : list
        Compiler flags of baseline features
    parse_dispatch_names : list
        Final CPU dispatch-able feature names(sorted from low to high)
    parse_target_groups : dict
        Dictionary containing initialized target groups that configured
        through class attribute `conf_target_groups`.

        The key is represent the group name and value is a tuple
        contains three items :
            - bool, True if group has the 'baseline' option.
            - list, list of CPU features.
            - list, list of extra compiler flags.

    """

    def __init__(self, cpu_baseline, cpu_dispatch) -> None: ...
    def parse_targets(self, source):
        """
        Fetch and parse configuration statements that required for
        defining the targeted CPU features, statements should be declared
        in the top of source in between **C** comment and start
        with a special mark **@targets**.

        Configuration statements are sort of keywords representing
        CPU features names, group of statements and policies, combined
        together to determine the required optimization.

        Parameters
        ----------
        source: str
            the path of **C** source file.

        Returns
        -------
        - bool, True if group has the 'baseline' option
        - list, list of CPU features
        - list, list of extra compiler flags
        """
        ...
    _parse_regex_arg = ...
    _parse_regex_target = ...

class CCompilerOpt(_Config, _Distutils, _Cache, _CCompiler, _Feature, _Parse):
    """
    A helper class for `CCompiler` aims to provide extra build options
    to effectively control of compiler optimizations that are directly
    related to CPU features.
    """

    def __init__(
        self, ccompiler, cpu_baseline=..., cpu_dispatch=..., cache_path=...
    ) -> None: ...
    def is_cached(self):
        """
        Returns True if the class loaded from the cache file
        """
        ...
    def cpu_baseline_flags(self):
        """
        Returns a list of final CPU baseline compiler flags
        """
        ...
    def cpu_baseline_names(self):
        """
        return a list of final CPU baseline feature names
        """
        ...
    def cpu_dispatch_names(self):
        """
        return a list of final CPU dispatch feature names
        """
        ...
    def try_dispatch(self, sources, src_dir=..., ccompiler=..., **kwargs):
        """
        Compile one or more dispatch-able sources and generates object files,
        also generates abstract C config headers and macros that
        used later for the final runtime dispatching process.

        The mechanism behind it is to takes each source file that specified
        in 'sources' and branching it into several files depend on
        special configuration statements that must be declared in the
        top of each source which contains targeted CPU features,
        then it compiles every branched source with the proper compiler flags.

        Parameters
        ----------
        sources : list
            Must be a list of dispatch-able sources file paths,
            and configuration statements must be declared inside
            each file.

        src_dir : str
            Path of parent directory for the generated headers and wrapped sources.
            If None(default) the files will generated in-place.

        ccompiler: CCompiler
            Distutils `CCompiler` instance to be used for compilation.
            If None (default), the provided instance during the initialization
            will be used instead.

        **kwargs : any
            Arguments to pass on to the `CCompiler.compile()`

        Returns
        -------
        list : generated object files

        Raises
        ------
        CompileError
            Raises by `CCompiler.compile()` on compiling failure.
        DistutilsError
            Some errors during checking the sanity of configuration statements.

        See Also
        --------
        parse_targets :
            Parsing the configuration statements of dispatch-able sources.
        """
        ...
    def generate_dispatch_header(self, header_path):
        """
        Generate the dispatch header which contains the #definitions and headers
        for platform-specific instruction-sets for the enabled CPU baseline and
        dispatch-able features.

        Its highly recommended to take a look at the generated header
        also the generated source files via `try_dispatch()`
        in order to get the full picture.
        """
        ...
    def report(self, full=...): ...

def new_ccompiler_opt(compiler, dispatch_hpath, **kwargs):
    """
    Create a new instance of 'CCompilerOpt' and generate the dispatch header
    which contains the #definitions and headers of platform-specific instruction-sets for
    the enabled CPU baseline and dispatch-able features.

    Parameters
    ----------
    compiler : CCompiler instance
    dispatch_hpath : str
        path of the dispatch header

    **kwargs: passed as-is to `CCompilerOpt(...)`
    Returns
    -------
    new instance of CCompilerOpt
    """
    ...
