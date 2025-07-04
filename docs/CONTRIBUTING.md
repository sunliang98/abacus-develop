# Contributing to ABACUS

First of all, thank you for taking time to make contributions to ABACUS!
This file provides the more technical guidelines on how to realize it.
For more non-technical aspects, please refer to the [ABACUS Contribution Guide](./community/contribution_guide.md)

## Table of Contents

- [Got a question?](#got-a-question)
- [Structure of the package](#structure-of-the-package)
- [Submitting an Issue](#submitting-an-issue)
- [Comment style for documentation](#comment-style-for-documentation)
- [Code formatting style](#code-formatting-style)
- [Generating code coverage report](#generating-code-coverage-report)
- [Adding a unit test](#adding-a-unit-test)
- [Running unit tests](#running-unit-tests)
- [Debugging the codes](#debugging-the-codes)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [Commit message guidelines](#commit-message-guidelines)

## Got a question?

Please referring to our GitHub [issue tracker](https://github.com/deepmodeling/abacus-develop/issues), and our developers are willing to help.
If you find a bug, you can help us by submitting an issue to our GitHub Repository. Even better, you can submit a Pull Request with a patch. You can request a new feature by submitting an issue to our GitHub Repository.
If you would like to implement a new feature, please submit an issue with a proposal for your work first, and that ensures your work collaborates with our development road map well. For a major feature, first open an issue and outline your proposal so that it can be discussed. This will also allow us to better coordinate our efforts, prevent duplication of work, and help you to craft the change so that it is successfully accepted into the project.

## Structure of the package

Please refer to [our instructions](./quick_start/easy_install.md) on how to installing ABACUS.
The source code of ABACUS is based on several modules. Under the ABACUS root directory, there are the following folders:

- `cmake`: relevant files for finding required packages when compiling the code with cmake;
- `docs`: documents and supplementary info about ABACUS;
- `examples`: some examples showing the usage of ABACUS;
- `source`: the source code in separated modules, under which a `test` folder for its unit tests;
- `tests`: End-to-end test cases;
- `tools`: the script for generating the numerical atomic orbitals.

For those who are interested in the source code, the following figure shows the structure of the source code.

```text
|-- source_base                 A basic module including
|   |                           (1) Mathematical library interface functions: BLAS, LAPACK, Scalapack;
|   |                           (2) Custom data classes: matrix, vector definitions and related functions;
|   |                           (3) Parallelization functions: MPI, OpenMP;
|   |                           (4) Utility functions: timer, random number generator, etc.
|   |                           (5) Global parameters: input parameters, element names, mathematical and physical constants.
|   |-- module_container        The container module for storing data and performing operations on them and on different architectures.
|-- source_basis                Basis means the basis set to expand the wave function.
|   |-- module_ao               Atomic orbital basis set to be refactored.
|   |-- module_nao              New numerical atomic orbital basis set for two-center integrals in LCAO calculations
|   `-- module_pw               Data structures and relevant methods for planewave involved calculations
|-- source_cell                 The module for defining the unit cell and its operations, and reading pseudopotentials.
|   |-- module_neighbor         The module for finding the neighbors of each atom in the unit cell.
|   |-- module_paw              The module for performing PAW calculations.
|   |-- module_symmetry         The module for finding the symmetry operations of the unit cell.
|-- source_estate            The module for defining the electronic state and its operations.
|   |-- module_charge           The module for calculating the charge density, charge mixing
|   |-- potentials              The module for calculating the potentials, including Hartree, exchange-correlation, local pseudopotential, etc.
|-- source_esolver              The module defining task-specific driver of corresponding workflow for evaluating energies, forces, etc., including lj, dp, ks, sdft, ofdft, etc.
|   |                           TDDFT, Orbital-free DFT, etc.
|-- source_hamilt       The module for defining general Hamiltonian that can be used both in PW and LCAO calculations.
|   |-- module_ewald            The module for calculating the Ewald summation.
|   |-- module_surchem          The module for calculating the surface charge correction.
|   |-- module_vdw              The module for calculating the van der Waals correction.
|   |-- module_xc               The module for calculating the exchange-correlation energy and potential.
|-- source_lcao          The module for defining the Hamiltonian in LCAO calculations.
|   |-- hamilt_lcaodft          The module for defining the Hamiltonian in LCAO-DFT calculations.
|   |   |-- operator_lcao       The module for defining the operators in LCAO-DFT calculations.
|   |-- module_deepks           The module for defining the Hamiltonian in DeepKS calculations.
|   |-- module_dftu             The module for defining the Hamiltonian in DFT+U calculations.
|   |-- module_gint             The module for performing grid integral in LCAO calculations.
|   |-- module_hcontainer       The module for storing the Hamiltonian matrix in LCAO calculations.
|   |-- module_tddft            The module for defining the Hamiltonian in TDDFT calculations.
|   `-- module_ri               The module for performing RI calculations.
|-- source_pw            The module for defining the Hamiltonian in PW calculations.
|   |-- hamilt_ofdft            The module for defining the Hamiltonian in OFDFT calculations.
|   |-- hamilt_pwdft            The module for defining the Hamiltonian in PW-DFT calculations.
|   |   |-- operator_pw         The module for defining the operators in PW-DFT calculations.
|   `-- hamilt_stodft           The module for defining the Hamiltonian in STODFT calculations.
|-- source_hsolver              The module for solving the Hamiltonian with different diagonalization methods, including CG, Davidson in PW
|   |                           calculations, and scalapack and genelpa in LCAO calculations.
|-- source_io                   The module for reading of INPUT files and output properties including band structure, density of states, charge density, etc.
|-- source_md                   The module for performing molecular dynamics.
|-- source_psi                  The module for defining the wave function and its operations.
`-- source_relax                The module for performing structural optimization, optimized for cell and ion simultaneously.
```

## Submitting an Issue

Before you submit an issue, please search the issue tracker, and maybe your problem has been discussed and fixed. You can [submit new issues](https://github.com/deepmodeling/abacus-develop/issues/new/choose) by filling our issue forms.
To help us reproduce and confirm a bug, please provide a test case and building environment in your issue.

## Comment style for documentation

ABACUS uses Doxygen to generate docs directly from `.h` and `.cpp` code files.

For comments that need to be shown in documents, these formats should be used -- **Javadoc style** (as follow) is recommended, though Qt style is also ok. See it in [official manual](https://www.doxygen.nl/manual/docblocks.html).

A helpful VS Code extension -- [Doxygen Documentation Generator](https://marketplace.visualstudio.com/items?itemName=cschlosser.doxdocgen), can help you formating comments.

An practical example is class [LCAO_Deepks](https://github.com/deepmodeling/abacus-develop/blob/deepks/source/source_lcao/module_deepks/LCAO_deepks.h), the effects can be seen on [readthedocs page](https://abacus-deepks.readthedocs.io/en/latest/DeePKS_API/classLCAO__Descriptor.html#exhale-class-classLCAO-Descriptor)

- Tips
  - Only comments in .h file will be visible in generated  by Doxygen + Sphinx;
  - Private class members will not be documented;
  - Use [Markdown features](https://www.doxygen.nl/manual/markdown.html), such as using a empty new line for a new paragraph.

- Detailed Comment Block

    ```cpp
    /**
    * ... text ...
    */
    ```

- Brief + Detailed Comment Block

    ```cpp
    /// Brief description which ends at this dot. Details follow
    /// here.

    /// Brief description.
    /** Detailed description. */
    ```

- Comments After the Item: Add a "<"

    ```cpp
    int var; /**<Detailed description after the member */
    int var; ///<Brief description after the member
    ```

- Parameters
    usage: `[in],[out],[in,out] description`
    *e.g.*

    ```cpp
    void foo(int v/**< [in] docs for input parameter v.*/);
    ```

    or use `@param` command.

- Formula
  - inline: `\f$myformula\f$`
  - separate line: `\f[myformula\f]`
  - environment: `\f{environment}{myformula}`
  - *e.g.*

    ```latex
    \f{eqnarray*}{
            g &=& \frac{Gm_2}{r^2} \\
            &=& \frac{(6.673 \times 10^{-11}\,\mbox{m}^3\,\mbox{kg}^{-1}\,
                \mbox{s}^{-2})(5.9736 \times 10^{24}\,\mbox{kg})}{(6371.01\,\mbox{km})^2} \\
            &=& 9.82066032\,\mbox{m/s}^2
    \f}
    ```

## Code formatting style

We use `clang-format` as our code formatter. The `.clang-format` file in root directory describes the rules to conform with.
For Visual Studio Code developers, the [official extension of C/C++](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools) provided by Microsoft can help you format your codes following the rules. With this extension installed, format your code with `shift+command/alt+f`.
Configure your VS Code settings as `"C_Cpp.clang_format_style": "file"` (you can look up this option by pasting it into the search box of VS Code settings page), and all this stuff will take into effect. You may also set `"editor.formatOnSave": true` to avoid formatting files everytime manually.

We use <https://pre-commit.ci/> to format the code.
It is performed after pushing new commits to a PR. You might need to pull the changes before adding new commits.

To use pre-commit locally (**generally not required**):
Please install the pre-commit tool by running the following command:

```bash
pip install pre-commit
pip install clang-tidy clang-format # if you haven't installed them
```

Then, run the following command to install the pre-commit hooks:

```bash
pre-commit install
```

## Adding a unit test

We use [GoogleTest](https://github.com/google/googletest) as our test framework. Write your test under the corresponding module folder at `abacus-develop/tests`, then append the test to `tests/CMakeLists.txt`. If there are currently no unit tests provided for the module, do as follows. `source_base` provides a simple demonstration.

- Add a folder named `test` under the module.
- Append the content below to `CMakeLists.txt` of the module:

    ```cmake
    IF (BUILD_TESTING)
      add_subdirectory(test)
    endif()
    ```

- Add a blank `CMakeLists.txt` under `module*/test`.

To add a unit test:

- Write your test under `GoogleTest` framework.
- Add your testing source code with suffix `*_test.cpp` in `test` directory.
- Append the content below to `CMakeLists.txt` of the module:

    ```cmake
    AddTest(
      TARGET <module_name>_<test_name> # this is the executable file name of the test
      SOURCES <test_name>.cpp

      # OPTIONAL: if this test requires external libraries, add them with "LIBS" statement.
      LIBS math_libs # `math_libs` includes all math libraries in ABACUS.
    )
    ```

- Build with `-D BUILD_TESTING=1` flag, `cmake` will look for `GoogleTest` in the default path (usually `/usr/local`); if not found, you can specify the path with `-D GTEST_DIR`. You can find built testing programs under `build/source/<module_name>/test`.
- Follow the installing procedure of CMake. The tests will move to `build/test`.
- Considering `-D BUILD_TESTING=1`, the compilation will be slower compared with the case `-D BUILD_TESTING=0`.

## Running unit tests

1. Compiling ABACUS with unit tests.

    In order to run unit tests, ABACUS needs to be configured with `-D BUILD_TESTING=ON` flag. For example:

    ```bash
    cmake -B build -DBUILD_TESTING=ON
    ```

    then build ABACUS and unit testing with

    ```bash
    cmake --build build -j${number of processors}
    ```

    It is import to run the folloing command before running unit tests:

    ```bash
    cmake --install build
    ```

    to install mandatory supporting input files for unit tests.
    If you modified the unit tests to add new tests or learn how to write unit tests, it is convenient to run

    ```bash
    cmake --build build -j${number of processors} --target ${unit test name}
    ```

    to build a specific unit test. And please remember to run `cmake --install build` after building the unit test if the unit test requires supporting input files.

2. Running unit tests

    The test cases are located in `build/source/${module_name}/test` directory. Note that there are other directory names for unit tests, for example, `test_parallel` for running parallel unit tests, `test_pw` for running unit tests only used in plane wave basis calculation.

    You can run a single test in the specific directory. For example, run

    ```bash
    ./cell_unitcell_test
    ```

    under the directory of `build/source/source_cell/test` to run the test `cell_unitcell_test`.
    However, it is more convenient to run unit tests with `ctest` command under the `build` directory. You can check all unit tests by

    ```bash
    ctest -N
    ```

    The results will be shown as

    ```text
    Test project /root/abacus/build
    Test   #1: integrated_test
    Test   #2: Container_UTs
    Test   #3: base_blas_connector
    Test   #4: base_blacs_connector
    Test   #5: base_timer
    ...
    ```

    Note that the first one is integrated test, which is not a unit test. It is the test
    suite for testing the whole ABACUS package. The examples are located in the `tests/integrate` directory.

    To run a subset of tests, run the following command

    ```bash
    ctest -R <test-match-pattern> -V
    ```

    For example, `ctest -R cell` will perform tests with name matched by `cell`.
    You can also run a single test with

    ```bash
    ctest -R <test-name>
    ```

    For example, `ctest -R cell_unitcell_test_readpp` will   perform test `cell_unitcell_test_readpp`.
    To run all the unit tests, together with the integrated test, run

    ```bash
    cmake --build build --target test ARGS="-V --timeout 21600"
    ```

    in the `abacus-develop` directory.

## Adding an integrate test
The integrate test is a test suite for testing the whole ABACUS package. The examples are located in the `tests/integrate` directory. Before adding a new test, please firstly read `README.md` in `tests/integrate` to understand the structure of the integrate test. To add an integrate test:
1. Add a new directory under `tests/integrate` for the new test.
2. Prepare the input files for the new test. 
    - The input files should be placed in the new directory. Pseudopotential files and orbital files should be placed in `tests/PP_ORB`. You should define the correct `pseudo_dir` and `orb_dir`(if need orbital files) in INPUT with the relative path to the `tests/PP_ORB` directory, and be sure the new test can be run successfully.
    - The running time of the new test should not exceed 20 seconds. You can try to reduce the time by below methods (on the premise of ensuring the effectiveness of the test):
        - Reduce the number of atoms in the unit cell (1~2 atoms).
        - Reduce the number of k-points (`1 1 1` or `2 2 2`).
        - Reduce ecutwfc (20~50 Ry).
        - Reduce the number of steps for relax or md job (2~3 steps). 
        - Reduce the basis set for LCAO calculations (DZP orbital and 6 a.u. cutoff).
    - For PW calculations, should set `pw_seed 1` in INPUT file to ensure the reproducibility of the test.
3. Generate the reference results for the new test.
    - Run the new test with GNU compiler and 4 MPI processes 2 OpenMP threads. The command is `OMP_NUM_THREADS=2 mpirun -np 4 abacus > log.txt`.
    - Execute tests/integrate/tools/catch_properties.sh script to generate the reference results. At the new test directory, run `bash ../tools/catch_properties.sh result.ref`.
      A `result.ref` file may be like:
        ```text
         etotref -3439.007931317310
         etotperatomref -3439.0079313173
         totaltimeref 2.78
        ``` 
    - If you want to test the correctness of some output files, you need to do extra below steps:
        1. add the corresponding comparison method in `catch_properties.sh`. For example, to verify whether the output of the BANDS_1.dat file is correct, you need to add the following code in `catch_properties.sh`:
            ```bash
            has_band=$(awk '$1=="out_band" {a=$2} END{print a}' INPUT)  # check if the BAND is outputed
            if ! test -z "$has_band"  && [  $has_band == 1 ]; then      # if band is outputed, then check if the band is correct
	            bandref=refBANDS_1.dat                                  # this file should be prepared in new test directory
	            bandcal=OUT.autotest/BANDS_1.dat                        # this file is generated by each run of test
                python3 ../tools/CompareFile.py $bandref $bandcal 8     # compare the new band file with the reference file
	            echo "CompareBand_pass $?" >>$1                         # record the comparison result, $? is the return value of last command
            fi
            ```
            `CompareFile.py` is used to determine if two files are identical. It accepts three arguments: the first two are the files to be compared, and the third specifies the precision     for comparing numerical values. The comparison fails if the difference between any two corresponding numerical values exceeds 1e-{precision} (such as: 1e-8 in previous case).  If the files are identical, the script returns 0; otherwise, it returns 1.

        2. Add the reference file (such as: `refBANDS_1.dat` in previous case) to the new test directory.

        3. Add the reference comparison result to the `result.ref` file. For example, `CompareBand_pass 0` means the comparison of the band file is passed. (This statement should be added before the `totaltimeref` line)
4. Add a `jd` file in the new test directory, which is one setence to describe the new test.
5. Add the new test to `tests/integrate/CASES_CPU.txt` file (or `tests/integrate/CASES_GPU.txt` file if it is for GPU verion).
6. Enter directory tests/integrate and run `bash Autotest.sh -r <the-new-test-name>` to check if the new test can be run successfully.



## Debugging the codes

For the unexpected results when developing ABACUS, [GDB](https://www.sourceware.org/gdb/) will come in handy.

1. Compile ABACUS with debug mode.

    ```bash
    cmake -B build -DCMAKE_BUILD_TYPE=Debug
    ```

2. After building and installing the executable, enter the input directory, and launch the debug session with `gdb abacus`. For [debugging in Visual Studio Code](https://code.visualstudio.com/docs/cpp/cpp-debug), please set [cwd](https://code.visualstudio.com/docs/cpp/launch-json-reference#_cwd) to the input directory, and [program](https://code.visualstudio.com/docs/cpp/launch-json-reference#_program-required) to the path of ABACUS executable.

3. Set breakpoints, and run ABACUS by typing "run" in GDB command line interface. If the program hits the breakpoints or exception is throwed, GDB will stop at the erroneous code line. Type "where" to show the stack backtrace, and "print i" to get the value of variable i.

4. For debugging ABACUS in multiprocessing situation, `mpirun -n 1 gdb abacus : -n 3 abacus` will attach GDB to the master process, and launch 3 other MPI processes.

For segmentation faults, ABACUS can be built with [Address Sanitizer](https://github.com/google/sanitizers/wiki/AddressSanitizer) to locate the bugs.

```bash
cmake -B build -DENABLE_ASAN=1
```

Run ABACUS as usual, and it will automatically detect the buffer overflow problems and memory leaks. It is also possible to [use GDB with binaries built by Address Sanitizer](https://github.com/google/sanitizers/wiki/AddressSanitizerAndDebugger).

[Valgrind](https://valgrind.org/) is another option for performing dynamic analysis.

## Adding a new building component

ABACUS uses CMake as its default building system. To add a new building component:

1. Add an `OPTION` to toggle the component to the `CMakeLists.txt` file under root directory. For example:

    ```cmake
    OPTION(ENABLE_NEW_COMPONENT "Enable new component" OFF)
    ```

2. Add the new component. For example:

    ```cmake
    IF (ENABLE_NEW_COMPONENT)
      add_subdirectory(module_my_new_feature) # if the feature is implemented in a subdirectory
      find_package(NewComponent REQUIRED) # if third-party libs are required
      target_link_libraries(${ABACUS_BIN_NAME} PRIVATE NewComponent) # if the component is linked
      include_directories(${NewComponent_INCLUDE_DIRS}) # if the component is included
    endif()
    ```

3. Add the required third-party libraries to Dockerfiles.
4. After the changes above are merged, submit another PR to build and test the new component in the CI pipeline.
    - For integration test and unit test: add `-DENABLE_NEW_COMPONENT=ON` to the building step at `.github/workflows/test.yml`.
    - For building test: add `-DENABLE_NEW_COMPONENT=ON` as a new configuration at `.github/workflows/build_test_cmake.yml`.

## Generating code coverage report

This feature requires using GCC compiler. We use `gcov` and `lcov` to generate code coverage report.

1. Add `-DENABLE_COVERAGE=ON` for CMake configure command.

    ```bash
    cmake -B build -DBUILD_TESTING=ON -DENABLE_COVERAGE=ON
    ```

2. Build, install ABACUS, and run test cases. Please note that since all optimizations are disabled to gather running status line by line, the performance is drastically decreased. Set a longer time out to ensure all tests are executed.

    ```bash
    cmake --build build --target test ARGS="-V --timeout 21600"
    ```

    If configuration fails unfortunately, you can find [required files](https://github.com/baixiaokuang/CMake-codecov/tree/master/cmake) (including three *.cmake and llvm-cov-wrapper), and copy these four files into `/abacus-develop/cmake`. Alternatively, you can define the path with option `-D CMAKE_CURRENT_SOURCE_DIR`.

3. Generate HTML report.

    ```bash
    cd build/
    make lcov
    ```

Now you can copy `build/lcov` to your local device, and view `build/lcov/html/all_targets/index.html`.

We use [Codecov](https://codecov.io/) to host and visualize our [**code coverage report**](https://app.codecov.io/gh/deepmodeling/abacus-develop). Analysis is scheduled after a new version releases; this [action](https://github.com/deepmodeling/abacus-develop/actions/workflows/coverage.yml) can also be manually triggered.

## Submitting a Pull Request

1. [Fork](https://docs.github.com/en/github/getting-started-with-github/fork-a-repo) the [ABACUS repository](https://github.com/deepmodeling/abacus-develop). If you already had an existing fork, [sync](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork) the fork to keep your modification up-to-date.

2. Pull your forked repository, create a new git branch, and make your changes in it:

     ```shell
     git checkout -b my-fix-branch
     ```

3. Coding your patch, including appropriate test cases and docs.
To run a subset of unit test, use `ctest -R <test-match-pattern>` to perform tests with name matched by given pattern.

4. After tests passed, commit your changes [with a proper message](#commit-message-guidelines).

5. Push your branch to GitHub:

    ```shell
    git push origin my-fix-branch
    ```

6. In GitHub, send a pull request (PR) with `deepmodeling/abacus-develop:develop` as the base repository. It is **required** to document your PR following [our guidelines](#commit-message-guidelines).

7. If more changes are needed, you can add more commits to your branch and push them to GitHub. Your PR will be updated automatically.

8. After your pull request is merged, you can safely delete your branch and sync the changes from the main (upstream) repository:

    - Delete the remote branch on GitHub either [through the GitHub web UI](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-branches-in-your-repository/deleting-and-restoring-branches-in-a-pull-request#deleting-a-branch-used-for-a-pull-request) or your local shell as follows:

        ```shell
        git push origin --delete my-fix-branch
        ```

    - Check out the master branch:

        ```shell
        git checkout develop -f
        ```

    - Delete the local branch:

        ```shell
        git branch -D my-fix-branch
        ```

    - Update your master with the latest upstream version:

        ```shell
        git pull --ff upstream develop
        ```

## Commit message guidelines

A well-formatted commit message leads a more readable history when we look through some changes, and helps us generate change log.
We follow up [The Conventional Commits specification](https://www.conventionalcommits.org) for commit message format.
This format is also required for PR title and message.
The commit message should be structured as follows:

```text
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

- Header
  - type: The general intention of this commit
    - `Feature`: A new feature
    - `Fix`: A bug fix
    - `Docs`: Only documentation changes
    - `Style`: Changes that do not affect the meaning of the code
    - `Refactor`: A code change that neither fixes a bug nor adds a feature
    - `Perf`: A code change that improves performance
    - `Test`: Adding missing tests or correcting existing tests
    - `Build`: Changes that affect the build system or external dependencies
    - `CI`: Changes to our CI configuration files and scripts
    - `Revert`: Reverting commits
  - scope: optional, could be the module which this commit changes; for example, `orbital`
  - description: A short summary of the code changes: tell others what you did in one sentence.
- Body: optional, providing detailed, additional, or contextual information about the code changes, e.g. the motivation of this commit, referenced materials, the coding implementation, and so on.
- Footer: optional, reference GitHub issues or PRs that this commit closes or is related to. [Use a keyword](https://docs.github.com/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue#linking-a-pull-request-to-an-issue-using-a-keyword) to close an issue, e.g. "Fix #753".

Here is an example:

```text
Fix(lcao): use correct scalapack interface.

`pzgemv_` and `pzgemm_` used `double*` for alpha and beta parameters but not `complex*` , this would cause error in GNU compiler.

Fix #753.
```
