# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.28

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = C:\tools\msys64\clang64\bin\cmake.exe

# The command to remove a file.
RM = C:\tools\msys64\clang64\bin\cmake.exe -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = H:\Documents\image-processing-class

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = H:\Documents\image-processing-class\build

# Include any dependencies generated for this target.
include CMakeFiles/test.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/test.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test.dir/flags.make

CMakeFiles/test.dir/classtest.cpp.obj: CMakeFiles/test.dir/flags.make
CMakeFiles/test.dir/classtest.cpp.obj: CMakeFiles/test.dir/includes_CXX.rsp
CMakeFiles/test.dir/classtest.cpp.obj: H:/Documents/image-processing-class/classtest.cpp
CMakeFiles/test.dir/classtest.cpp.obj: CMakeFiles/test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=H:\Documents\image-processing-class\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/test.dir/classtest.cpp.obj"
	C:\tools\msys64\clang64\bin\clang++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/test.dir/classtest.cpp.obj -MF CMakeFiles\test.dir\classtest.cpp.obj.d -o CMakeFiles\test.dir\classtest.cpp.obj -c H:\Documents\image-processing-class\classtest.cpp

CMakeFiles/test.dir/classtest.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/test.dir/classtest.cpp.i"
	C:\tools\msys64\clang64\bin\clang++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E H:\Documents\image-processing-class\classtest.cpp > CMakeFiles\test.dir\classtest.cpp.i

CMakeFiles/test.dir/classtest.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/test.dir/classtest.cpp.s"
	C:\tools\msys64\clang64\bin\clang++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S H:\Documents\image-processing-class\classtest.cpp -o CMakeFiles\test.dir\classtest.cpp.s

# Object files for target test
test_OBJECTS = \
"CMakeFiles/test.dir/classtest.cpp.obj"

# External object files for target test
test_EXTERNAL_OBJECTS =

bin/test.exe: CMakeFiles/test.dir/classtest.cpp.obj
bin/test.exe: CMakeFiles/test.dir/build.make
bin/test.exe: CMakeFiles/test.dir/linkLibs.rsp
bin/test.exe: CMakeFiles/test.dir/objects1.rsp
bin/test.exe: CMakeFiles/test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=H:\Documents\image-processing-class\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable bin\test.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\test.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test.dir/build: bin/test.exe
.PHONY : CMakeFiles/test.dir/build

CMakeFiles/test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\test.dir\cmake_clean.cmake
.PHONY : CMakeFiles/test.dir/clean

CMakeFiles/test.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" H:\Documents\image-processing-class H:\Documents\image-processing-class H:\Documents\image-processing-class\build H:\Documents\image-processing-class\build H:\Documents\image-processing-class\build\CMakeFiles\test.dir\DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/test.dir/depend

