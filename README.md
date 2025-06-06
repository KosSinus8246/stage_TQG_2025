The version 2 is the actual version of TQG solve (@uthor : Dimitri Moreau).
The f90 version is coded by analogy to TQG solve python (@uthor : Dimitri Moreau)
(need to be debuged for the moment)

The bis in the name means that the script solves QG and TQG cases.

Important : to compile the f90 files
ifort is deprecated, we need to use ifx
DO NOT use the optimisation at order 2 and
force the order 0 or 1.

To compile the f90 files

1) ifx -O0 -qmkl program.f90 -o program
2) ./program
