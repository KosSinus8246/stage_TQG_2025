Hello Quasi-Geostrophy enthusiast,

The version 2 is the actual version of TQG solve 1D TARANIS (@uthor : Dimitri Moreau)

The version 3 is the actual version of TQG solve 2D JULIE (@uthor : Dimitri Moreau)

The f90 version is coded by analogy to TQG solve python (@uthor : Dimitri Moreau)
(need to be debuged for the moment -- lapack issues)

The bis in the name means that the script solves QG and TQG cases.

The codes that are in ''KERNEL_...'' folders are ready to
test different values of Theta0/U0, F1star, Lstar ...

The code TQG solve named ''JULIE'' is a 2D adaption 
of the TQG solve v2 bis ''TARANIS''

To compile the f90 files

1) gfortran myfile.f90 -llapack -lblas -o program
2) ./program

Enjoy !
