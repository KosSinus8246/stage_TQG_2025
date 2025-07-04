program test_fftw3
	implicit none
	
	! gfortran fftw_test.f90 -lfftw3 -lm -o fftw_test
	! lfftw3 is the fftpack and lm a math pack
	 
	
	 
	integer :: N, i
	integer :: FFTW_FORWARD=1, FFTW_ESTIMATE=0, FFTW_BACKWARD=-1
	integer(8) :: plan, plan2
	real(8) :: f0,f1,PI,tmax,Fs,t,x,y,fact,A0,A1, dt
	double complex, allocatable :: xt(:), fourier1(:)
	real(8), allocatable :: tvec(:), freq(:)
	
	
	tmax=2.
	PI = 3.141592
	N = 1000
	Fs=N/tmax 
	!N = floor(tmax*Fs) 
	dt = tmax/N
	 
	A0 = 1.	!amplitude 0
	A1 = 0.5 !amplitude 1
	f0 =2. 	!fréquence 0
	f1 = 5.	!fréquence 1

	 

	 
	allocate(xt(N)) 
	allocate(fourier1(N))
	allocate(tvec(N))
	allocate(freq(N))
	 
	! We need to define the plane were we work (spectral plane with fftw3 tools)
	 
	call dfftw_plan_dft_1d(plan,N,xt,fourier1,FFTW_FORWARD,FFTW_ESTIMATE)
	 
	
	
	
	do i=1,N
	 
		tvec(i+1) = tvec(i) + dt
		! cmplx(a,b) creates : a+i.b number
	 
	 	xt(i) = A0*cmplx(sin(2*PI*f0*tvec(i)),0) + A1*cmplx(sin(2*PI*f1*tvec(i)),0) 
	 	
	 	! to create the equivalent of
	 	! freq = np.concatenate((np.arange(0,N//2)*Fs/N, np.arange(-N//2,0)*Fs/N))
	 	
	 	if (i <= int(N/2)) then
	 		freq(i) = i*(Fs/N)	
	 	end if
	 	
	 	if (i > int(N/2)) then
	 		freq(i) = i*(Fs/N) - Fs
	 	end if
	 
	 
	end do
	 
	!--------------------------
	! Says to fftw3 to compute dft with current values
	!--------------------------
	 
	! Forward Fourier transform
	 
	call dfftw_execute(plan)
	call dfftw_destroy_plan(plan)
	
	

	
	
	
	print *, 'time, xt, freq, Re{fft}'
	! print result propely (in columns)
	do i=1,n
		write(*,*) tvec(i), real(xt(i)), freq(i), abs(real(fourier1(i)))*(N/2)
	end do
	 
	
	 
	 
	 
	print *, fourier1
	 

 
end program test_fftw3
