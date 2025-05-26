program tqg_solve
	implicit none

	! basic parameters
	real :: dk,dy, Ly, Lk, ymin, kmin
	integer :: Ny, Nk, i, ik
	! main parameters
	real :: beta, F1star, Theta0_U0, Theta0, U0, K2
	! vectors
	real, allocatable :: Un(:), Vn(:), y_l(:), k(:)
	real, allocatable :: G12(:), G11(:), F11(:)
	! matrix
	real, allocatable :: A11(:,:), A12(:,:), A21(:,:), A22(:,:)
	real, allocatable :: B11(:,:), B12(:,:), B21(:,:), B22(:,:)
	real, allocatable :: A(:,:), B(:,:)

	! for LAPACK computation
	real, allocatable :: alphar(:), alphai(:), beta_eig(:), work(:)
	real :: dummyVL(1,1), dummyVR(1,1) 
	integer :: lwork, info

	print *, '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
	print *, '~~~~~~~~~~~~~~~TQG_SOLVE_2_BIS_FORTRAN~~~~~~~~~~~~~~~'
	print *, '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
	print *, '-----------------------------------------------------'




	!Ny = 60
	!Nk = 51
	Ny = 2
	Nk = 2

	dk = 0.1
	Lk = 0.1+dk*Nk
	Ly = 3.14
	dy = Ly/Ny

	beta = 0
	F1star = 0
	U0 = 1
	Theta0_U0 = 1
	Theta0 = Theta0_U0 * U0

	! using the previous defined vectors
	allocate(Un(Ny), Vn(Ny), y_l(Ny), k(Nk), G12(Ny), G11(Ny), F11(Ny))

	! using the previous defined matrix
	allocate(A11(Ny,Ny), A12(Ny,Ny), A21(Ny,Ny), A22(Ny,Ny))
	allocate(B11(Ny,Ny), B12(Ny,Ny), B21(Ny,Ny), B22(Ny,Ny))
	allocate(A(2*Ny, 2*Ny), B(2*Ny, 2*Ny))

	! matrices full of 0 of the problem (no need loops)
	B12 = 0
	B21 = 0

	! filling vectors
	do i=1,Ny
		! Arrays
		y_l(i+1) = y_l(i)+dy
		Un(i) = U0*exp(real(y_l(i)**2))
		Vn(i) = Un(i)*dy**2
		G12(i) = -2*y_l(i)*Theta0*exp(real(-y_l(i))**2)
		G11(i) = 2*Un(i)*(1-2*y_l(i)**2) + F1star*Un(i) + beta - G12(i)
		F11(i) = G11(i)*dy**2


	end do


	! main loop
	do ik=1, Nk

		k(ik+1) = k(ik)+dk
		K2 = (k(ik)**2 + F1star)*dy**2


		do i=1,Ny
			! B matrix
			B11(i,i) = -(2 + K2)
			B22(i,i) = 1

			if (i>0) B11(i,i-1) = 1
			if (i < Ny-1) B11(i,i+1) = 1

			! A matrix

			A11(i,i) = -Un(i) * (2+K2) + F11(i)
			if (i>0) A11(i,i-1) = Un(i)
			if (i<Ny-1) A11(i,i+1) = Un(i)


			A12(i,i) = -Vn(i)
			A21(i,i) = -G12(i)
			A22(i,i) = Un(i)


			! concatenate A
			A(1:Ny,1:Ny) = A11
			A(1:Ny,Ny+1:2*Ny) = A12
			A(Ny+1:2*Ny,1:Ny) = A21
			A(Ny+1:2*Ny,Ny+1:2*Ny) = A22

			! concatenate B

			B(1:Ny,1:Ny) = B11
			B(1:Ny,Ny+1:2*Ny) = B12
			B(Ny+1:2*Ny,1:Ny) = B21
			B(Ny+1:2*Ny,Ny+1:2*Ny) = B22





		end do

	end do



	! Allocate LAPACK output and workspace
	allocate(alphar(2*Ny), alphai(2*Ny), beta_eig(2*Ny))
	lwork = 8 * 2*Ny
	allocate(work(lwork))


	! LAPACK generalized eigensolver
    	call DGGEV('N', 'N', 2*Ny, A, 2*Ny, B, 2*Ny, alphar, alphai, beta_eig, &
               dummyVL, 1, dummyVR, 1, work, lwork, info)
               
        if (info /= 0) then
		print *, 'DGGEV failed, info = ', info
	else
		print *, 'Eigenvalues (lambda = alpha / beta):'
		do i = 1, 2*Ny
		    print *, 'λ(', i, ') = (', alphar(i), '+', alphai(i), 'i ) /', beta_eig(i)
		end do
    	end if




	print *, 'OK'

	print *, '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'




end program
