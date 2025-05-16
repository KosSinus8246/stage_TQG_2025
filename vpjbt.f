       program vpjbt
*
       include 'pjbt.f'
*
       complex alpha(ny),z(ny,ny),kappa(ny)
       real beta(ny),wk(iwk)
       integer ier,ia,irow
*
       call inijet
       do 2 iB=1,nB
       bet=bet0+dB*float(iB-1)
       do 5 ik=1,nk
       xk=xk0+dk*float(ik-1)
       call formab
       call eigzf(A,ny,B,ny,ny,ijob,alpha,beta,z,ny,wk,ier)
       do 3 ia=1,ny
       if(beta(ia).ne.0.0) kappa(ia)=alpha(ia)/beta(ia)
3      continue
       ci=0.0
       do 4 ia=1,ny
       if(aimag(kappa(ia)).ge.ci) then
       ci=aimag(kappa(ia))
       irow=ia
       endif
4      continue
       cr=real(kappa(irow))
       tau=ci*xk
c       print *,bet,xk,tau
       print *,bet,xk,tau
c       print *,rdef,bet,xk,tau
5      continue
2      continue
1      continue
       stop
       end

c ***************************************

       subroutine inijet

       include 'pjbt.f'

       do 1 iy=1,ny
       y=ymin+(ymax-ymin)*float(iy-1)/float(ny-1)
       y=y/ay
       U1(iy)=U0*exp(-y*y)
       dQ1(iy)=2.0*U1(iy)*(1.0-2.0*y*y)
       dQ1(iy)=dQ1(iy)+F1*U1(iy)
1      continue
       return
       end

c ***************************************

       subroutine formab

       include 'pjbt.f'

       dy=(ymax-ymin)/float(ny-1)
       ek=xk*dy
c fill matrix with zeros
       do 1 i=1,ny
       do 1 j=1,ny
       A(i,j)=0.0
       B(i,j)=0.0
1      continue
c fill diagonals of A and B
       do 2 i=1,ny
       A(i,i)=U1(i)*(-2.0-ek*ek-F1*dy*dy)+dy*dy*(dQ1(i)+bet)
       B(i,i)=-2.0-ek*ek-F1*dy*dy
2      continue
c fill sub- and superdiagonals
       do 3 i=1,ny-1
       A(i,i+1)=U1(i)
       A(i+1,i)=U1(i+1)
       B(i,i+1)=1.0
       B(i+1,i)=1.0
3      continue
c boundary conditions in y=0 and y=ymax
       if(bcv0) then
       A(1,2)=2.0*A(1,2)
       B(1,2)=2.0*B(1,2)
       else
       A(1,2)=0.0
       B(1,2)=0.0
       endif
       A(ny,ny-1)=0.0
       B(ny,ny-1)=0.0
       return
       end
*
*****************************************************************
*
      subroutine eigzf  (a,ia,b,ib,n,ijob,alfa,beta,z,iz,wk,ier)
c                                  specifications for arguments
      integer            ia,ib,n,ijob,iz,ier
      real               a(ia,n),b(ib,n),alfa(1),wk(n,1),z(1),beta(n)
c                                  specifications for local variables
      integer            jer,ker,j,i,iiz,npi,ja,iz2,n2,is,ig,igz,lw,lz,
     *                   kz
      real               anorm,asum,pi,sumz,sumr,sumi,s,z11,
     1                   bsum,reps,zero,one,thous,bnorm,sums,sumj,
     2                   epsa,epsb
      data               reps/0.00000001/
      data               zero/0.0/,one/1.0/,thous/1000.0/
c                                  initialize error parameters
c                                  first executable statement
      ier = 0
      jer = 0
      ker = 0
      if (ijob.ge.0.and.ijob.le.3) go to 5
c                                  warning error - ijob is not in the
c                                    range
      ker = 66
      ijob = 1
      go to 10
    5 if (ijob.eq.0) go to 20
   10 if (iz.ge.n) go to 15
c                                  warning error - iz is less than n
c                                    eigenvectors can not be computed,
c                                    ijob set to zero
      ker = 67
      ijob = 0
   15 if (ijob.eq.3) go to 85
   20 if (ijob.ne.2) go to 30
c                                  save input a and b if ijob = 2
      do 25 j=1,n
      do 25 i=1,n
         wk(i,j) = a(i,j)
         wk(i,j+n) = b(i,j)
   25 continue
   30 continue
      iiz = n
      if (ijob.eq.0) iiz = 1
      if (ijob.eq.0.and.n.eq.1) z11 = z(1)
      call eqzqf (a,ia,b,ib,n,z,iiz)
      call eqztf (a,ia,b,ib,n,epsa,epsb,z,iiz,jer)
      call eqzvf (a,ia,b,ib,n,epsa,epsb,alfa(1),alfa(n+1),beta,z,iiz)
      if (ijob.eq.0.and.n.eq.1) z(1) = z11
      if (ijob.le.1) go to 40
c                                  move original matrices back
c                                    to a and b
      do 35 i=1,n
      do 35 j=1,n
         a(i,j) = wk(i,j)
         b(i,j) = wk(i,j+n)
   35 continue
   40 continue
c                                  convert alfa to complex format
      do 45 i=1,n
         npi = n+i
         wk(i,1) = alfa(npi)
   45 continue
      ja = n+n
      j = n
      do 50 i=1,n
         alfa(ja-1) = alfa(j)
         alfa(ja) = wk(j,1)
         ja = ja-2
         j = j-1
   50 continue
      if (ijob.eq.0) go to 115
c                                  convert z (eigenvectors) to complex
c                                    format z(iz,n)
      iz2 = iz+iz
      n2 = n+n
      j = n
   55 if (j.lt.1) go to 80
      if (alfa(j+j).eq.zero) go to 70
c                                  move pair of complex conjugate
c                                    eigenvectors
      is = iz2*(j-1)+1
      ig = n*(j-2)+1
      igz = ig+n
c                                  move complex conjugate eigenvector
      do 60 i=1,n
         z(is) = z(ig)
         z(is+1) = -z(igz)
         is = is+2
         ig = ig+1
         igz = igz+1
   60 continue
c                                  move complex eigenvector
      is = iz2*(j-2)+1
      ig = is+iz2
      do 65 i=1,n
         z(is) = z(ig)
         z(is+1) = -z(ig+1)
         is = is+2
         ig = ig+2
   65 continue
      j = j-2
      go to 55
c                                  move real eigenvector
   70 is = iz2*(j-1)+n2
      ig = n*j
      do 75 i=1,n
         z(is-1) = z(ig)
         z(is) = zero
         is = is-2
         ig = ig-1
   75 continue
      j = j-1
      go to 55
c                                  z is now in complex format z(iz,n).
   80 if (ijob.le.1) go to 115
      wk(1,1) = thous
      if (jer.ne.0) go to 115
c                                  compute max-norm of a and b
   85 anorm = zero
      bnorm = zero
      do 95 i=1,n
         asum = zero
         bsum = zero
         do 90 j=1,n
            asum = asum+abs(a(i,j))
            bsum = bsum+abs(b(i,j))
   90    continue
         anorm = amax1(anorm,asum)
         bnorm = amax1(bnorm,bsum)
   95 continue
      if (anorm.eq.zero) anorm = one
      if (bnorm.eq.zero) bnorm = one
c                                  compute performance index
      pi = zero
      lw = 1
      iz2 = iz+iz
      do 110 j=1,n
         s = zero
         sumz = zero
         lz = iz2*(j-1)+1
         do 105 l=1,n
            sumz = sumz+cabs(cmplx(z(lz),z(lz+1)))
            kz = iz2*(j-1)+1
            sumr = zero
            sums = zero
            sumi = zero
            sumj = zero
            do 100 k=1,n
               sumr = sumr+a(l,k)*z(kz)
               sumi = sumi+a(l,k)*z(kz+1)
               sums = sums+b(l,k)*z(kz)
               sumj = sumj+b(l,k)*z(kz+1)
               kz = kz+2
  100       continue
            sumr = beta(j)*sumr-alfa(lw)*sums+alfa(lw+1)*sumj
            sumi = beta(j)*sumi-alfa(lw)*sumj-alfa(lw+1)*sums
            s = amax1(s,cabs(cmplx(sumr,sumi)))
            lz = lz+2
  105    continue
         sumr = cabs(cmplx(alfa(lw),alfa(lw+1)))*bnorm
         sumr = sumz*(abs(beta(j))*anorm+sumr)
         if(sumr .ne. zero) pi = amax1(pi,s/sumr)
         lw = lw+2
  110 continue
      pi = pi/reps
      wk(1,1) = pi
  115 continue
      ier = max0(ker,jer)
      return
      end
c
      subroutine eqzqf  (a,ia,b,ib,n,z,iz)
c                                  specifications for arguments
      integer            ia,ib,iz
      real               a(ia,n),b(ib,n),z(iz,1)
c                                  specifications for local variables
      integer            i,j,nm1,l
      real               s,u2,t,v1,v2,rho,r,u1,sd,zero,one
      logical            wantx
      data               zero/0.0/,one/1.0/
c                                  first executable statement
      wantx = .false.
      if (iz.lt.n) go to 15
      wantx = .true.
c                                  initialize z, used to save
c                                     transformations
      do 10 i=1,n
         do 5 j=1,n
            z(i,j) = zero
    5    continue
         z(i,i) = one
   10 continue
c                                  reduce b to upper triangular form
   15 nm1 = n-1
      if (n.le.1) go to 110
      do 65 l=1,nm1
         l1 = l+1
         s = zero
         do 20 i=l1,n
            s = s+abs(b(i,l))
   20    continue
         if (s.eq.zero) go to 65
         s = s+abs(b(l,l))
         r = zero
         sd = one/s
         do 25 i=l,n
            b(i,l) = b(i,l)*sd
            r = r+b(i,l)**2
   25    continue
         r = sqrt(r)
         if (b(l,l).lt.zero) r = -r
         b(l,l) = b(l,l)+r
         rho = r*b(l,l)
         sd = one/rho
         do 40 j=l1,n
            t = zero
            do 30 i=l,n
               t = t+b(i,l)*b(i,j)
   30       continue
            t = -t*sd
            do 35 i=l,n
               b(i,j) = b(i,j)+t*b(i,l)
   35       continue
   40    continue
         do 55 j=1,n
            t = zero
            do 45 i=l,n
               t = t+b(i,l)*a(i,j)
   45       continue
            t = -t*sd
            do 50 i=l,n
               a(i,j) = a(i,j)+t*b(i,l)
   50       continue
   55    continue
         b(l,l) = -s*r
         do 60 i=l1,n
            b(i,l) = zero
   60    continue
   65 continue
c                                  reduce a to upper hessenberg,
c                                  keep b triangular
      if (n.le.2) go to 110
      nm2 = n-2
      do 105 k=1,nm2
         k1 = k+1
         nk1 = n-k1
         do 100 lb=1,nk1
            l = n-lb
            l1 = l+1
            call vhsh2r (a(l,k),a(l1,k),u1,u2,v1,v2)
            if (u1.ne.one) go to 80
            do 70 j=k,n
               t = a(l,j)+u2*a(l1,j)
               a(l,j) = a(l,j)+t*v1
               a(l1,j) = a(l1,j)+t*v2
   70       continue
            a(l1,k) = zero
            do 75 j=l,n
               t = b(l,j)+u2*b(l1,j)
               b(l,j) = b(l,j)+t*v1
               b(l1,j) = b(l1,j)+t*v2
   75       continue
   80       call vhsh2r (b(l1,l1),b(l1,l),u1,u2,v1,v2)
            if (u1.ne.one) go to 100
            do 85 i=1,l1
               t = b(i,l1)+u2*b(i,l)
               b(i,l1) = b(i,l1)+t*v1
               b(i,l) = b(i,l)+t*v2
   85       continue
            b(l1,l) = zero
            do 90 i=1,n
               t = a(i,l1)+u2*a(i,l)
               a(i,l1) = a(i,l1)+t*v1
               a(i,l) = a(i,l)+t*v2
   90       continue
            if (.not.wantx) go to 100
            do 95 i=1,n
               t = z(i,l1)+u2*z(i,l)
               z(i,l1) = z(i,l1)+t*v1
               z(i,l) = z(i,l)+t*v2
   95       continue
  100    continue
  105 continue
  110 return
      end
c
      subroutine eqztf  (a,ia,b,ib,n,epsa,epsb,z,iz,ier)
c                                  specifications for arguments
      integer            ia,ib,n,iz,ier
      real               a(ia,n),b(ib,n),z(iz,1),epsa,epsb
c                                  specifications for local variables
      integer            i,j,m,iter,lb,l,l1,m1,lor1,morn,k,k1,k2,
     *                   k3,km1
      real               t,v1,a10,a21,a34,b11,b34,old1,bnorm,
     1                   u1,v2,a11,a22,a43,b12,b44,old2,const,
     2                   u2,v3,a12,a30,a44,b22,eps,zero,one,
     3                   u3,ani,a20,a33,bni,b33,anorm
      logical            wantx,mid
      data               eps/0.00000001/
      data               zero/0.0/,one/1.0/
c                                  first executable statement
      ier = 0
      wantx = .false.
      if (iz.ge.n) wantx = .true.
c                                  initialize iter. compute epsa,epsb
      anorm = zero
      bnorm = zero
      do 10 i=1,n
         ani = zero
         if (i.ne.1) ani = abs(a(i,i-1))
         bni = zero
         do 5 j=i,n
            ani = ani+abs(a(i,j))
            bni = bni+abs(b(i,j))
    5    continue
         if (ani.gt.anorm) anorm = ani
         if (bni.gt.bnorm) bnorm = bni
   10 continue
      if (anorm.eq.zero) anorm = eps
      if (bnorm.eq.zero) bnorm = eps
      epsa = eps*anorm
      epsb = eps*bnorm
c                                  reduce a to quasi-triangular,
c                                     keep b triangular
      m = n
      iter = 0
   15 if (m.le.2) go to 110
c                                  check for convergence or reducibility
      do 20 lb=1,m
         l = m+1-lb
         if (l.eq.1) go to 30
         if (abs(a(l,l-1)).le.epsa) go to 25
   20 continue
   25 a(l,l-1) = zero
      if (l.lt.m-1) go to 30
      m = l-1
      iter = 0
      go to 15
c                                  check for small top of b
   30 if (abs(b(l,l)).gt.epsb) go to 45
      b(l,l) = zero
      l1 = l+1
      call vhsh2r (a(l,l),a(l1,l),u1,u2,v1,v2)
      if (u1.ne.one) go to 40
      do 35 j=l,n
         t = a(l,j)+u2*a(l1,j)
         a(l,j) = a(l,j)+t*v1
         a(l1,j) = a(l1,j)+t*v2
         t = b(l,j)+u2*b(l1,j)
         b(l,j) = b(l,j)+t*v1
         b(l1,j) = b(l1,j)+t*v2
   35 continue
   40 l = l1
      go to 25
c                                  begin one qz step. iteration strategy
   45 m1 = m-1
      l1 = l+1
      const = 0.75
      iter = iter+1
      if (iter.eq.1) go to 50
      if (abs(a(m,m-1)).lt.const*old1) go to 50
      if (abs(a(m-1,m-2)).lt.const*old2) go to 50
      if (iter.eq.10) go to 55
      if (iter.gt.30) go to 105
c                                  zeroth column of a
   50 b11 = b(l,l)
      b22 = b(l1,l1)
      if (abs(b22).lt.epsb) b22 = epsb
      b33 = b(m1,m1)
      if (abs(b33).lt.epsb) b33 = epsb
      b44 = b(m,m)
      if (abs(b44).lt.epsb) b44 = epsb
      a11 = a(l,l)/b11
      a12 = a(l,l1)/b22
      a21 = a(l1,l)/b11
      a22 = a(l1,l1)/b22
      a33 = a(m1,m1)/b33
      a34 = a(m1,m)/b44
      a43 = a(m,m1)/b33
      a44 = a(m,m)/b44
      b12 = b(l,l1)/b22
      b34 = b(m1,m)/b44
      a10 = ((a33-a11)*(a44-a11)-a34*a43+a43*b34*a11)/a21+a12-a11*b12
      a20 = (a22-a11-a21*b12)-(a33-a11)-(a44-a11)+a43*b34
      a30 = a(l+2,l1)/b22
      go to 60
c
c                                  ad hoc shift
   55 a10 = zero
      a20 = one
      a30 = 1.1605
   60 old1 = abs(a(m,m-1))
      old2 = abs(a(m-1,m-2))
      if (.not.wantx) lor1 = l
      if (wantx) lor1 = 1
      if (.not.wantx) morn = m
      if (wantx) morn = n
c                                  begin main loop
      do 100 k=l,m1
         mid = k.ne.m1
         k1 = k+1
         k2 = k+2
         k3 = k+3
         if (k3.gt.m) k3 = m
         km1 = k-1
         if (km1.lt.l) km1 = l
c                                  zero a(k+1,k-1) and a(k+2,k-1)
         if (k.eq.l) call vhsh3r (a10,a20,a30,u1,u2,u3,v1,v2,v3)
         if (k.gt.l.and.k.lt.m1) call vhsh3r (a(k,km1),a(k1,km1),a(k2,km
     1   1),u1,u2,u3,v1,v2,v3)
         if (k.eq.m1) call vhsh2r (a(k,km1),a(k1,km1),u1,u2,v1,v2)
         if (u1.ne.one) go to 70
         do 65 j=km1,morn
            t = a(k,j)+u2*a(k1,j)
            if (mid) t = t+u3*a(k2,j)
            a(k,j) = a(k,j)+t*v1
            a(k1,j) = a(k1,j)+t*v2
            if (mid) a(k2,j) = a(k2,j)+t*v3
            t = b(k,j)+u2*b(k1,j)
            if (mid) t = t+u3*b(k2,j)
            b(k,j) = b(k,j)+t*v1
            b(k1,j) = b(k1,j)+t*v2
            if (mid) b(k2,j) = b(k2,j)+t*v3
   65    continue
         if (k.eq.l) go to 70
         a(k1,k-1) = zero
         if (mid) a(k2,k-1) = zero
c                                  zero b(k+2,k+1) and b(k+2,k)
   70    if (k.eq.m1) go to 85
         call vhsh3r (b(k2,k2),b(k2,k1),b(k2,k),u1,u2,u3,v1,v2,v3)
         if (u1.ne.one) go to 85
         do 75 i=lor1,k3
            t = a(i,k2)+u2*a(i,k1)+u3*a(i,k)
            a(i,k2) = a(i,k2)+t*v1
            a(i,k1) = a(i,k1)+t*v2
            a(i,k) = a(i,k)+t*v3
            t = b(i,k2)+u2*b(i,k1)+u3*b(i,k)
            b(i,k2) = b(i,k2)+t*v1
            b(i,k1) = b(i,k1)+t*v2
            b(i,k) = b(i,k)+t*v3
   75    continue
         b(k2,k) = zero
         b(k2,k1) = zero
         if (.not.wantx) go to 85
         do 80 i=1,n
            t = z(i,k2)+u2*z(i,k1)+u3*z(i,k)
            z(i,k2) = z(i,k2)+t*v1
            z(i,k1) = z(i,k1)+t*v2
            z(i,k) = z(i,k)+t*v3
   80    continue
c                                  zero b(k+1,k)
   85    call vhsh2r (b(k1,k1),b(k1,k),u1,u2,v1,v2)
         if (u1.ne.one) go to 100
         do 90 i=lor1,k3
            t = a(i,k1)+u2*a(i,k)
            a(i,k1) = a(i,k1)+t*v1
            a(i,k) = a(i,k)+t*v2
            t = b(i,k1)+u2*b(i,k)
            b(i,k1) = b(i,k1)+t*v1
            b(i,k) = b(i,k)+t*v2
   90    continue
         b(k1,k) = zero
         if (.not.wantx) go to 100
         do 95 i=1,n
            t = z(i,k1)+u2*z(i,k)
            z(i,k1) = z(i,k1)+t*v1
            z(i,k) = z(i,k)+t*v2
   95    continue
c                                  end main loop
  100 continue
c                                  end one qz step
      go to 15
  105 ier = 128+m
  110 return
      end
c
      subroutine eqzvf  (a,ia,b,ib,n,epsa,epsb,alfr,alfi,beta,z,iz)
c                                  specifications for arguments
      integer            ia,ib,n,iz
      real               a(ia,n),b(ib,n),alfr(n),alfi(n),beta(n),z(iz,1)
      real               epsa,epsb
c                                  specifications for local variables
      integer            m,l,i,j,l1,k,mr,mi,iret
      real               e,cz,ti,v1,a21,b11,sqr,szr,a11r,a21r,
     1                   an,ei,tr,v2,a22,b12,ssi,a12i,a22i,
     2                   c,r,bn,er,u1,a11,bdi,b22,ssr,a12r,
     3                   a22r,d,t,cq,u2,a12,bdr,sqi,szi,a11i,a21i,
     4                   di,sli,tlk,tklr,tllr,dr,slr,tll,almi,
     5                   tkki,tlki,s,sk,ski,tkk,almr,tkkr,tlkr,sl,skr,
     6                   tkl,alfm,tkli,tlli,xr,xi,yr,yi,zr,zi,
     7                   h,betm,zero,one,f,half
      logical            wantx,flip
      data               zero/0.0/,half/0.5/,one/1.0/
c                                  first executable statement
      wantx = .false.
      if (iz.ge.n) wantx = .true.
c                                  find eigenvalues of quasi-triangular
c                                     matrices
      m = n
    5 continue
      if (m.eq.1) go to 10
      if (a(m,m-1).ne.zero) go to 15
c                                  one-by-one submatrix, one real root
   10 alfr(m) = a(m,m)
      if(b(m,m).lt.zero) alfr(m) = -alfr(m)
      beta(m) = abs(b(m,m))
      alfi(m) = zero
      m = m-1
      go to 70
c                                  two-by-two submatrix
   15 l = m-1
      if (abs(b(l,l)).gt.epsb) go to 20
      b(l,l) = zero
      call vhsh2r (a(l,l),a(m,l),u1,u2,v1,v2)
      go to 50
   20 if (abs(b(m,m)).gt.epsb) go to 25
      b(m,m) = zero
      call vhsh2r (a(m,m),a(m,l),u1,u2,v1,v2)
      bn = zero
      go to 30
   25 an = abs(a(l,l))+abs(a(l,m))+abs(a(m,l))+abs(a(m,m))
      bn = abs(b(l,l))+abs(b(l,m))+abs(b(m,m))
      f = one/an
      a11 = a(l,l)*f
      a12 = a(l,m)*f
      a21 = a(m,l)*f
      a22 = a(m,m)*f
      f = one/bn
      b11 = b(l,l)*f
      b12 = b(l,m)*f
      b22 = b(m,m)*f
      e = a11/b11
      c = ((a22-e*b22)/b22-(a21*b12)/(b11*b22))*half
      d = c*c+(a21*(a12-e*b12))/(b11*b22)
      if (d.lt.zero) go to 65
c                                  two real roots
c                                  zero both a(m,l) and b(m,l)
      if (c.ge.zero) e = e+(c+sqrt(d))
      if (c.lt.zero) e = e+(c-sqrt(d))
      a11 = a11-e*b11
      a12 = a12-e*b12
      a22 = a22-e*b22
      flip = (abs(a11)+abs(a12)).ge.(abs(a21)+abs(a22))
      if (flip) call vhsh2r (a12,a11,u1,u2,v1,v2)
      if (.not.flip) call vhsh2r (a22,a21,u1,u2,v1,v2)
   30 if (u1.ne.one) go to 45
      do 35 i=1,m
         t = a(i,m)+u2*a(i,l)
         a(i,m) = a(i,m)+v1*t
         a(i,l) = a(i,l)+v2*t
         t = b(i,m)+u2*b(i,l)
         b(i,m) = b(i,m)+v1*t
         b(i,l) = b(i,l)+v2*t
   35 continue
      if (.not.wantx) go to 45
      do 40 i=1,n
         t = z(i,m)+u2*z(i,l)
         z(i,m) = z(i,m)+v1*t
         z(i,l) = z(i,l)+v2*t
   40 continue
   45 if (bn.eq.zero) go to 60
      flip = an.ge.abs(e)*bn
      if (flip) call vhsh2r (b(l,l),b(m,l),u1,u2,v1,v2)
      if (.not.flip) call vhsh2r (a(l,l),a(m,l),u1,u2,v1,v2)
   50 if (u1.ne.one) go to 60
      do 55 j=l,n
         t = a(l,j)+u2*a(m,j)
         a(l,j) = a(l,j)+v1*t
         a(m,j) = a(m,j)+v2*t
         t = b(l,j)+u2*b(m,j)
         b(l,j) = b(l,j)+v1*t
         b(m,j) = b(m,j)+v2*t
   55 continue
   60 a(m,l) = zero
      b(m,l) = zero
      alfr(l) = a(l,l)
      alfr(m) = a(m,m)
      if(b(l,l).lt.zero) alfr(l) = -alfr(l)
      if(b(m,m).lt.zero) alfr(m) = -alfr(m)
      beta(l) = abs(b(l,l))
      beta(m) = abs(b(m,m))
      alfi(m) = zero
      alfi(l) = zero
      m = m-2
      go to 70
c                                  two complex roots
   65 er = e+c
      ei = sqrt(-d)
      a11r = a11-er*b11
      a11i = ei*b11
      a12r = a12-er*b12
      a12i = ei*b12
      a21r = a21
      a21i = zero
      a22r = a22-er*b22
      a22i = ei*b22
      flip = (abs(a11r)+abs(a11i)+abs(a12r)+abs(a12i))
     1.ge.(abs(a21r)+abs(a22r)+abs(a22i))
      if (flip) call vhsh2c (a12r,a12i,-a11r,-a11i,cz,szr,szi)
      if (.not.flip) call vhsh2c (a22r,a22i,-a21r,-a21i,cz,szr,szi)
      flip = an.ge.(abs(er)+abs(ei))*bn
      if (flip) call vhsh2c (cz*b11+szr*b12,szi*b12,szr*b22,szi*b22,cq,s
     1qr,sqi)
      if (.not.flip) call vhsh2c (cz*a11+szr*a12,szi*a12,cz*a21+szr*a22,
     1szi*a22,cq,sqr,sqi)
      ssr = sqr*szr+sqi*szi
      ssi = sqr*szi-sqi*szr
      tr = cq*cz*a11+cq*szr*a12+sqr*cz*a21+ssr*a22
      ti = cq*szi*a12-sqi*cz*a21+ssi*a22
      bdr = cq*cz*b11+cq*szr*b12+ssr*b22
      bdi = cq*szi*b12+ssi*b22
      r = sqrt(bdr*bdr+bdi*bdi)
      beta(l) = bn*r
      alfr(l) = an*(tr*bdr+ti*bdi)/r
      alfi(l) = an*(tr*bdi-ti*bdr)/r
      tr = ssr*a11-sqr*cz*a12-cq*szr*a21+cq*cz*a22
      ti = -ssi*a11-sqi*cz*a12+cq*szi*a21
      bdr = ssr*b11-sqr*cz*b12+cq*cz*b22
      bdi = -ssi*b11-sqi*cz*b12
      r = sqrt(bdr*bdr+bdi*bdi)
      beta(m) = bn*r
      alfr(m) = an*(tr*bdr+ti*bdi)/r
      alfi(m) = an*(tr*bdi-ti*bdr)/r
      m = m-2
   70 if (m.gt.0) go to 5
      if (.not.wantx) go to 240
c                                  find eigenvectors of quasi-triangular
c                                     matrices by backsubstitution
c                                  use b for intermediate storage,
c                                     m-th vector in b(.,m)
      m = n
   75 continue
      if (alfi(m).ne.zero) go to 110
      alfm = alfr(m)
      betm = beta(m)
      b(m,m) = one
      l = m-1
      if (l.eq.0) go to 105
   80 continue
      l1 = l+1
      sl = zero
      do 85 j=l1,m
         sl = sl+(betm*a(l,j)-alfm*b(l,j))*b(j,m)
   85 continue
      if (l.eq.1) go to 90
      if (betm*a(l,l-1).ne.zero) go to 95
   90 d = betm*a(l,l)-alfm*b(l,l)
      if (d.eq.zero) d = (epsa+epsb)*half
      b(l,m) = -sl/d
      l = l-1
      go to 105
   95 k = l-1
      sk = zero
      do 100 j=l1,m
         sk = sk+(betm*a(k,j)-alfm*b(k,j))*b(j,m)
  100 continue
      tkk = betm*a(k,k)-alfm*b(k,k)
      tkl = betm*a(k,l)-alfm*b(k,l)
      tlk = betm*a(l,k)
      tll = betm*a(l,l)-alfm*b(l,l)
      d = tkk*tll-tkl*tlk
      if (d.eq.zero) d = (epsa+epsb)*half
      b(l,m) = (tlk*sk-tkk*sl)/d
      flip = abs(tkk).ge.abs(tlk)
      if (flip) b(k,m) = -(sk+tkl*b(l,m))/tkk
      if (.not.flip) b(k,m) = -(sl+tll*b(l,m))/tlk
      l = l-2
  105 if (l.gt.0) go to 80
      m = m-1
      go to 165
  110 almr = alfr(m-1)
      almi = alfi(m-1)
      betm = beta(m-1)
      mr = m-1
      mi = m
c                                  let m-th component = (0.0,-1.0) so
c                                     that b is triangular
c                                  (m-1)st = -(betm*a(m,m)-alfm*b(m,m))*
c                                     *(m-th)/(betm*a(m,m-1))
      b(m-1,mr) = almi*b(m,m)/(betm*a(m,m-1))
      b(m-1,mi) = (betm*a(m,m)-almr*b(m,m))/(betm*a(m,m-1))
      b(m,mr) = zero
      b(m,mi) = -one
      l = m-2
      if (l.eq.0) go to 160
  115 continue
      l1 = l+1
      slr = zero
      sli = zero
      do 120 j=l1,m
         tr = betm*a(l,j)-almr*b(l,j)
         ti = -almi*b(l,j)
         slr = slr+tr*b(j,mr)-ti*b(j,mi)
         sli = sli+tr*b(j,mi)+ti*b(j,mr)
  120 continue
      if (l.eq.1) go to 125
      if (betm*a(l,l-1).ne.zero) go to 135
  125 dr = betm*a(l,l)-almr*b(l,l)
      di = -almi*b(l,l)
      iret = 1
      xr = -slr
      xi = -sli
      yr = dr
      yi = di
      go to 225
  130 b(l,mr) = zr
      b(l,mi) = zi
      l = l-1
      go to 160
  135 k = l-1
      skr = zero
      ski = zero
      do 140 j=l1,m
         tr = betm*a(k,j)-almr*b(k,j)
         ti = -almi*b(k,j)
         skr = skr+tr*b(j,mr)-ti*b(j,mi)
         ski = ski+tr*b(j,mi)+ti*b(j,mr)
  140 continue
      tkkr = betm*a(k,k)-almr*b(k,k)
      tkki = -almi*b(k,k)
      tklr = betm*a(k,l)-almr*b(k,l)
      tkli = -almi*b(k,l)
      tlkr = betm*a(l,k)
      tlki = zero
      tllr = betm*a(l,l)-almr*b(l,l)
      tlli = -almi*b(l,l)
      dr = tkkr*tllr-tkki*tlli-tklr*tlkr
      di = tkkr*tlli+tkki*tllr-tkli*tlkr
      if (dr.eq.zero.and.di.eq.zero) dr = (epsa+epsb)*half
      iret = 2
      xr = tlkr*skr-tkkr*slr+tkki*sli
      xi = tlkr*ski-tkkr*sli-tkki*slr
      yr = dr
      yi = di
      go to 225
  145 b(l,mr) = zr
      b(l,mi) = zi
      flip = (abs(tkkr)+abs(tkki)).ge.abs(tlkr)
      iret = 3
      if (flip) go to 150
      xr = -slr-tllr*b(l,mr)+tlli*b(l,mi)
      xi = -sli-tllr*b(l,mi)-tlli*b(l,mr)
      yr = tlkr
      yi = tlki
      go to 225
  150 xr = -skr-tklr*b(l,mr)+tkli*b(l,mi)
      xi = -ski-tklr*b(l,mi)-tkli*b(l,mr)
      yr = tkkr
      yi = tkki
      go to 225
  155 b(k,mr) = zr
      b(k,mi) = zi
      l = l-2
  160 if (l.gt.0) go to 115
      m = m-2
  165 if (m.gt.0) go to 75
c                                  transform to original coordinate
c                                     system
      m = n
  170 continue
      do 180 i=1,n
         s = zero
         do 175 j=1,m
            s = s+z(i,j)*b(j,m)
  175    continue
         z(i,m) = s
  180 continue
      m = m-1
      if (m.gt.0) go to 170
c                                  normalize so that largest
c                                     component = 1.
      m = n
  185 continue
      s = zero
      if (alfi(m).ne.zero) go to 200
      do 190 i=1,n
         r = abs(z(i,m))
         if (r.lt.s) go to 190
         s = r
         d = z(i,m)
  190 continue
      do 195 i=1,n
         z(i,m) = z(i,m)/d
  195 continue
      m = m-1
      go to 220
  200 do 205 i=1,n
         r = abs(z(i,m-1))+abs(z(i,m))
         if (r .eq. zero) go to 205
         r = r*sqrt((z(i,m-1)/r)**2+(z(i,m)/r)**2)
         if (r.lt.s) go to 205
         s = r
         dr = z(i,m-1)
         di = z(i,m)
  205 continue
      iret = 4
      i = 0
  210 i = i+1
      xr = z(i,m-1)
      xi = z(i,m)
      yr = dr
      yi = di
      go to 225
  215 z(i,m-1) = zr
      z(i,m) = zi
      if (i.lt.n) go to 210
      m = m-2
  220 if (m.gt.0) go to 185
      go to 240
  225 if (abs(yr).lt.abs(yi)) go to 230
      h = yi/yr
      f = yr+h*yi
      zr = (xr+h*xi)/f
      zi = (xi-h*xr)/f
      go to 235
  230 h = yr/yi
      f = yi+h*yr
      zr = (h*xr+xi)/f
      zi = (h*xi-xr)/f
  235 go to (130,145,155,215), iret
  240 return
      end
c
      subroutine vhsh2c (ajr,aji,ajp1r,ajp1i,c,sr,si)
c
      real               ajr,aji,ajp1r,ajp1i,c,sr,si,r,zero,one
      data               zero,one/0.,1./
c                                  first executable statement
      if(ajp1r.eq.zero.and.ajp1i.eq.zero) go to 5
      if(ajr.eq.zero.and.aji.eq.zero) go to 10
      r=sqrt(ajr*ajr+aji*aji)
      c=r
      sr=(ajr*ajp1r+aji*ajp1i)/r
      si=(ajr*ajp1i-aji*ajp1r)/r
      r=one/sqrt(c*c+sr*sr+si*si)
      c=c*r
      sr=sr*r
      si=si*r
      go to 9005
    5 c=one
      sr=zero
      si=zero
      go to 9005
   10 c=zero
      sr=one
      si=zero
 9005 return
      end
c
      subroutine vhsh2r (aj,ajp1,uj,ujp1,vj,vjp1)
c
      real               aj,ajp1,uj,ujp1,vj,vjp1,s,r,zero,one
      data               zero,one/0.,1./
c                                  first executable statement
      uj=zero
      ujp1=zero
      vj=zero
      vjp1=zero
      if(ajp1.eq.zero) go to 9005
      s=abs(aj)+abs(ajp1)
      uj=aj/s
      ujp1=ajp1/s
      r=sqrt(uj*uj+ujp1*ujp1)
      if(uj.lt.zero) r=-r
      vj=-(uj+r)/r
      vjp1=-ujp1/r
      uj=one
      ujp1=vjp1/vj
 9005 return
      end
c
      subroutine vhsh3r (aj,ajp1,ajp2,uj,ujp1,ujp2,vj,vjp1,vjp2)
c
      real               aj,ajp1,ajp2,uj,ujp1,ujp2,vj,vjp1,vjp2,zero
      real               one,s,r,rd
      data               zero,one/0.,1./
c                                  first executable statement
      uj=zero
      ujp1=zero
      ujp2=zero
      vj=zero
      vjp1=zero
      vjp2=zero
      if(ajp1.eq.zero.and.ajp2.eq.zero) go to 9005
      s=abs(aj)+abs(ajp1)+abs(ajp2)
      rd=one/s
      uj=aj*rd
      ujp1=ajp1*rd
      ujp2=ajp2*rd
      r=sqrt(uj*uj+ujp1*ujp1+ujp2*ujp2)
      if(uj.lt.zero) r=-r
      rd=one/r
      vj=-(uj+r)*rd
      vjp1=-ujp1*rd
      vjp2=-ujp2*rd
      uj=one
      ujp1=vjp1/vj
      ujp2=vjp2/vj
 9005 return
      end

