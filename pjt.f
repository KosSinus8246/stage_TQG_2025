       parameter(ny=60,ny2=2*ny,iwk=2*ny2*ny2)
       parameter(ijob=2)
       parameter(xk0=0.1,bet0=0.0)
       parameter(dk=0.1,nk=51)
       parameter(dB=0.1,nB=1)
       parameter(ay=1.0,U0=1.0,BB0=0.1)
       parameter(ymin=0.1,ymax=3.141592)
       parameter(F1=0.0)
* F1 = 1/Rd^2
* ny=nb pts demi-latitude, ijob = 2 --> vec pp + val pp
* xk0=premier nb d'onde, dk=increment nb d'onde, nk=nb de
* variations de k, B=beta, F=2*sigma2(harvard)
* ay=largeur courant, d=h1, h2=1-d
       logical bcv0
       parameter(bcv0=.true.)
* if bcv0=true, the velocity is odd in y=0 (sinuous mode)
* and the streamfunction even (phi(-dy)=phi(dy))
       common/prom/U1(ny),dQ1(ny),dB1(ny)
       common/resm/A(ny2,ny2),B(ny2,ny2),bet,xk

