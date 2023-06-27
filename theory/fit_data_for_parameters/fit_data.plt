
#filecontent
#  1: nu_rec
#  2: P(nu_rec) for nu_ext = 0.66kHz / K_ext = 66
#  3: ...
# 15: P(nu_rec) for nu_ext = 1.18kHz / K_ext = 118
filename="activity_distributions.txt"
nu_ext(i) = 0.66 + 0.04*(i-1)
#nu_ext(i) = 66 + 4*(i-1)

###################################################
###################################################
# massive simultaneous fit

# generate "3D file"
# K nu P(nu) -> if we can parameterize the variables
# alternatively: nu P(nu) successively such that the z-dim is counting variable
# if we want control over the variables
system("rm tmp.dat; touch tmp.dat")
#do for [i=1:14]{
#  cmd1=sprintf("awk '$%d<1e5{print %f \" \" $1 \" \" $%d}' %s >> tmp.dat",i+1, nu_ext(i),  i+1, filename)
#  cmd2=sprintf("echo \"\n\" >> tmp.dat")
#  system(sprintf("%s; %s;", cmd1,cmd2))
#}

ivals="1 5 10 14"
do for [i in ivals]{
  cmd1=sprintf("awk '$%d<1e5{print $1 \" \" $%d}' %s >> tmp.dat",i+1, i+1, filename)
  cmd2=sprintf("echo \"\n\" >> tmp.dat")
  system(sprintf("%s; %s;", cmd1,cmd2))
}


N=512.
tau=6.
alpha=15.
beta=10.
b=12.
sigma=0.0001;
logP01(nu) = logA01 -2*N/sigma01**2*((1.-h01)*log(nu)+(tau01-alpha+h01*(1+beta))*nu+b01/2*nu**2)
logP02(nu) = logA02 -2*N/sigma02**2*((1.-h02)*log(nu)+(tau02-alpha+h02*(1+beta))*nu+b02/2*nu**2)
logP03(nu) = logA03 -2*N/sigma03**2*((1.-h03)*log(nu)+(tau03-alpha+h03*(1+beta))*nu+b03/2*nu**2)
logP04(nu) = logA04 -2*N/sigma04**2*((1.-h04)*log(nu)+(tau04-alpha+h04*(1+beta))*nu+b04/2*nu**2)
logP05(nu) = logA05 -2*N/sigma05**2*((1.-h05)*log(nu)+(tau05-alpha+h05*(1+beta))*nu+b05/2*nu**2)
logP06(nu) = logA06 -2*N/sigma06**2*((1.-h06)*log(nu)+(tau06-alpha+h06*(1+beta))*nu+b06/2*nu**2)
logP07(nu) = logA07 -2*N/sigma07**2*((1.-h07)*log(nu)+(tau07-alpha+h07*(1+beta))*nu+b07/2*nu**2)
logP08(nu) = logA08 -2*N/sigma08**2*((1.-h08)*log(nu)+(tau08-alpha+h08*(1+beta))*nu+b08/2*nu**2)
logP09(nu) = logA09 -2*N/sigma09**2*((1.-h09)*log(nu)+(tau09-alpha+h09*(1+beta))*nu+b09/2*nu**2)
logP10(nu) = logA10 -2*N/sigma10**2*((1.-h10)*log(nu)+(tau10-alpha+h10*(1+beta))*nu+b10/2*nu**2)
logP11(nu) = logA11 -2*N/sigma11**2*((1.-h11)*log(nu)+(tau11-alpha+h11*(1+beta))*nu+b11/2*nu**2)
logP12(nu) = logA12 -2*N/sigma12**2*((1.-h12)*log(nu)+(tau12-alpha+h12*(1+beta))*nu+b12/2*nu**2)
logP13(nu) = logA13 -2*N/sigma13**2*((1.-h13)*log(nu)+(tau13-alpha+h13*(1+beta))*nu+b13/2*nu**2)
logP14(nu) = logA14 -2*N/sigma14**2*((1.-h14)*log(nu)+(tau14-alpha+h14*(1+beta))*nu+b14/2*nu**2)

f(x,y)= (y== 0 ? logP01(x) :\
        (y== 1 ? logP05(x) :\
        (y== 2 ? logP10(x) :\
        (        logP14(x)  \
        ))));
f(x,y) = logP01(x)
sigma01=1.0;h01=(0.1); logA01=1.0; a01=0.1; b01=0.01;tau01=tau;
sigma02=1.0;h02=(0.1); logA02=1.0; a02=0.1; b02=0.01;tau02=tau;
sigma03=1.0;h03=(0.1); logA03=1.0; a03=0.1; b03=0.01;tau03=tau;
sigma04=1.0;h04=(0.1); logA04=1.0; a04=0.1; b04=0.01;tau04=tau;
sigma05=1.0;h05=(0.1); logA05=1.0; a05=0.1; b05=0.01;tau05=tau;
sigma06=1.0;h06=(0.1); logA06=1.0; a06=0.1; b06=0.01;tau06=tau;
sigma07=1.0;h07=(0.1); logA07=1.0; a07=0.1; b07=0.01;tau07=tau;
sigma08=1.0;h08=(0.1); logA08=1.0; a08=0.1; b08=0.01;tau08=tau;
sigma09=1.0;h09=(0.1); logA09=1.0; a09=0.1; b09=0.01;tau09=tau;
sigma10=1.0;h10=(0.1); logA10=1.0; a10=0.1; b10=0.01;tau10=tau;
sigma11=1.0;h11=(0.1); logA11=1.0; a11=0.1; b11=0.01;tau11=tau;
sigma12=1.0;h12=(0.1); logA12=1.0; a12=0.1; b12=0.01;tau12=tau;
sigma13=1.0;h13=(0.1); logA13=1.0; a13=0.1; b13=0.01;tau13=tau;
sigma14=1.0;h14=(0.1); logA14=1.0; a14=0.1; b14=0.01;tau14=tau;


#ATTENTION: fit on the level of population activity per bin (5ms)!
scale = 512*0.005
fit[1:100*scale][:] f(x,y) "./tmp.dat" u ($1*scale):-2:(log($2)):(1) via \
        A01, sigma01, h01, b01, tau01,\
        alpha, beta
#        A05, sigma05, h05, b05, tau05,\
#        A10, sigma10, h10, b10, tau10,\
#        A14, sigma14, h14, b14, tau14,\

#set print "coefficients"
#print "#nu_ext sigma h, a b"
#print nu_ext( 1),"\t",sigma01,"\t",logh01,"\t",a01,"\t",b01
#print nu_ext( 5),"\t",sigma05,"\t",logh05,"\t",a05,"\t",b05
#print nu_ext(10),"\t",sigma10,"\t",logh10,"\t",a10,"\t",b10
#print nu_ext(14),"\t",sigma14,"\t",logh14,"\t",a14,"\t",b14

set logscale y
set yrange [:4]
plot[:120] \
for [i in ivals] filename u 1:1+i title nu_ext(i) ls i,\
for [j=1:words(ivals)] exp(f(x*scale,j-1)) w l notitle ls word(ivals,j) lw 3

pause -1



# first fit initial range up to 50 with first 2 terms
f(nu_ext,nu)=-exp(log_h)*nu + (a+a1*nu_ext**(-s1))*nu**(1.5) + (b+b1**nu_ext**(-s1))*nu**(2.) + c*nu**(3.)
#f(x,s)= (s==0 ? f1(x) : (s==1 ? f2(x): (s==2 ? f3(x): f4(x)) ) )
log_h=log(0.10);
s1=1.0;
s2=1.0;
b=1.0;
c=0;
fit [:][:50] f(x,y) "./tmp.dat" u 1:2:3:(1) via log_h, a, s1, b, a1, b1

#h_err = h_err/FIT_STDFIT
#s_err = s_err/FIT_STDFIT
#a_err = a_err/FIT_STDFIT
#b_err = b_err/FIT_STDFIT
#c_err = c_err/FIT_STDFIT
#set print "result_fit_potential.dat"
#print "#simultaneous fit to the function f(K,nu) = -hx + 1/K^alpha (a*nu^(3/2) + b*nu^2 + c*nu^(5/2) +d*nu^3)"
#print "#Q\t h\t err(h)\t s\t err(s)\t a\t err(a)\t b\t err(b)\t c\t err(c)"
#print FIT_P, "\t", h, "\t", h_err, "\t", s, "\t", s_err, "\t", a, "\t", a_err, "\t", b, "\t", b_err, "\t", c, "\t", c_err
#set print

plot[0:50] \
for [i=1:14] filename u 1:1+i title nu_ext(i) ls i,\
for [i=1:14] f(nu_ext(i),x) w l notitle ls i lw 2,\


c=1
fit [:][:] f(x,y) "./tmp.dat" u 1:2:3:(1) via a, b, a1, b1,c

plot[:] \
for [i=1:14] filename u 1:1+i title nu_ext(i) ls i,\
for [i=1:14] f(nu_ext(i),x) w l notitle ls i lw 2,\

pause -1
