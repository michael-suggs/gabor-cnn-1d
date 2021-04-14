% Create operators for dimension d
clear all

d=4;
N=d^2;
md=exp(2*pi*i*(0:d-1)/d);
Md=diag(md);
Fd=fft(eye(d))/sqrt(d);
Td=real(Fd'*Md*Fd)
Id=eye(d);
%Zak transfrom
ZN=kron(Fd,Id);

%Create operators for dimension N
m=exp(2*pi*i*(0:N-1)/N);
M=diag(m);
FN=fft(eye(N))/d;
T=real(FN'*M*FN);

%  T^d/2 Matrix and Zak 
A=(ZN*T^(d/2)*ZN')

%Create block
b=ones(N,1)/d;
b1=ZN*b;
b2=FN*b
b3=[ones(d/2,1);zeros(N-d/2,1)];
b4=ZN'*(1:N)';
[ZN*b4 ZN*T^(d/2)*b4]

%Create Gabor matrix G from block function
b3=b3/norm(b3)/sqrt(2)
G=b3;
for j=0:2*d-1
    for k=0:d-1
        G=[G M^(k*d)*T^(-j*d/2)*b3];
    end
end
%trim G to N x 2N
G=G(:,2:2*N+1)
real(G*G')
