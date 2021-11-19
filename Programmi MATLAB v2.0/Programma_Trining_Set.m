%% Programma Umeyama
clear all; close all; clc;
n = 29; %numero di targhe da processare
rand_vector = randperm(29);
for j=1:n
    A = imread(strcat('Targhe_Training\Targa_',int2str(rand_vector(j)),'.png')); %Legge l'immagine nella stessa directory in cui si trova il programma
    A= A(:,:,1); %(quali pixel prendere sull'asse Y, quali pixel prendere sull'asse X, quale scomposizione dell'imagine prendere)
    dimensione_im_x=size(A,1); %Salvo la dimensione dell'immagine sull'asse X
    dimensione_im_y=size(A,2); %Salvo la dimensione dell'immagine sull'asse Y
    Blurred1 = A;
    i=1;
    for k=0:3 
        v=0;
        theta=(pi/4)*k;
        lambda=sqrt(2)^(v+1);
        gb_r0(:,:,i)=gabor_real(lambda,theta,9); 
        i=i+1;
    end
    i=1;
    for k=0:3 
        v=0;  
        theta=(pi/4)*k;
        lambda=sqrt(2)^(v+1);
        gb_i0(:,:,i)=gabor_imag(lambda,theta,9);
        i=i+1;
    end
    i=1;
    for k=0:3
        v=1; 
        theta=(pi/4)*k;
        lambda=sqrt(2)^(v+1);
        gb_r1(:,:,i)=gabor_real(lambda,theta,13);
        i=i+1;
    end
    i=1;
    for k=0:3
        v=1; 
        theta=(pi/4)*k;
        lambda=sqrt(2)^(v+1);
        gb_i1(:,:,i)=gabor_imag(lambda,theta,13);
        i=i+1;
    end
 
%% APPLICO ALL'IMMAGINE BLURRED1 OGNUNO DEI 9 FILTRI E CREO LA MATRICE X1
i=1; 
B=Blurred1(:,:,1);
X1(1,:)=B(:)'; 
for p = 1:4
    gaborMag1_imag=imfilter(B,gb_i0(:,:,p), 'replicate'); %Input array values outside the bounds of the array are assumed to equal the nearest array border value.
    i = i+1;
    X1(i,:)=gaborMag1_imag(:)'; 
end

for p = 1:4
    gaborMag1_imag=imfilter(B,gb_i1(:,:,p), 'replicate');
    i = i+1;
    X1(i,:)=gaborMag1_imag(:)'; 
end
         
for p = 1:4
    gaborMag1_real=imfilter(B,gb_r0(:,:,p), 'replicate');
    i = i+1;
    X1(i,:)=gaborMag1_real(:)';
end

for p = 1:4
    gaborMag1_real=imfilter(B,gb_r1(:,:,p),'replicate');
    i = i+1;
    X1(i,:)=gaborMag1_real(:)';
end

%% Whitening e centering

X1 = double(X1);
m1 = mean(X1')';    
X1m = X1-m1; 

C = cov(X1m'); 
[U,D,V] = svd(C); 

%Tolgo i valori inferiori a 0.0001
L=1;
while ( D(L,L)>0.0001) && (L<17)
    L=L+1;
end 

D_n=D(1:L,1:L);         
U_n=U(:,1:L);          
z=inv(sqrtm(D_n))*U_n'*X1m;

%% RETE NEURALE
if j==1
    w=randn(L,1);
    w=w/norm(w);
    w_random=w;
end
EPS=10^-6;
NIT=0;
NMAX=700;
Mi=0.0000154;  
i=1;
DIFF=100;
wait=waitbar(0,strcat('Stato processamento targa ',' ',int2str(j),'/',int2str(n)));
while (NIT<NMAX)        
        y=(w')*z;
        w_hat_new=w+Mi*z*(tanh( y' ))/dimensione_im_x*dimensione_im_y;
        w_new=w_hat_new/norm(w_hat_new);
        DIFF=norm((w_new-w),inf);
        waitbar(NIT/NMAX);
        disp(DIFF); disp(NIT);
        conv(i)=mean(log(cosh(y)));
        matrice_w(:,i)=w;
        i=i+1;
        w=w_new;
        NIT=NIT+1;
end
close(wait);
if j==1
    total_conv=conv;
    total_matrice_w=matrice_w;
else
    total_conv=[total_conv conv];
    total_matrice_w=[total_matrice_w matrice_w];
end
end
%% GRAFICO 
subplot (1,2,1); 
plot(total_matrice_w'), xlabel('Iterations','interpreter','latex'), ylabel('Entries of the weight vector $$w$$','interpreter','latex'); 
axis tight;
subplot(1,2,2); 
plot(total_conv), xlabel('Iterations','interpreter','latex'), ylabel('Value of the learning function $$\varphi(w)$$','interpreter','latex'); 
axis tight;

Programma_Test_Set(w);