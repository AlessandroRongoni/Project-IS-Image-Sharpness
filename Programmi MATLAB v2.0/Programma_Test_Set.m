function Programma_Test_Set(w)
n = 2;
for j=1:n
    A = imread(strcat('Targhe_Test\Test_',int2str(j),'.png'));    
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
aa = mean(double(A(:))); % Shift dei valori per la grafica
bb = sqrt(var(double(A(:)))); % Scalatura dei valori per la grafica
y = -sign(w(1))*(w')*z*bb + aa;
%% Trasformo il vettore in una matrice
y_m=reshape(uint8(y),[dimensione_im_x,dimensione_im_y]);
%% DISEGNO COMPARATIVO
figure;
subplot(1,2,1);  imshow(A(:,:,1)); title('Original','interpreter','latex');
subplot(1,2,2);  imshow(y_m); title('Recovered','interpreter','latex');
end