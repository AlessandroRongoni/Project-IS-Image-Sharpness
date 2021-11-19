%% Programma Manifold

clear;  close all;  clc

[file,path] = uigetfile({'*.png;*.jpg;*.jpeg','File Immagine (*.png;*.jpg;*.jpeg)'},'Seleziona un File');
if isequal(file,0)
   disp('User selected Cancel');
else
   disp(['User selected ', fullfile(path,file)]);
end
A = imread(strcat(path,file));
A=A(:,:,1);
dimensione_im_x=size(A,1);
dimensione_im_y=size(A,2);

figure('Name', 'Original image');imshow(A);title('Original image');
choice = menu('Choose a PSF','Motion1','Gaussian(1,1)','Gaussian(2,2)','Motion2','Gaussian','No blur');
switch choice
    
    case 1
%                 figure;
                PSF1 = fspecial('motion',10,15); 
                Blurred1 = imfilter(A,PSF1,'replicate','conv'); 
%                 subplot(1,2,1); imshow(Blurred1); title('Blurred Image 1');
%                 subplot(1,2,2); imagesc(PSF1); title('Point Spread Function 1: Motion');
    
    case 2
           
%                 figure;
                PSF2 = fspecial('gaussian',[7 7],1); 
                Blurred1 = imfilter(A,PSF2,'replicate','conv');
%                 subplot(1,2,1);imshow(Blurred1); title('Blurred Image 2');
%                 subplot(1,2,2);imagesc(PSF2); title('Point Spread Function 2: Gaussian');
                
    case 3
           
%                 figure;
                PSF2 = fspecial('gaussian',[13 13],2); 
                Blurred1 = imfilter(A,PSF2,'replicate','conv');
%                 subplot(1,2,1);imshow(Blurred1); title('Blurred Image 2');
%                 subplot(1,2,2);imagesc(PSF2); title('Point Spread Function 2: Gaussian');
    
    case 4
               
%                 figure;
                PSF3 = fspecial('motion',10,30); 
                Blurred1 = imfilter(A,PSF3,'replicate','conv');
%                 subplot(1,2,1); imshow(Blurred1); title('Blurred Image 3');
%                 subplot(1,2,2); imagesc(PSF3); title('Point Spread Function 3: Motion');
    case 5
           
%                 figure;
                PSF2 = fspecial('gaussian',[20 10]); 
                Blurred1 = imfilter(A,PSF2,'replicate','conv');
%                 subplot(1,2,1);imshow(Blurred1); title('Blurred Image 2');
%                 subplot(1,2,2);imagesc(PSF2); title('Point Spread Function 4: Gaussian');
                
    otherwise
                Blurred1 = A;      
end

%% FILTRI DI GABOR

% figure('Name', 'Gabor filter with v=0');

i=1;
for k=0:3     
    v=0;            
%         subplot(2,4,k+1);
        theta=(pi/4)*k;
        lambda=sqrt(2)^(v+1);
        gb_r0(:,:,i)=gabor_real(lambda,theta,9);
%         title(sprintf('REAL with v=%d', v));
        i=i+1;
        
end

i=1;
for k=0:3       
    v=0;  
        
%         subplot(2,4,k+1+4);
        theta=(pi/4)*k;
        lambda=sqrt(2)^(v+1);
        gb_i0(:,:,i)=gabor_imag(lambda,theta,9);
%         title(sprintf('IMAGINARY with v=%d', v));
        i=i+1;
end

i=1;
% figure('Name', 'Gabor filter with v=1');
for k=0:3       
    v=1;   
%         subplot(2,4,k+1);
        theta=(pi/4)*k;
        lambda=sqrt(2)^(v+1);
        gb_r1(:,:,i)=gabor_real(lambda,theta,13);
%         title(sprintf('REAL with v=%d', v));
        i=i+1;
end

i=1;
for k=0:3       
    v=1;  
        
%         subplot(2,4,k+1+4);
        theta=(pi/4)*k;
        lambda=sqrt(2)^(v+1);
        gb_i1(:,:,i)=gabor_imag(lambda,theta,13);
%         title(sprintf('IMAGINARY with v=%d', v));
        i=i+1;
end   

%% APPLICO ALL'IMMAGINE BLURRED1 OGNUNO DEI 9 FILTRI E CREO LA MATRICE X1

i=1; 
B=Blurred1(:,:,1);
X1(1,:)=B(:)';

choice = menu('Apply Gabor filters','Apply','Do not apply');
switch choice
    case 1
%         figure;
%         subplot(2, 4, 1);
        for p = 1:4
%             subplot(2,4,p);
            gaborMag1_imag=imfilter(B,gb_i0(:,:,p), 'replicate'); 
%             imshow(gaborMag1_imag);
%             title(sprintf('IMAGINARY with v=0'));
            i = i+1;
            X1(i,:)=gaborMag1_imag(:)';    
        end
        
        for p = 1:4
%             subplot(2,4,p+4);
            gaborMag1_imag=imfilter(B,gb_i1(:,:,p), 'replicate');
%             imshow(gaborMag1_imag);
%             title(sprintf('IMAGINARY with v=1'));
            i = i+1;
            X1(i,:)=gaborMag1_imag(:)';  
        end
        
%         figure;
%         subplot(2, 4, 1);
        for p = 1:4
%             subplot(2,4,p);
            gaborMag1_real=imfilter(B,gb_r0(:,:,p), 'replicate');
%             imshow(gaborMag1_real);
%             title(sprintf('REAL with v=0'));
            i = i+1;
            X1(i,:)=gaborMag1_real(:)';
        end
        
        for p = 1:4
%             subplot(2,4,p+4);
            gaborMag1_real=imfilter(B,gb_r1(:,:,p),'replicate');
%             imshow(gaborMag1_real);
%             title(sprintf('REAL with v=1'));
            i = i+1;
            X1(i,:)=gaborMag1_real(:)';
        end
        
    otherwise
        X1(2:17,:) = zeros(16,dimensione_im_x*dimensione_im_y);
end

%% Whitening e centering

X1 = double(X1);
m1 = mean(X1')';    
X1m = X1-m1; 
C = cov(X1m'); 
[U,D, V] = svd(C); 

%Tolgo i valori inferiori a 0.0001
l=1; %é una L
while ( D(l,l)>0.0001) && (l<17)
    l=l+1;
end  

D_n=D(1:l,1:l);        
U_n=U(:,1:l);        
z=inv(sqrtm(D_n))*U_n'*X1m;

%% RETE NEURALE
EPS=10^-5;
NIT=0;
NMAX=4000;
Mi=0.0000354;
w=randn(l,1); %(L,1)
w=w/norm(w);
w_random=w;
i=1;
wait=waitbar(0,'Please wait...');
while (NIT<NMAX) 
       
        y=(w')*z;
        gradient=(z-w*y)*tanh(y')/dimensione_im_x*dimensione_im_y;
        w_new=exp_map(w,Mi*gradient);    
        DIFF=norm((w_new-w),inf);
        waitbar(NIT/NMAX);
        disp(DIFF); disp(NIT);
        matrice_w(:,i)=w;
        conv(i)=mean(log(cosh(y)));
        i=i+1;
        w=w_new;
        NIT=NIT+1;
end
close(wait)

aa = mean(double(A(:))); % Shift dei valori per la grafica
bb = sqrt(var(double(A(:)))); % Scalatura dei valori per la grafica
y = -sign(w(1))*(w')*z *bb + aa;

%% Transformo il vettore in una matrice
y_m=reshape(uint8(y),[dimensione_im_x,dimensione_im_y]);

%% GRAFICO
figure; subplot (1,2,1); plot(matrice_w','linewidth',1.5); xlim([1 NMAX]); xlabel('Iterations','interpreter','latex'), ylabel('Entries of the weight vector $$w$$','interpreter','latex'); 
%title('Grafico convergenza della rete neurale');
subplot(1,2,2); plot(conv,'linewidth',1.5), xlim([1 NMAX]); xlabel('Iterations','interpreter','latex'), ylabel('Value of the learning function $$\varphi(w)$$','interpreter','latex'); 
%title(sprintf('Grafico convergenza y'));
figure; plot(1:dimensione_im_x*dimensione_im_y,y,'r',1:dimensione_im_x*dimensione_im_y,A(:),'b');
        legend('NN output','Original image')

%% DISEGNO COMPARATIVO
figure;
subplot(1,3,1);  imshow(A(:,:,1)); title('Original','interpreter','latex');
subplot(1,3,2);  imshow(B); title('Blurred','interpreter','latex');
subplot(1,3,3);  imshow(y_m); title('Recovered','interpreter','latex');

R_colonna1 = corr2(A,Blurred1)
R_colonna2 = corr2(A,y_m)

%% Definizione di mappa esponenziale
function tak=exp_map(x,v)
tak=x*cos(norm(v))+(v*sin(norm(v)))/norm(v);
end