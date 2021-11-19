function gb=gabor_imag(lambda,theta,x)

sz=fix(x);
% if mod(sz,2)==0, sz=sz+1;end

% alternatively, use a fixed size
%sz = 17;
 
[x y]=meshgrid(-fix(sz/2):fix(sz/2),fix(sz/2):-1:fix(-sz/2));
% x (right +)
% y (up +)

% Rotation 
x_theta=x*cos(theta)+y*sin(theta);
y_theta=-x*sin(theta)+y*cos(theta);
 
gb=exp(-0.5*(x.^2/lambda^2+y.^2/lambda^2)).*cos((pi*x_theta)/lambda);

%imshow(gb)  