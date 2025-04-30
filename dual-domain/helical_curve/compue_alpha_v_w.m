function [alpha,v,w]=compue_alpha_v_w(DSO,DSD,h,theta,x,y,z)
[x_cor,y_cor,z_cor]=meshgrid(x,y,z);
% x_cor=permute(x_cor,[2,1,3]); %%%%%
% y_cor=permute(y_cor,[2,1,3]);
% z_cor=permute(z_cor,[2,1,3]);
img_shape=size(x_cor);
alpha=zeros([length(theta),img_shape],'single');
alpha=gpuArray(alpha);
v=alpha;
w=alpha;
for i=1:length(theta)
tmp=DSO-x_cor.*cos(theta(i))-y_cor.*sin(theta(i));
tmp1=atan((-x_cor.*sin(theta(i))+y_cor.*cos(theta(i)))./tmp);
tmp2=DSD.*cos(tmp1).*(z_cor-h.*theta(i))./tmp;
v(i,:,:,:)=tmp;
alpha(i,:,:,:)=tmp1;
w(i,:,:,:)=tmp2;
end