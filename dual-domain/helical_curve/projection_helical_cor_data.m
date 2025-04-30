function [xq,yq,zq,delt_t]=projection_helical_cor_data(theta,theta_start,DSD,DSO,p,cor_alpha,cor_w,cor_x,cor_y,cor_z,delt_t)
nproj=length(theta);
% [x,y,z]=size(img);

[alpha_cor,w_cor]=meshgrid(cor_alpha,cor_w);
alpha_cor=alpha_cor';
w_cor=w_cor';

rr=max(max(abs(cor_x(:)),abs(cor_y(:))));
tt=[DSO-rr:delt_t:DSO+rr];
tt=repmat(tt',[1,length(cor_alpha),length(cor_w)]);
tt=permute(tt,[2,3,1]);

x0=DSO*cos(theta);
y0=DSO*sin(theta);
% x0=-DSO*sin(theta);
% y0=DSO*cos(theta);
z0=p*(theta-theta_start)/2/pi;

xq=zeros(length(theta),size(tt,1),size(tt,2),size(tt,3));
yq=xq;
zq=xq;
delt_x=cor_x(2)-cor_x(1);
delt_y=cor_y(2)-cor_y(1);
delt_z=cor_z(2)-cor_z(1);
for i=1:nproj
    tmp_angle=alpha_cor-theta(i);
    xx=-DSD*cos(tmp_angle);
    yy=DSD*sin(tmp_angle);
    zz=w_cor;
    LL=sqrt(DSD^2+w_cor.^2);
    tx=x0(i)+tt.*xx./LL;
    ty=y0(i)+tt.*yy./LL;
    tz=z0(i)+tt.*zz./LL;
    yq(i,:,:,:)=(tx-cor_x(1))/delt_x;
    xq(i,:,:,:)=(ty-cor_y(1))/delt_y; %%%%%%% due to the direction of interp3d, xq and yq should be interchanged.
    zq(i,:,:,:)=(tz-cor_z(1))/delt_z;
end