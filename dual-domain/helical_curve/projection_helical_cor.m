function pf=projection_helical_cor(img,theta,theta_start,DSD,DSO,p,cor_alpha,cor_w,cor_x,cor_y,cor_z,delt_t)
img=padarray(img,[0,0,1],'both');
delt_z=cor_z(2)-cor_z(1);
cor_z=[cor_z(1)-delt_z,cor_z,cor_z(end)+delt_z];
nproj=length(theta);
% [x,y,z]=size(img);
cor_alpha=gpuArray(cor_alpha);
cor_w=gpuArray(cor_w);
[alpha_cor,w_cor]=meshgrid(cor_alpha,cor_w);
alpha_cor=alpha_cor';
w_cor=w_cor';

rr=max(max(abs(cor_x(:))),max(abs(cor_y(:))));
tt=[DSO-rr:delt_t:DSO+rr];
tt=repmat(tt',[1,length(cor_alpha),length(cor_w)]);
tt=permute(tt,[2,3,1]);

x0=DSO*cos(theta);
y0=DSO*sin(theta);
% x0=-DSO*sin(theta);
% y0=DSO*cos(theta);
z0=p*(theta-theta_start)/2/pi;

[x_cor,y_cor,z_cor]=meshgrid(cor_x,cor_y,cor_z);
% x_cor=permute(x_cor,[2,1,3]);
% y_cor=permute(y_cor,[2,1,3]);
% z_cor=permute(z_cor,[2,1,3]);
pf=zeros([length(cor_alpha),length(cor_w),nproj],'single');
pf=gpuArray(pf);
img=gpuArray(img);
% tt=gpuArray(tt);
for i=1:nproj
    tmp_angle=alpha_cor-theta(i);
    xx=-DSD*cos(tmp_angle);
    yy=DSD*sin(tmp_angle);
    zz=w_cor;
    LL=sqrt(DSD^2+w_cor.^2);
    xq=x0(i)+tt.*xx./LL;
    yq=y0(i)+tt.*yy./LL;
    zq=z0(i)+tt.*zz./LL;
    tmp=interp3(x_cor,y_cor,z_cor,img,xq,yq,zq);
    % tmp=interp3(x_cor,y_cor,z_cor,img,xq,yq,zq,'nearest');
%     tmp=interp3(x_cor,y_cor,z_cor,permute(img,[2,1,3]),xq,yq,zq,'nearest');
    tmp(isnan(tmp))=0;
%     t1(i,:,:,:)=tmp;
    pf(:,:,i)=sum(tmp,3)*delt_t;
    % if i==419
    %     debug;
    % end
end
% debug;
pf=gather(pf);