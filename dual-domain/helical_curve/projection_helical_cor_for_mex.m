function pf=projection_helical_cor_for_mex(img,theta,theta_start,DSD,DSO,p,cor_alpha,cor_w,cor_x,cor_y,cor_z,tt,delt_t)
coder.gpu.kernelfun;
pf=zeros([length(cor_alpha),length(cor_w),length(theta)],'single');

nproj=length(theta);
% [x,y,z]=size(img);

[alpha_cor,w_cor]=meshgrid(cor_alpha,cor_w);
alpha_cor=alpha_cor';
w_cor=w_cor';

% rr=max(abs(cor_x(:)),abs(cor_y(:)));
% tt=[DSO-rr:delt_t:DSO+rr];
tt=repmat(tt',[1,length(cor_alpha),length(cor_w)]);
tt=permute(tt,[2,3,1]);

x0=DSO*cos(theta);
y0=DSO*sin(theta);
% x0=-DSO*sin(theta);
% y0=DSO*cos(theta);
z0=p*(theta-theta_start)/2/pi;

[x_cor,y_cor,z_cor]=meshgrid(cor_x,cor_y,cor_z);

% pf=gpuArray(pf);
% tt=gpuArray(tt);
cc=size(tt,3);
for i=1:nproj
    tmp_angle=alpha_cor-theta(i);
    xx=-DSD*cos(tmp_angle);
    yy=DSD*sin(tmp_angle);
    zz=w_cor;
    LL=sqrt(DSD^2+w_cor.^2);
    LL=repmat(LL,[1,1,size(tt,3)]);
    xq=x0(i)+tt.*repmat(xx,[1,1,cc])./LL;
    yq=y0(i)+tt.*repmat(yy,[1,1,cc])./LL;
    zq=z0(i)+tt.*repmat(zz,[1,1,cc])./LL;
    tmp=interp3(x_cor,y_cor,z_cor,img,xq,yq,zq);
%     tmp=interp3(x_cor,y_cor,z_cor,permute(img,[2,1,3]),xq,yq,zq);
%     tmp=interp3(x_cor,y_cor,z_cor,permute(img,[2,1,3]),xq,yq,zq,'nearest');
    tmp(isnan(tmp))=0;
%     t1(i,:,:,:)=tmp;
    pf(:,:,i)=sum(tmp,3)*delt_t;

end