function img=backproj_cor(g_c,alpha_star,v_star,w_star,cor_alpha,cor_w,index)
[ntheta,nx,ny,nz]=size(alpha_star);
% assert(length(theta)==ntheta);
% pho=zeros(ntheta,nx,ny,nz);
[ntheta,nalpha,nw]=size(g_c);
% cor_alpha=[-rDalpha:1:rDalpha]*delt_alpha;
% cor_w=[-rDw:1:rDw]*delt_w;
assert(length(cor_alpha)==nalpha);
assert(length(cor_w)==nw);
[alpha_cor,w_cor]=meshgrid(cor_alpha,cor_w);
g5=zeros([nx,ny,nz],'single');
if isgpuarray(alpha_star)
    g5=gpuArray(g5);
end
% index=gpuArray(index);
% alpha_star=gpuArray(alpha_star);
% w_star=gpuArray(w_star);
% v_star=gpuArray(v_star);
for i=1:ntheta
    tmp=squeeze(g_c(i,:,:));
%     temp1=interp2(alpha_cor,w_cor,tmp',gpuArray(alpha_star(i,:,:,:)),gpuArray(w_star(i,:,:,:)),'nearest');
    % temp1=interp2(alpha_cor,w_cor,tmp',(alpha_star(i,:,:,:)),(w_star(i,:,:,:)));
    temp1=interp2(alpha_cor,w_cor,tmp',(alpha_star(i,:,:,:)),(w_star(i,:,:,:)),'nearest');
    temp1(isnan(temp1))=0;
%     g5=g5+squeeze(temp1.*gpuArray(index(i,:,:,:))./gpuArray(v_star(i,:,:,:))); 
    g5=g5+squeeze(temp1.*index(i,:,:,:)./v_star(i,:,:,:));
end
img=g5/2/pi;
img=gather(img);