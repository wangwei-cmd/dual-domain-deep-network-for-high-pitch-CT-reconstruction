function g3=rebin_cor(g1,DSD,h,DSO,cor_phi,cor_alpha,cor_w)
% g2=g1;
[ntheta,nalpha,nw]=size(g1);
% cor_w=[-rDw:1:rDw]*delt_w;
assert(length(cor_w)==nw);
dist=sqrt(DSD.^2+cor_w.^2);
g1=permute(g1,[1,3,2]);
g1=DSD*g1./dist;
g1=permute(g1,[1,3,2]);

assert(length(cor_alpha)==nalpha);
[alpha_cor,phi_cor]=meshgrid(cor_alpha,cor_phi);
alpha_cor=alpha_cor';
phi_cor=phi_cor';
phi_cor(phi_cor==0)=eps;  %%%avoid inf/NAN break;
w_phi=DSD*h/DSO*(phi_cor.*cos(alpha_cor)+phi_cor./tan(phi_cor).*sin(alpha_cor));
w_phi(phi_cor==0)=DSD*h/DSO*sin(alpha_cor(phi_cor==0));
% w_phi(:,rDphi+1)=(w_phi(:,rDphi)+w_phi(:,rDphi+2))/2;

g3=zeros([ntheta,nalpha,length(cor_phi)],'single');
if isgpuarray(g1)
     g3=gpuArray(g3);
end

[alpha_cor,w_cor]=meshgrid(cor_alpha,cor_w);
[alpha_cor_1,~]=meshgrid(cor_alpha,cor_phi);
for i=1:ntheta
    tmp=squeeze(g1(i,:,:))';
    tmp1=squeeze(interp2(alpha_cor,w_cor,tmp,alpha_cor_1,w_phi'));
%     g3(i,:,:)=gather(tmp1');
    g3(i,:,:)=tmp1';
end
g3(isnan(g3))=0;


% g3=zeros([length(cor_phi),ntheta,nalpha],'single');
% if isgpuarray(g1)
%      g3=gpuArray(g3);
% end
% for i=1:ntheta
%     for j=1:length(cor_alpha)
%     tmp=squeeze(g1(i,j,:))';
%     % tmp1=squeeze(interp2(alpha_cor,w_cor,tmp,alpha_cor_1,w_phi'));
%     tmp1=squeeze(interp1(cor_w,tmp,w_phi(j,:)));
%     tmp1(w_phi(j,:)>cor_w(end))=tmp(end);
%     tmp1(w_phi(j,:)<cor_w(1))=tmp(1);
%     g3(:,i,j)=tmp1';
%     end
% end
% g3(isnan(g3))=0;
% g3=permute(g3,[2,3,1]);





end









% function interp_phi=interp1d3(cor_w,g2,w_phi)
% g2_pad=padarray(g2,[0,0,1],0,'post');
% delt_w=cor_w(2)-cor_w(1);
% rDw=floor(length(cor_w)/2);
% assert((2*rDw+1)==size(g2,3));
% id=w_phi/delt_w;
% id1=ceil(id);
% id2=floor(id);
% w2=id1-id;
% w1=1-w2;
% 
% id1=id1+rDw+1;
% id1(id1>2*rDw+1)=2*rDw+2;
% id1(id1<1)=2*rDw+2;
% id2=id2+rDw+1;
% id2(id2>2*rDw+1)=2*rDw+2;
% id2(id2<1)=2*rDw+2;
% 
% % inter_phi=tmp(id1)*w1+tmp(id2)*w2;
% interp_phi=zeros(size(g2,1),size(g2,2),size(w_phi,2));
% interp_phi=gpuArray(interp_phi);
% for i=1:size(g2,1)
%     tmp1=squeeze(g2_pad(i,:,:));
%     interp_phi(i,:,:)=tmp1(id1).*w1+tmp1(id2).*w2;
% end 
% end
%     
% 
% 
