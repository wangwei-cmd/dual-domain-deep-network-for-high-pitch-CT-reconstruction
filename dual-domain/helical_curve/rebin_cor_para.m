function [dist,w_phi]=rebin_cor_para(DSD,h,DSO,cor_phi,cor_alpha,cor_w)
% g2=g1;
% [ntheta,nalpha,nw]=size(g1);
% cor_w=[-rDw:1:rDw]*delt_w;
% assert(length(cor_w)==nw);
dist=sqrt(DSD.^2+cor_w.^2);
% g1=permute(g1,[1,3,2]);
% g2=DSD*g1./dist;
% g2=permute(g2,[1,3,2]);

% assert(length(cor_alpha)==nalpha);
[alpha_cor,phi_cor]=meshgrid(cor_alpha,cor_phi);
alpha_cor=alpha_cor';
phi_cor=phi_cor';
phi_cor(phi_cor==0)=eps;  %%%avoid inf/NAN break;
w_phi=DSD*h/DSO*(phi_cor.*cos(alpha_cor)+phi_cor./tan(phi_cor).*sin(alpha_cor));
w_phi(phi_cor==0)=DSD*h/DSO*sin(alpha_cor(phi_cor==0));
w_phi=(w_phi-cor_w(1))/(cor_w(2)-cor_w(1));