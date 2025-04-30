function rf=recon_helical(sin,theta,h,DSD,DSO,tol,maxiter,x_cor,y_cor,z_cor,alpha_cor,w_cor,phi_cor)
delt_alpha=alpha_cor(2)-alpha_cor(1);
delt_theta=theta(2)-theta(1);
sin=gpuArray(sin);
alpha_cor=gpuArray(alpha_cor);w_cor=gpuArray(w_cor);phi_cor=gpuArray(phi_cor);
x_cor=gpuArray(x_cor);y_cor=gpuArray(y_cor);z_cor=gpuArray(z_cor);

z_cor_rec=z_cor+h*theta(1);
t1=zeros(length(z_cor_rec),1);
t2=zeros(length(z_cor_rec),1);
rf=zeros(length(x_cor),length(y_cor),length(t1));
for i=1:length(t1)
[s_b,s_t]=solve_pi(x_cor,y_cor,z_cor_rec(i),h/DSO,DSO,tol,maxiter);
s_min=min(min(s_b));s_max=max(max(s_t));
t1(i)=squeeze(floor((s_min-theta(1))/delt_theta))+1;
t2(i)=squeeze(ceil((s_max-theta(1))/delt_theta))+1;
t_need=theta(t1(i):t2(i));
Lt(i)=length(t_need);
ind=index_theta(s_b,s_t,t_need);
[alpha_s,v_s,w_s]=compue_alpha_v_w(DSO,DSD,h,t_need,x_cor,y_cor,z_cor_rec(i));
% w_need=(ind~=0).*w_s;
% w_max=max(w_max,max(abs(w_need(:))));
% a_need=(ind~=0).*alpha_s;
% a_max=max(a_max,max(abs(a_need(:))));  %%%check if the scanning gemometry is set right.
rf(:,:,i)=backproject_helical(sin(:,:,[t1(i):t2(i)]),DSD,h,DSO,phi_cor,...
          alpha_cor,w_cor,delt_alpha,delt_theta,alpha_s,v_s,w_s,ind,Lt(i));
i
end