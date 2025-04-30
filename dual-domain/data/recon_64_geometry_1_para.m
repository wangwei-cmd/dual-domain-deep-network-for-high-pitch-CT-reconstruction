clear;
addpath('../helical_curve')
pre='/media/ubuntu/Expansion/LDCT/manifest-1648648375084/LDCT-and-Projection-data/';
list=dir([pre,'*/*/*Full *rojections*']);

geometry=readNPY('geometry_p=0.9.npy');
id=geometry(:,1);

% for iid=1:length(id)
iid=1;
% ppre='/media/ubuntu/Expansion/LDCT/manifest-1648648375084/LDCT-and-Projection-data/C004/08-30-2018-NA-NA-33450/1.000000-Full dose projections-47507/*.dcm';
ppre=[list(id(iid)).folder,'/',list(id(iid)).name,'/*.dcm'];
[phi,z,DSD,DSO,DefTimes,alpha_center,w_center,dalpha,dw,nalpha,nw,Hufactor,p_factor]=read_para(ppre);
% sin=read_data(ppre);

% sin=sin(end:-1:1,:,end:-1:1);
% sin=sin(:,21:44,:);
LL=length(phi);
interval=1;
% sin=sin(:,:,1:interval:end);

delt_theta=2*pi/DefTimes;

p=p_factor*dw*nw*DSO/DSD;
h=p/2/pi;
% sin_interp=sin;

x_cor=linspace(-255.5,255.5,512)*0.7813;
y_cor=linspace(-255.5,255.5,512)*0.7813;


z_interval=ceil(h*2);
z_cycle=linspace(h*2*pi,h*4*pi,z_interval);
z_cycle_1=z_cycle(1:end-1);%%%remove the last overlapped point.

x_cor=single(x_cor);y_cor=single(y_cor);z_cor=single(z_cycle_1);
x_cor=gpuArray(x_cor);y_cor=gpuArray(y_cor);z_cor=gpuArray(z_cor);

delt_alpha=dalpha/DSD;
alpha_cor=([1:nalpha]-alpha_center)*delt_alpha;
alpha_cor=-alpha_cor(end:-1:1);
w_cor=([1:nw]-w_center)*dw;
rFOV=x_cor(end); 
half_fan=asin(rFOV/DSO);
% delt_phi=delt_alpha*4;
% rDphi=ceil((pi/2+half_fan)/delt_phi);
% phi_cor=[-rDphi:1:rDphi]*delt_phi+0.25*delt_phi;
rDphi=ceil(length(w_cor)/2);
delt_phi=(pi/2+half_fan)/(rDphi-1);
phi_cor=[-rDphi:1:rDphi-1]*delt_phi+0.25*delt_phi;
% phi_cor=[-rDphi:1:rDphi]*delt_phi;
alpha_cor=single(alpha_cor);w_cor=single(w_cor);phi_cor=single(phi_cor);
alpha_cor=gpuArray(alpha_cor);
w_cor=gpuArray(w_cor);
% w_cor=w_cor(21:44);

phi_cor=gpuArray(phi_cor);

tol=1e-3; 
maxiter=100;
w_max=0;
a_max=0;
% theta=[0:LL-1]*delt_theta+phi(end)-pi/2;
theta=[0:LL-1]*delt_theta-pi/2;
theta=theta(1:interval:end);
delt_theta=theta(2)-theta(1);
z_cor_rec=z_cor+h*theta(1);

[s_b,s_t]=solve_pi(x_cor,y_cor,z_cor_rec,h/DSO,DSO,tol,maxiter);
s_min=min(min(s_b));s_max=max(max(s_t));
t1=squeeze(floor((s_min-theta(1))/delt_theta));
t2=squeeze(ceil((s_max-theta(1))/delt_theta));
theta1=min(t1(:));theta2=max(t2(:));
for i=1:length(t1)
% for i=1
t_need=single(theta([t1(i):t2(i)]));
Lt(i)=length(t_need);
ind{i}=gather(index_theta(s_b(:,:,i),s_t(:,:,i),t_need));
[a1,v1,w1]=compue_alpha_v_w(DSO,DSD,h,t_need,x_cor,y_cor,z_cor_rec(i));
alpha_s{i}=gather(a1);
v_s{i}=gather(v1);
w_s{i}=gather(w1);
% rf1(:,:,i)=backproject_helical(gpuArray(sin(:,:,[t1(i):t2(i)])),DSD,h,DSO,...
%                                  phi_cor,alpha_cor,w_cor,delt_alpha,delt_theta,...
%                                  gpuArray(alpha_s{i}),gpuArray(v_s{i}),gpuArray(w_s{i}),ind{i},Lt(i));
end

for i=1:length(t1)
    [alpha_q,w_q]=backproj_cor_para(alpha_s{i},w_s{i},alpha_cor,w_cor);
    writeNPY(gather(single(alpha_q)),['./py/64_geometry_1/alpha_q_',num2str(i),'.npy'])
    writeNPY(gather(single(w_q)),['./py/64_geometry_1/w_q_',num2str(i),'.npy'])
    writeNPY(gather(single(v_s{i})),['./py/64_geometry_1/v_s_',num2str(i),'.npy'])
    writeNPY(gather(single(ind{i})),['./py/64_geometry_1/ind_',num2str(i),'.npy'])
end
writeNPY(gather(t1-theta1),'./py/64_geometry_1/t1.npy');
writeNPY(gather(t2-theta1+1),'./py/64_geometry_1/t2.npy'); %%%%%python not include last index.
alpha_cor_shift=alpha_cor+0.5*(alpha_cor(2)-alpha_cor(1));
[dist,w_phi]=rebin_cor_para(DSD,h,DSO,phi_cor,alpha_cor_shift,w_cor);
h_matrix=makefilter(alpha_cor_shift);
[k_index,k_weight]=solve_k_line(gather(alpha_cor),gather(w_cor),gather(phi_cor),DSD,h,DSO);
writeNPY(gather(single(h_matrix)),'./py/64_geometry_1/h_matrix.npy');
writeNPY(gather(single(k_index)),'./py/64_geometry_1/k_index.npy');
writeNPY(gather(single(k_weight)),'./py/64_geometry_1/k_weight.npy');
w_phi=gather(w_phi);phi_cor=gather(phi_cor);delt_alpha=gather(delt_alpha);
delt_theta=gather(delt_theta);dist=gather(dist);DSD=gather(DSD);
alpha_cor=gather(alpha_cor);
save('./py/64_geometry_1/helical_back_cycle.mat','w_phi','phi_cor','delt_alpha',...
                                'delt_theta','dist','DSD','alpha_cor')
rmpath('../helical_curve/')