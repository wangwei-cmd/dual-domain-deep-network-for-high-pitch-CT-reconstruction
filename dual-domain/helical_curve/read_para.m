% clear
function [phi,z,DSD,DSO,DefTimes,alpha_center,w_center,dalpha,dw,nalpha,nw,Hufactor,p_factor]=read_para(pre)
% pre='./dcmproj_reference/*.dcm';
% pre='./dcmrecon_reference/*.dcm';
pre=dir(pre);
LL=length(pre);
phi=zeros(LL,1);
z=phi;
% pho=phi;
% d=phi;

name=[pre(10).folder,'/',pre(10).name];
HEADER = dicominfo(name, 'dictionary','DICOM-CT-PD-dict_v8.txt');
DSD=double(HEADER.ConstantRadialDistance);
DSO=double(HEADER.DetectorFocalCenterRadialDistance);
DefTimes=double(HEADER.NumberofSourceAngularSteps);
alpha_center=double(HEADER.DetectorCentralElement(1));
w_center=double(HEADER.DetectorCentralElement(2));
dalpha=double(HEADER.DetectorElementTransverseSpacing);
dw=double(HEADER.DetectorElementAxialSpacing);
nalpha=double(HEADER.Height);
nw=double(HEADER.Width);
Hufactor=char(HEADER.Private_7041_1001);
Hufactor=convertCharsToStrings(Hufactor);
Hufactor=str2double(Hufactor);
p_factor=double(HEADER.SpiralPitchFactor);

parfor i=1:LL
% for i=1:LL
name=[pre(i).folder,'/',pre(i).name];
HEADER = dicominfo(name, 'dictionary','DICOM-CT-PD-dict_v8.txt');
phi(i)=HEADER.DetectorFocalCenterAngularPosition;
z(i)=HEADER.DetectorFocalCenterAxialPosition;
% pho(i)=HEADER.DetectorFocalCenterRadialDistance;
% d(i)=HEADER.ConstantRadialDistance;
% tmp=HEADER.DetectorCentralElement;


% centerx(i)=tmp(1);
% centery(i)=tmp(2);
end