function h_matrix=makefilter(u_cor)
% u_cor=[-rDu:1:rDu]*delt_u;
delt_u=u_cor(2)-u_cor(1);
L=length(u_cor);
uu=[-L+1:L-1]*delt_u-0.5*delt_u;
b=1/2/delt_u;
sin_u=sin(uu);
h_k=hilbert_fun(b,sin_u);
% h_k=u_cor./sin_u.*hilbert_fun(b,u_cor);
h_k(isnan(h_k))=0;
h_matrix=zeros(L,L);
for i=1:L
    h_matrix(i,:)=h_k(i+L-1:-1:i);
end






function h_t=hilbert_fun(b,t)
h_t=(1-cos(2*pi*b*t))./(pi*t);
% h_t=1/pi./t;
h_t(t==0)=0;
L=length(h_t);
fht=fft(h_t);
% filter='ram-lak';
filter='hamming';
% filter='hann';
switch filter
    case 'ram-lak'
        % %         Do nothing
    case 'hamming'
        window=fftshift(hamming(L))';
        h_t=real(ifft(fht.*window(1:end)));
    case 'hann'
        window=fftshift(hanning(L))';
        h_t=real(ifft(fht.*window(1:end)));
%     case 'hamming'
%         window=fftshift(hamming(L+1))';
%         h_t=real(ifft(fht.*window(1:end-1)));
%     case 'hann'
%         window=fftshift(hanning(L+1))';
%         h_t=real(ifft(fht.*window(1:end-1)));

%%%%the following two freq are not valid
%     case 'shepp-logan'
%         % be careful not to divide by 0:
%         freq=fftfreq(L);
%         freq=freq(2:end)*pi;
%         window=sin(freq)/freq;
%         F=fht;
%         F(2:end)=fht(2:end).*window;
%         h_t=real(ifft(F));
%     case 'cosine'
%         freq=linspace(0,pi,L+1);
%         window=sin(freq(1:end-1));
%         h_t=real(ifft(fht.*window));

end

