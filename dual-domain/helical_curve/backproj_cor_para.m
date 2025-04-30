function [alpha_q,w_q]=backproj_cor_para(alpha_star,w_star,cor_alpha,cor_w)
alpha_q=(alpha_star-cor_alpha(1))/(cor_alpha(2)-cor_alpha(1));
w_q=(w_star-cor_w(1))/(cor_w(2)-cor_w(1));

