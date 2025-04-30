clear;
sin_name=dir('./rec_p=0.9/sin_64_*.npy');
sin_label=[];
sin_ini=[];
img_label=[];
img_ini=[];
for i =1:length(sin_name)-8
        if i == 1
            sin_label = readNPY([sin_name(i).folder,'/',sin_name(i).name]);
            img_label = readNPY([sin_name(i).folder,'/',strrep(sin_name(i).name,'sin','img')]);
            sin_ini = readNPY([sin_name(i).folder,'/',strrep(sin_name(i).name,'64','24')]);
            img_ini = readNPY([sin_name(i).folder,'/',strrep(strrep(sin_name(i).name,'64','24'),'sin','img')]);
        else
            sin_label = cat(1,sin_label, readNPY([sin_name(i).folder,'/',sin_name(i).name]));
            img_label = cat(1,img_label, readNPY([sin_name(i).folder,'/',strrep(sin_name(i).name,'sin','img')]));
            sin_ini = cat(1,sin_ini,readNPY([sin_name(i).folder,'/',strrep(sin_name(i).name,'64','24')]));
            img_ini=cat(1,img_ini,readNPY([sin_name(i).folder,'/',strrep(strrep(sin_name(i).name,'64','24'),'sin','img')]));
        end
end
writeNPY(single(img_ini),'/media/ubuntu/UBUNTU-SERV/helical_CT_data_fast_pitch/img_ini_train.npy');
writeNPY(single(sin_ini),'/media/ubuntu/UBUNTU-SERV/helical_CT_data_fast_pitch/sin_ini_train.npy');
writeNPY(single(img_label),'/media/ubuntu/UBUNTU-SERV/helical_CT_data_fast_pitch/img_label_train.npy');
writeNPY(single(sin_label),'/media/ubuntu/UBUNTU-SERV/helical_CT_data_fast_pitch/sin_label_train.npy')



% clear;
sin_label=[];
sin_ini=[];
img_label=[];
img_ini=[];
for i =length(sin_name)-7:length(sin_name)
        if i == 1
            sin_label = readNPY([sin_name(i).folder,'/',sin_name(i).name]);
            img_label = readNPY([sin_name(i).folder,'/',strrep(sin_name(i).name,'sin','img')]);
            sin_ini = readNPY([sin_name(i).folder,'/',strrep(sin_name(i).name,'64','24')]);
            img_ini = readNPY([sin_name(i).folder,'/',strrep(strrep(sin_name(i).name,'64','24'),'sin','img')]);
        else
            sin_label = cat(1,sin_label, readNPY([sin_name(i).folder,'/',sin_name(i).name]));
            img_label = cat(1,img_label, readNPY([sin_name(i).folder,'/',strrep(sin_name(i).name,'sin','img')]));
            sin_ini = cat(1,sin_ini,readNPY([sin_name(i).folder,'/',strrep(sin_name(i).name,'64','24')]));
            img_ini=cat(1,img_ini,readNPY([sin_name(i).folder,'/',strrep(strrep(sin_name(i).name,'64','24'),'sin','img')]));
        end
end
writeNPY(single(img_ini),'/media/ubuntu/UBUNTU-SERV/helical_CT_data_fast_pitch/img_ini_test.npy');
writeNPY(single(sin_ini),'/media/ubuntu/UBUNTU-SERV/helical_CT_data_fast_pitch/sin_ini_test.npy');
writeNPY(single(img_label),'/media/ubuntu/UBUNTU-SERV/helical_CT_data_fast_pitch/img_label_test.npy');
writeNPY(single(sin_label),'/media/ubuntu/UBUNTU-SERV/helical_CT_data_fast_pitch/sin_label_test.npy')