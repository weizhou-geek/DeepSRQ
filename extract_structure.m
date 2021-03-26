% Demo script

close all;
clear all;
clc


dirs='./super-resolved_images/';
img_dir=dir(dirs);

for id=3:length(img_dir)
    id
    savefileS = strcat('./structure_images/');
    mkdir(savefileS)
    
	filename=img_dir(id).name;
    imgname=[dirs, filename];
    img=imread(imgname);
    S = tsmooth(img,0.3);
    imwrite(S, [savefileS, '/', filename]);
end
