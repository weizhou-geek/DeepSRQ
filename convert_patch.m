close all;
clear all;
clc


dirs='./structure_images/';
img_dir=dir(dirs);

for id=3:length(img_dir)
    id
    filename=img_dir(id).name;
    imgname=[dirs,filename];
    savefile = strcat('./structure_patches/',filename(1:end-4));
    mkdir(savefile)
    
    img=imread(imgname);
    imgsize=size(img);
    row=imgsize(2);
    col=imgsize(1);

    L = size(img);
    height = 32;
    width = 32;
    max_row = floor(L(1)/height);
    max_col = floor(L(2)/width);
    seg = cell(max_row,max_col);
    %patch
    for row = 1:max_row      
        for col = 1:max_col        
            seg(row,col)= {img((row-1)*height+1:row*height,(col-1)*width+1:col*width,:)};   
        end
    end
    for i=1:max_row*max_col
        imwrite(seg{i}, [savefile,'\',filename(1:end-4),'-',int2str(i),'.bmp']);
    end
end

end
