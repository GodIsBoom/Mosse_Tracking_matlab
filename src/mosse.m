% get images from source directory
datadir = '../data/';
dataset = 'Surfer';
path = [datadir dataset];
img_path = [path '/img/'];
D = dir([img_path, '*.jpg']);
seq_len = length(D(not([D.isdir])));                   %��ȡͼƬ���г���
if exist([img_path num2str(1, '%04i.jpg')], 'file'),   %num2str������ת��Ϊ�ַ�����/
    img_files = num2str((1:seq_len)', [img_path '%04i.jpg']);   %�õ����ж�Ӧ��ͼƬ
    %disp(img_files);
else
    error('No image files found in the directory.');
end

% select target from first frame
im = imread(img_files(1,:));    %��ȡ��һ֡��ͼƬ
%[X,map]=imread('forest.png')  �����XΪͼ����ɫֵ��map����ɫ��(ͨ��)
f = figure('Name', 'Select object to track'); imshow(im);   
%figure��ʾ����������/imshow��ʾ�ڵ���������ʾ��һ֡ͼƬ
rect = getrect; %getrect�����������ָ������
%����rect����ֵΪѡ�������Դ�Ϊ�����½ǵ����꣬��ȣ��߶�
close(f); clear f;
center = [rect(2)+rect(4)/2 rect(1)+rect(3)/2];%��þ��ο�����ĵ��ݣ������꣨�� ������һ֡ͼ������ĵ�

% plot gaussian
sigma = 100;
gsize = size(im);   %��ȡ��һ֡ͼƬ�ĳߴ�:�� �� 3ͨ��
[R,C] = ndgrid(1:gsize(1), 1:gsize(2)); %ndgrid ����gsize(1)xgsize(2)�ĸ�ά���󣬲���1��������/�������������
%ndgrid�÷���https://blog.csdn.net/u012183487/article/details/76149279
g = gaussC(R,C, sigma, center);  %���ø�˹����
g = mat2gray(g);%��һ��double�����������ת����ֵ��Χ��[0,1]�Ĺ�һ��double������

% randomly warp original image to create training set
if (size(im,3) == 3) %size(A,n)��size�����ؾ��������������
    img = rgb2gray(im);  %��RGBͼ����ɫͼת��Ϊ�Ҷ�ͼ��
end
%imcrop��ʾ����һ֡ѡ��ľ��δ�С�ü�ͼƬ
img = imcrop(img, rect); %��ѡ���ľ��ο�ü���һ֡�Ҷ�ͼ��С
%imcrop �� https://blog.csdn.net/llxue0925/article/details/80431508
g = imcrop(g, rect); %
G = fft2(g);   %��ά���ٸ���Ҷ�任������Ӧ���g���G
height = size(g,1);   %height=179
width = size(g,2);    %width=130
%f��ʾ����ͼ��
fi = preprocess(imresize(img, [height width])); 
%�������òü����ͼƬ�ߴ磬�ٵ���preprocess������ͼ�����ݽ��б�׼������
Ai = (G.*conj(fft2(fi))); %conj�������ڼ���"����"x�Ĺ���ֵ����F*--��F�Ĺ���
Bi = (fft2(fi).*conj(fft2(fi)));
% %������͹�ʽ�����һ��H
% N = 128;
% for i = 1:N
%     fi = preprocess(rand_warp(img));
%     Ai = Ai + (G.*conj(fft2(fi)));
%     Bi = Bi + (fft2(fi).*conj(fft2(fi)));
% end

% % MOSSE online training regimen
eta = 0.125;   %�ٷ��Ƽ���etaֵ0.125
fig = figure('Name', 'MOSSE');
mkdir(['results_' dataset]);
for i = 1:size(img_files, 1)  %��������ͼƬ
    %size(img_files, 1) ��ʾͼƬ������
    img = imread(img_files(i,:));   %��ȡ��i��ͼƬ
    im = img;
    if (size(img,3) == 3)  %��ͨ��Ϊ3
        img = rgb2gray(img);  %ת�ɻҶ�ͼ
    end
    if (i == 1)      %��Ϊ��һ��ͼʱ
        Ai = eta.*Ai;
        Bi = eta.*Bi;
    else
        Hi = Ai./Bi;   %���ǵ�һ��ͼ��ʱ������˲�H*
        fi = imcrop(img, rect);   %�ٰ�ָ�����δ�С���вü�2-376�Ķ�ӦͼƬ
        fi = preprocess(imresize(fi, [height width]));  %Ԥ����õ����
        gi = uint8(255*mat2gray(ifft2(Hi.*fft2(fi))));  %�õ���Ӧֵ��� ת����ֵ��Χ��[0,1]�Ĺ�һ��double������ 
        maxval = max(gi(:));  %�õ������Ӧֵ
        [P, Q] = find(gi == maxval);  %�ҵ������Ӧֵ����Ӧ��gi
        dx = mean(P)-height/2;
        dy = mean(Q)-width/2;

        
        rect = [rect(1)+dy rect(2)+dx width height];  %�������ĵ�����
        fi = imcrop(img, rect);     %���¸��µ�rectĿ�����вü�ͼƬ 
        fi = preprocess(imresize(fi, [height width]));
        %������Ai��Bi
        Ai = eta.*(G.*conj(fft2(fi))) + (1-eta).*Ai;
        Bi = eta.*(fft2(fi).*conj(fft2(fi))) + (1-eta).*Bi;
    end
    
    % visualization
    text_str = ['Frame: ' num2str(i)];   %ÿһ��ͼƬ����һ֡
    box_color = 'green';
    position=[1 1];
    result = insertText(im, position,text_str,'FontSize',15,'BoxColor',...
                     box_color,'BoxOpacity',0.4,'TextColor','white');
    result = insertShape(result, 'Rectangle', rect, 'LineWidth', 3); %���ƾ��� ��Ϊ3
    imwrite(result, ['results_' dataset num2str(i, '/%04i.jpg')]);
    %imwrite(result, ['results_temp' dataset num2str(i, '/%04i.jpg')]);
    imshow(result);
end
