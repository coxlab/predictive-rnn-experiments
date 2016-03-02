
frame_size=28;
m=zeros(frame_size);

radius=frame_size/4;
center=round(frame_size/2);
for i=1:frame_size
    for j=1:frame_size
        d=sqrt((i-center)^2+(j-center)^2);
        if d<radius
            m(i,j)=1;
        end
    end
end

figure
imshow(m)

%saveas(gcf,[getDropboxDir 'Cox_Lab/Predictive_Networks/images/circle_' num2str(frame_size) '.jpg'])
save([getDropboxDir 'Cox_Lab/Predictive_Networks/images/circle_' num2str(frame_size) '.mat'],'m')


h=round(sin(pi/6)*radius);
l=round(cos(pi/6)*radius);
x=[center-l, center, center+l];
y=[center-h, center+l, center-h];
m=flipud(poly2mask(x,y,frame_size,frame_size));

figure
imshow(m)

%saveas(gcf,[getDropboxDir 'Cox_Lab/Predictive_Networks/images/triangle_' num2str(frame_size) '.jpg'])
save([getDropboxDir 'Cox_Lab/Predictive_Networks/images/triangle_' num2str(frame_size) '.mat'],'m')

d=round(sin(pi/4)*radius);
x=[center-d,center-d,center+d,center+d];
y=[center-d,center+d,center+d,center-d];
m=poly2mask(x,y,frame_size,frame_size);

figure
imshow(m)

%saveas(gcf,[getDropboxDir 'Cox_Lab/Predictive_Networks/images/square_' num2str(frame_size) '.jpg'])
save([getDropboxDir 'Cox_Lab/Predictive_Networks/images/square_' num2str(frame_size) '.mat'],'m')
