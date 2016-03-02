

im=zeros(200);
x=25+rand*50;
y=25+rand*50;
r=5+rand*5;

for i=1:size(im,1)
    for j=1:size(im,2)
        d=sqrt((i-x)^2+(j-y)^2);
        if d<=r
            im(i,j)=1;
        end
    end
end

imshow(im)