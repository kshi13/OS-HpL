run('C:\Program Files\MATLAB\vlfeat-0.9.21/toolbox/vl_setup')
close all
clear all
addpath('data')
image_name = {'Airplane','Barbara','Beans','Boat','Bridge','Cameraman','Clock',...
                'Couple','Hill','House','Man','Peppers'};
rate=[5 10 15];
local_scale=10; px_h=5; py_h=5; num_s=21;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:length(image_name)
    if i~=2, continue, end
    clean_image=sprintf('%s.mat',image_name{i})
    load(clean_image); 
    for j=1:length(rate)
        if j~=1, continue, end
        noisy_image = sprintf('%s_%d.mat',image_name{i},rate(j));
        load(noisy_image);

        [n1,n2]=size(Io); n=n1*n2;
        mask=In==Io;
        In_v=reshape(In',[n,1]);
        id=reshape(mask',[n,1]);
        id=find(id);
        g=In_v(id);

        %  Inilization speed up the convergence
        fw=zeros(n,1);
        id1=find(mask(:,:));
        id0=find(~mask(:,:));
        fw(id1)=Io(id1);
        fw(id0)=mean(Io(id1))+std(Io(id1))*randn(size(id0));
        fw=reshape(fw,n1,n2);
        In=fw;
        In(In(:)>255)=255; In(In(:)<0)=0;

        %% HyperGraph 2 Laplacian Operator-splitting
        p=1;
        [W_full,id_col]=weight_hg_knn(Io,px_h,py_h,local_scale,num_s,p);
        [Wij_all,W_Laplace_all,coe_matrix_fix_all]=initialization_OS(W_full,id_col,id);
        clear W_full;

        u=reshape(In',[n,1]);
        % p=1 -> inpainting_HyperGp1L_OS; p=2 -> inpainting_HyperGp2L_OS
        [u,energy,PSNR]=inpainting_HyperGp1L_OS(u/255,Io,id_col,g/255,id,...
                                                Wij_all,W_Laplace_all,coe_matrix_fix_all,p);                                            
        u_OS=reshape(u*255,n2,n1)';
        psnr(u_OS,Io)
    end
end

% figure(1);imshow(uint8(In));
figure(2);imshow(uint8(u_OS));
figure(3); plot(energy);
figure(4); plot(PSNR);


% figure(5); plot(energy,'LineWidth',2.0);
% xlim([0 50]);
% xlabel('Epoch Number');
% ylabel('Energy')
% set(gca,'fontsize',13,'fontname','Times','looseInset',[0 0 0 0]);
% set(gca,'looseInset',get(gca,'TightInset'));
% imwrite(uint8(u_OS),'Barbara_trueweight_p2_rate15.png');

