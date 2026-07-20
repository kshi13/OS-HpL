run('C:\Program Files\MATLAB\vlfeat-0.9.21/toolbox/vl_setup')
close all
clear all
addpath('data')
image_name = {'Airplane','Barbara','Beans','Boat','Bridge','Cameraman','Clock',...
    'Couple','Hill','House','Man','Peppers'};
rate=[5 10 15];
local_scale=10; px_h=5; py_h=5; num_s=21;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
PSNR_GL=zeros(length(image_name),length(rate));
SSIM_GL=zeros(length(image_name),length(rate));
TIME_GL=zeros(length(image_name),length(rate));
PSNR_PD=PSNR_GL; SSIM_PD=SSIM_GL; TIME_PD=TIME_GL;
PSNR_OS=PSNR_GL; SSIM_OS=SSIM_GL; TIME_OS=TIME_GL;

for i=1:length(image_name)
%     if i~=1, continue, end
    clean_image=sprintf('%s.mat',image_name{i})
    load(clean_image);
    for j=1:length(rate)
%         if j~=3, continue, end
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
        
        %% Graph Laplacian
        tic
        u_GL=inpaint_GL2(In,mask,local_scale,px_h,py_h,Io);
        psnr(u_GL,Io)
        PSNR_GL(i,j)=psnr(u_GL,Io);
        [ssims, ~] = ssim(u_GL,Io);
        SSIM_GL(i,j)=ssims;
        TIME_GL(i,j)=toc;
        
        %% HyperGraph 2 Laplacian Primal-Dual
        tic
        u_PD=u_GL; p=2;
        for ii=1:1  %% update the weight
            [W_full,id_col]=weight_hg_knn(u_PD,px_h,py_h,local_scale,num_s,p);
            u=reshape(u_PD',[n,1]);
            u=inpainting_HyperGp2L_PD(Io,u/255,W_full,id_col,g/255,id,num_s);
            u_PD=reshape(u*255,n2,n1)';
            psnr(u_PD,Io)
        end % end of updating weight
        PSNR_PD(i,j)=psnr(u_PD,Io);
        [ssims, ~] = ssim(u_PD,Io);
        SSIM_PD(i,j)=ssims;
        TIME_PD(i,j)=toc;
        
        %% HyperGraph 2 Laplacian Operator-splitting
        tic
        u_OS=u_GL; p=2;
        for ii=1:1  %% update the weight
            [W_full,id_col]=weight_hg_knn(u_OS,px_h,py_h,local_scale,num_s,p);
            [Wij_all,W_Laplace_all,coe_matrix_fix_all]=initialization_OS(W_full,id_col,id);
            clear W_full;
            u=reshape(u_OS',[n,1]);
            [u]=inpainting_HyperGp2L_OS(u/255,Io,id_col,g/255,id,...
                Wij_all,W_Laplace_all,coe_matrix_fix_all,p);
            u_OS=reshape(u*255,n2,n1)';
            psnr(u_OS,Io)
        end % end of updating weight
        PSNR_OS(i,j)=psnr(u_OS,Io);
        [ssims, ~] = ssim(u_OS,Io);
        SSIM_OS(i,j)=ssims;
        TIME_OS(i,j)=toc;
        
        %         if (i==2 && j==3) || (i==10 && j==1) || (i==12 && j==2)
        %         image_GL=sprintf('GL_%s_%d.png',image_name{i},rate(j));
        %         imwrite(uint8(u_GL),image_GL);
        %         image_PD=sprintf('PD_%s_%d.png',image_name{i},rate(j));
        %         imwrite(uint8(u_PD),image_PD);
        %         image_OS=sprintf('OS_%s_%d.png',image_name{i},rate(j));
        %         imwrite(uint8(u_OS),image_OS);
        %         end
        
    end
    
end
% save('PSNR_GL.mat','PSNR_GL');
% save('SSIM_GL.mat','SSIM_GL');
% save('TIME_GL.mat','TIME_GL');
% save('PSNR_PD.mat','PSNR_PD');
% save('SSIM_PD.mat','SSIM_PD');
% save('TIME_PD.mat','TIME_PD');
% save('PSNR_OS.mat','PSNR_OS');
% save('SSIM_OS.mat','SSIM_OS');
% save('TIME_OS.mat','TIME_OS');


