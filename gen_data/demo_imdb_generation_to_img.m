

function demo_imdb_generation_to_img(varargin)
mainPath = '/research1/db/IR_normal_small/save%03d/' ;
savePath = '/research1/db/IR_normal_small/save%03d/%03d/' ;

% 
% for i = 1 : 101,
%     for j = 1 : 9,
%          savePath_a = sprintf(savePath,i,j) ;
%          rmdir(savePath_a,'s');
%     end
% end



%1 to 101
sub_Path = '%d/' ;
%1 to 9
img_Format = '*.bmp' ;
sve_dt_Format = '%03d.png' ;
sve_gt_Format = '%03d_gt.png' ;

val_s = 1 ;
sets = ones(1,9) ;
sets(val_s) = 2 ;
sets_all = [] ;
Dt_path = [];
Gt_path = [];

chk_List = zeros(101,9) ;
for i = 1 : 101,
    for j = 1 : 9,
        
        
        [im,~] = demo_mydir([sprintf([mainPath,sub_Path],i,j),img_Format]) ;
        %1 : 10, exceptation
        if length(im) ~= 5, chk_List(i,j) = 1 ; continue ; end;
        imo = [] ;
        for k = 2 : length(im),
            tmp = imread(im(k).name) ;
            imo = cat(3,imo,tmp);
        end % 1to3 : groundtruth , 4to6 : 4,6,9 (training) 
        if size(imo,3) ~= 6, chk_List(i,j) = 2 ; continue ; end;
        
        mask = im2bw(imo(:,:,1:3),0.05) ;
        
        step_slide = 64 ;
        size_slide = 224 ;
        h_slide = floor((size(mask,1)-size_slide)/step_slide) ; 
        w_slide = floor((size(mask,2)-size_slide)/step_slide) ;
%         [ww,hh] = meshgrid(1:size(mask,2),1:size(mask,1));

        savePath_a = sprintf(savePath,i,j) ;
        if ~exist(savePath_a,'dir'), mkdir(savePath_a) ; end ;
        u =  0 ;
        for m = 1 : h_slide,
            for n = 1 : w_slide,
                hs = (m-1)*step_slide + 1;
                ws = (n-1)*step_slide + 1;
                he = hs+size_slide-1;
                we = ws+size_slide-1;
                loc_mask = mask(hs:he,ws:we) ;
                if nnz(loc_mask)/numel(loc_mask) < 0.95,
                    continue; end ;
                u = u + 1 ;
                if sets(j) == 1,
                    sets_all = [sets_all,1] ;
                else
                    sets_all = [sets_all,2] ;
                end
        
        
                gt_loc = imo(hs:he,ws:we,1:3) ;
                dt_loc = imo(hs:he,ws:we,4:6) ;
                
                imwrite(gt_loc,sprintf([savePath_a,sve_gt_Format],u)) ;
                imwrite(dt_loc,sprintf([savePath_a,sve_dt_Format],u)) ;
                Dt_path = [Dt_path;{sprintf([savePath_a,sve_dt_Format],u)}] ;
                Gt_path = [Gt_path;{sprintf([savePath_a,sve_gt_Format],u)}] ;
                
                
            end
        end
        fprintf('%d/%d : %d/%d..\n',i,101,j,9) ;
    end
end
imdb.dt = Dt_path ;
imdb.gt = Gt_path ;
imdb.sets = sets_all ;
save('imdb.mat','imdb') ;

save('chk_List.mat','chk_List') ;
        
        
        
        