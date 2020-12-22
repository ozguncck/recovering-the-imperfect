function IoU_track_uncertainty(path,h_size,w_size)
if (mod(h_size,64)~=0)
   h_size=64*(floor(h_size/64)+1)
end    
if (mod(w_size,64)~=0)
   w_size=64*(floor(w_size/64)+1)
end  
minSegmAreaPx = 100;
outDir = path;
useFillHoles = 1;%params.useFillHoles;
% read masks'names and sort them
list = dir(strcat(path, '/*.h5.mat'));
list = struct2table(list);
filenum = cellfun(@(x)sscanf(x,'im%d.h5.mat'), list.name);
T =  table(list, filenum);
T = [table T];
T = sortrows(T, 'filenum');

% initialize tracks 
tracks = [];
maskFOI = [];
strcat(T(1,1).list.folder{1},'/',T(1,1).list.name{1})
file_1 = load(strcat(T(1,1).list.folder{1},'/',T(1,1).list.name{1}));
map_1 = zeros(h_size,w_size,size(file_1.masks,2));

%entropy_1 = zeros(h_size,w_size,size(file_1.masks,2));
borders = zeros(h_size,w_size,1);
track_ids = zeros(size(file_1.masks,2));
cells_1 = [];

%map each bb to its position and binarize (nuc, cyto or both)
for i=1:size(file_1.rois,1)
    [maxi,arguments]=max(file_1.masks{i},[],3);
    map_1(file_1.rois(i,1)+1:file_1.rois(i,3),file_1.rois(i,2)+1:file_1.rois(i,4),i)=arguments-1;

    dil = zeros(h_size,w_size,1);
    erode = zeros(h_size,w_size,1);
    dil(file_1.rois(i,1)+1:file_1.rois(i,3),file_1.rois(i,2)+1:file_1.rois(i,4),1)=logical(arguments-1);
    erode(file_1.rois(i,1)+1:file_1.rois(i,3),file_1.rois(i,2)+1:file_1.rois(i,4),1)=logical(arguments-1);
    for j=1:10
        dil=imdilate(dil,[1 1 1;1 1 1;1 1 1],'same');
    end
    for j=1:5
        erode=imerode(erode,[1 1 1;1 1 1;1 1 1],'same');
    end


    dil=dil-erode;
    borders=borders+dil;                                                                        
    %entropy_1(file_1.rois(i,1)+1:file_1.rois(i,3),file_1.rois(i,2)+1:file_1.rois(i,4),i)=file_1.entropies{i};
end

map_1 = logical(map_1);


if (useFillHoles==1)
    for i=1:size(file_1.rois,1)%size(stmp)%
        map_1(:,:,i) = imfill(map_1(:,:,i),'holes');
    end
end



for i=1:size(map_1,3)
    tmp = bwconncomp(map_1(:,:,i),8);
    cells_1{i} = cat(1,tmp.PixelIdxList{:});    
    track_ids(i)=i;
end

map_1 = double(map_1);
for i=1:size(map_1,3)
    map_1(:,:,i)=map_1(:,:,i)*i;  
end
nObjects = double(size(cells_1,2));
for i = 1:nObjects
    tracks{i} = double([i 0 i]);
end

cells_2 = [];
for i = 2:size(list,1)  

    disp(strcat(T(i,1).list.folder{1},'/',T(i,1).list.name{1}))
    file_2 = load(strcat(T(i,1).list.folder{1},'/',T(i,1).list.name{1}));
    %entropy_2 = zeros(h_size,w_size,size(file_1.masks,2));
    map_2=zeros(h_size,w_size,size(file_2.masks,2));
    tmp_track_ids = zeros(size(file_2.masks,2));
    
    tmp_borders = zeros(h_size,w_size,1);

    for j=1:size(file_2.rois,1)
        [maxi,arguments]=max(file_2.masks{j},[],3);
        map_2(file_2.rois(j,1)+1:file_2.rois(j,3),file_2.rois(j,2)+1:file_2.rois(j,4),j)=arguments-1;
        %entropy_2(file_2.rois(j,1)+1:file_2.rois(j,3),file_2.rois(j,2)+1:file_2.rois(j,4),j)=file_2.entropies{j};
        dil = zeros(h_size,w_size,1);
        erode = zeros(h_size,w_size,1);
        dil(file_2.rois(j,1)+1:file_2.rois(j,3),file_2.rois(j,2)+1:file_2.rois(j,4),1)=logical(arguments-1);
        erode(file_2.rois(j,1)+1:file_2.rois(j,3),file_2.rois(j,2)+1:file_2.rois(j,4),1)=logical(arguments-1);
        for k=1:10
            dil=imdilate(dil,[1 1 1;1 1 1;1 1 1],'same');
        end
        for k=1:5
            erode=imerode(erode,[1 1 1;1 1 1;1 1 1],'same');
        end
        dil=dil-erode;
        tmp_borders=dil+tmp_borders;
        %tmp_borders=mod(tmp_borders,2);
    end
    
    borders(:,:,end+1)=tmp_borders;
    
    map_2 = logical(map_2);
    if (useFillHoles==1)
        for j=1:size(file_2.rois,1) %size(stmp)%
            map_2(:,:,j) = imfill(map_2(:,:,j),'holes');
        end
    end 

    for j=1:size(map_2,3)
        tmp = bwconncomp(map_2(:,:,j),8);
        cells_2{j} = cat(1,tmp.PixelIdxList{:});
    end
    curr_nObjects = double(size(cells_2,2));
    lblMatrix = zeros(double(size(cells_1,2)),curr_nObjects);
    for j = 1:size(cells_1,2)
        for k = 1:size(cells_2,2)
            union = cat(1,cells_1{j},cells_2{k});
            if (size(union,1)>0)
                lblMatrix(j,k) = (size(union,1)-size(unique(union),1))/size(unique(union),1);
            end
        end
    end
    [~,max_idx] = max(lblMatrix,[],2);
    linidx = sub2ind(size(lblMatrix),1:size(lblMatrix,1),max_idx');
    tmp_lblMatrix = zeros(size(lblMatrix));
    tmp_lblMatrix(linidx) = lblMatrix(linidx);
    lblMatrix = tmp_lblMatrix;
    [max_val,max_idx] = max(lblMatrix,[],1);
    nAddObj = sum(max_val==0);
    max_idx(max_val==0) = nObjects+1:nObjects+nAddObj;
    lblMapping = max_idx;    
    
%     xx=1:nObjects + nAddObj;
%     xx=setdiff(xx,lblMapping);
%     for j =1:size(xx,2)
%         cells_2{end+1}=cells_1{j};
%         lblMapping(size(cells_2,2))=j;
%         map_2(:,:,end+1)=map_1(:,:,j);
%     end


    for j=1:curr_nObjects
        if (lblMapping(j)>nObjects)
            % Add track
            tracks{lblMapping(j)} = double([lblMapping(j) (i-1) j]);
            tmp_track_ids(j)=lblMapping(j);
            
        elseif (track_ids(lblMapping(j))>=1 && track_ids(lblMapping(j))<=nObjects)
            % Update track
            track_entry = tracks{track_ids(lblMapping(j))};
            track_entry(end+1) = j;
            tracks{track_ids(lblMapping(j))} = track_entry;
            tmp_track_ids(j)=track_ids(lblMapping(j));

        end
    end    
    map_2 = double(map_2);
    for j=1:size(lblMapping,2)
        map_2(:,:,j)= map_2(:,:,j)*tmp_track_ids(j);
    end
    
    nObjects = nObjects + nAddObj; 
    
    
%     frameIdxStr = sprintf('%03d', i-1);
%     fn = ['mask' frameIdxStr '.tif'];
%     [maximum, index] = max(sum(sum(map_2,1),2));
%     fnOut = [outDir fn];
%     imwrite(map_2(:,:,index), fnOut,'WriteMode','overwrite');
% 
%     frameIdxStr = sprintf('%03d', i-1);
%     fn = ['uncertainty' frameIdxStr '.tiff'];
%     fnOut = [outDir fn];
%     imwrite(entropy_2(:,:,index), fnOut,'WriteMode','overwrite');
    
%     frameIdxStr = sprintf('%03d', i-1);
%     fn = ['mask' frameIdxStr '.tif'];
%     IS_mask = max(map_2,[],3);
%     fnOut = [outDir fn];
%     imwrite(uint16(IS_mask), fnOut,'WriteMode','overwrite');
%     
%     frameIdxStr = sprintf('%03d', i-1);
%     fn = ['uncertainty' frameIdxStr '.tiff'];
%     fnOut = [outDir fn];
%     imwrite(max(entropy_2,[],3), fnOut,'WriteMode','overwrite');
   
    cells_1 = cells_2;
    cells_2 = [];
    map_1 = map_2;
    track_ids = tmp_track_ids;
end

%
% Store tracking result
%
borders(borders>1)=[0];
save([outDir 'tracks.mat'], 'tracks');
save([outDir 'borders.mat'], 'borders');

for i=1:size(tracks,2)
    
   track = zeros(h_size,w_size,1);
   entropies = zeros(h_size,w_size,1);
   if size(tracks{i},2)>= size(list,1)-1 && tracks{i}(2)==0
       for j=0:size(tracks{i},2)-3
           mask = zeros(h_size,w_size,1);
           entropy = zeros(h_size,w_size,1);
           file= load(strcat(T(tracks{i}(2)+j+1,1).list.folder{1},'/',T(tracks{i}(2)+j+1,1).list.name{1}));
           [maxi,arguments]=max(file.masks{tracks{i}(3+j)},[],3);
           mask(file.rois(tracks{i}(3+j),1)+1:file.rois(tracks{i}(3+j),3),file.rois(tracks{i}(3+j),2)+1:file.rois(tracks{i}(3+j),4),1)=arguments-1;
           entropy(file.rois(tracks{i}(3+j),1)+1:file.rois(tracks{i}(3+j),3),file.rois(tracks{i}(3+j),2)+1:file.rois(tracks{i}(3+j),4),1)=file.entropies{tracks{i}(3+j)};      
           track(:,:,j+1)=mask;
           entropies(:,:,j+1)=entropy;
        
       end
       hdf5write([outDir 'track_' num2str(i) '.h5'], '/track', track);
       hdf5write([outDir 'entropy_' num2str(i) '.h5'], '/entropy', entropies);
   end
end


