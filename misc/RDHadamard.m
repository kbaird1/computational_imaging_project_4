clear all;
close all;
clc;
% Russian-Doll Hadamard
%function H = RDHadamard(N)
N =4096;
n = log2(N);
% Rule 1
H0 = [1,1;1,-1];
H_prev = H0;
for i = 1:n-1
    clc;
    i
    H_upgrade = kron(H_prev,H0);
    H_temp = zeros(size(H_prev,1),size(H_prev,2)*2);
    
    if (rem(i,2) == 0)
    for j = 1:size(H_prev,2)
        H_temp(:,1+2*(j-1):2*j) = repmat(H_prev(:,j),1,2);
    end
    else

        for j = 1:size(H_prev,1)
            H_prev_temp = reshape(H_prev(j,:),2*sqrt(2^(i-1)),sqrt(2^(i-1)));
            for kk = 1:sqrt(2^(i-1))
                if kk == 1
                    H_accum_temp = repmat(H_prev_temp(:,kk),1,2);
                else
                    H_accum_temp = [H_accum_temp,repmat(H_prev_temp(:,kk),1,2)];
                end
            end
            H_temp(j,:) = H_accum_temp(:);
        end
    end
    for j = 1:size(H_temp,1)
        H_temp1 = H_temp(j,:);
        for k = 1:size(H_upgrade,1)
            if H_upgrade(k,:) == H_temp1
                H_upgrade_temp = H_upgrade(j,:);
                H_upgrade(j,:) = H_upgrade(k,:);
                H_upgrade(k,:) = H_upgrade_temp;
            end
        end
    end
    
    
if (rem(i,2) == 0)
    H_prev = H_upgrade;
    H = H_upgrade;
else
    H = H_upgrade;

%     % Rule 3
% H_block = zeros(sqrt(size(H,1)),sqrt(size(H,2)),size(H,1));
% 
% for ii = 1:size(H,1)
%     H_temp = H(ii,:);
%     H_block(:,:,ii) = reshape(H_temp(:),sqrt(size(H,1)),sqrt(size(H,2)));
%     [gx gy] = gradient(H_block(:,:,ii));
%     H_edge(ii) = sum(sum(abs(gx.^2 + gy.^2)));
% end
% 
% [H_edge_up,idx] = sort(H_edge);
% H = H(idx,:);

% Rule 2
H_2quar = H(size(H,1)/4+1:size(H,1)/2,:);
H_transpose = zeros(size(H_2quar));
for ii = 1:size(H_2quar,1)
    H_temp = H_2quar(ii,:);
    H_temp = H_temp(:);
    H_temp_transpose = (reshape(H_temp,sqrt(size(H_2quar,2)),sqrt(size(H_2quar,2))))';
    H_transpose(ii,:) = H_temp_transpose(:);
end

for j = 1:size(H_transpose,1)
        H_temp1 = H_transpose(j,:);
        for k = 1:size(H,1)
            if H(k,:) == H_temp1
                H_upgrade_temp = H(size(H,1)/2+j,:);
                H(size(H,1)/2+j,:) = H(k,:);
                H(k,:) = H_upgrade_temp;
            end
        end
end

% Rule 3
H_block = zeros(sqrt(size(H,1)),sqrt(size(H,2)),size(H,1));

for iii = 1:size(H,1)
    H_temp = H(iii,:);
    H_block(:,:,iii) = reshape(H_temp(:),sqrt(size(H,1)),sqrt(size(H,2)));
    [gx,n_gx] = bwlabel((H_block(:,:,iii)+1)/2,4);
    [gy,n_gy] = bwlabel((-H_block(:,:,iii)+1)/2,4);
    H_edge(iii) = n_gx + n_gy;
end

% [H_edge_up,idx] = sort(H_edge);
% H = H(idx,:);

for iiii = 1:4
    H_edge_temp = H_edge(1+size(H,1)/4*(iiii-1):size(H,1)/4*iiii);
    [H_edge_temp_up,idx] = sort(H_edge_temp);
    H_temp = H(1+size(H,1)/4*(iiii-1):size(H,1)/4*iiii,:);
    H(1+size(H,1)/4*(iiii-1):size(H,1)/4*iiii,:) = H_temp(idx,:);
end
H_prev = H;
end
end

% if (rem(n,2) == 0)
% for kkk = 1:size(H,1)
%     figure;imagesc(reshape(H(kkk,:),sqrt(size(H,2)),sqrt(size(H,2))));
% end
% else
%     for kkk = 1:size(H,1)
%     figure;imagesc(reshape(H(kkk,:),2*sqrt(2^i),sqrt(2^i)));
%     end
% end

% for kkk = 1:size(H_next,1)
%     figure;imagesc(reshape(H_next(kkk,:),sqrt(size(H_next,2)),sqrt(size(H_next,2))));
% end
    
