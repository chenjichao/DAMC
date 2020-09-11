% close all
% clear
% clc
% restoredefaultpath
function parselog(filename,parameter)

    % set the parameter to parse
%     parameter_name = 'nOut';
    parameter_name = parameter;

    % read the text file
    addpath(genpath('logs'));
%     fileID = fopen('ELMDFAN_tuned-MSRCv1-20200328_184637.log','r');
    filename = sprintf('%s.log',filename);
    fileID = fopen(filename,'r');
    formatSpec = '%c';
    A = fscanf(fileID,formatSpec);
    fclose(fileID);

    % extract options of defined parameter
    newStr = split(A,'Parameters');
    candidates = split(extractBetween(newStr(2),...
        sprintf('%s: [',parameter_name),']'));

    % qia tou qu wei
    newStr = newStr(3:end);
    newStr(end) = extractBetween(newStr(end),'/','Highest');

    % assume there are 7 performance metrics, init result matrix.
    nMtr = 7;
    nCdd = length(candidates);
    best_results = zeros(nCdd,nMtr);
    best_std = zeros(nCdd,nMtr);

    nExp = length(newStr);
    for iExp = 1:nExp
    % for iExp = 1:1
        for iCdd = 1:nCdd
    %         newStr{iExp}
            k = strfind(newStr{iExp},...
                sprintf('%s: %s',parameter_name,candidates{iCdd}));

            if length(k) > 1
                disp('some error may occured')
            elseif ~isempty(k)
                lines = splitlines(newStr{iExp});
                lines = lines(2:nMtr+1);
                lines = split(lines);
                
                
%                 currentresults = str2num(cell2mat(lines(:,2)))';
                currentresults = cellfun(@str2num, lines(:,2))';
%                 currentstd = str2num(cell2mat(lines(:,3)))';
                currentstd = cellfun(@str2num, lines(:,3))';

    %             candidates = split(extractBetween(newStr{1},...
    %     sprintf('%s: [',parameter_name),']'));
    %             full_results = extractBetween(lines,':',')');
    %             mean_results = extractBetween(lines,' ',' ');
    %             currentresults =  str2num(cell2mat(mean_results))';
                [best_results(iCdd,:),I] = max(...
                    [currentresults;best_results(iCdd,:)],[],1);
                
                bothstd=[currentstd;best_std(iCdd,:)];
                best_std(iCdd,:) = bothstd(sub2ind(size(bothstd),I,1:length(I)));
                
            end  
        end
    end

    best_results = best_results*100;
    best_std = best_std*100;
    
    % print
    for iCdd = 1:nCdd
        fprintf('%s:\n    %.2f %.2f %.2f %.2f %.2f %.2f %.2f \n    %.2f %.2f %.2f %.2f %.2f %.2f %.2f \n',...
            candidates{iCdd},best_results(iCdd,:),best_std(iCdd,:)); 
        temp = [best_results(iCdd,:);best_std(iCdd,:)]'
    end
    % find the best of each metric
    [M,I] = max(best_results,[],1);
    fprintf('\nbest: %.2f %.2f %.2f %.2f %.2f %.2f %.2f \n',M);
    fprintf('best: %s %s %s %s %s %s %s',...
        candidates{I(1)},candidates{I(2)},candidates{I(3)},candidates{I(4)},...
        candidates{I(5)},candidates{I(6)},candidates{I(7)});
    % find the worst of each metric
    [M,I] = min(best_results,[],1);
    fprintf('\nworst: %.2f %.2f %.2f %.2f %.2f %.2f %.2f \n',M);
    fprintf('worst: %s %s %s %s %s %s %s\n',...
        candidates{I(1)},candidates{I(2)},candidates{I(3)},candidates{I(4)},...
        candidates{I(5)},candidates{I(6)},candidates{I(7)});


    % plot
    % find the range of the results
    minmin = min(min(best_results));
    maxmax = max(max(best_results));

    % as bar
%     subplot(1,2,1)
    bar3(best_results)
    xlabel('Performances')
    ylabel('log_{10}\delta')
%     ylabel('Number of Outputs')
%     zlabel('zzlabel')
    xticklabels({'ACC','NMI','PUR','PRE','REC','FSC','ARI'})
    yticklabels(candidates)
    zlim([0 100])
%     zlim([minmin maxmax])

    title('MSRCv1');

    % as surface
%     subplot(1,2,2)
%     surf(best_results)
%     zlim([minmin maxmax])
end