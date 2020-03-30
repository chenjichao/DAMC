close all
clear
clc
restoredefaultpath

% set the parameter to parse
parameter_name = 'nOut';

% read the text file
addpath(genpath('logs'));
% fileID = fopen('ELMDFAN_fixed-HD-20200322_162240.log','r');
fileID = fopen('ELMDFAN_fixed-MSRCv1-20200322_140952.log','r');


formatSpec = '%c';
A = fscanf(fileID,formatSpec);
fclose(fileID);

% extract options of defined parameter
newStr = split(A,'Done experiment');
newStr2 = newStr{1};
candidates = split(extractBetween(newStr{1},...
    sprintf('%s: [',parameter_name),']'));

% qia tou qu wei
newStr = newStr(2:end);
temp1 = split(newStr{end},'/');
temp2 = split(newStr{1},'/');
temp3 = split(newStr{1},'/');
fprintf('Completed (%s~%s)/%s experiments.\n',temp1{1},temp2{1},temp3{1});

if ~isempty(strfind(newStr{end},'Error'))
    temp = extractBetween(newStr(end),'in','Error');
    newStr{end} = temp{1};
end

% assume there are 7 performance metrics and time, init result matrix.
nMtr = 8;
nCdd = length(candidates);
best_results = zeros(nCdd,nMtr);

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
            lines = split(newStr{iExp},',');
            lines = lines(1:nMtr);
            
            
            rtime = split(lines{1},' ');
            rtime = rtime{end};
            rtime = str2num(rtime(1:end-1));
                
            temp = split(lines(2:end),': ');
            temp = str2num(cell2mat(temp(:,2)));
            currentresults = [temp',rtime];
            
            
%             candidates = split(extractBetween(newStr{1},...
%     sprintf('%s: [',parameter_name),']'));
            
%             full_results = extractBetween(lines,':',')');
%             mean_results = extractBetween(lines,' ',' ');
%             currentresults =  str2num(cell2mat(mean_results))';
            best_results(iCdd,:) = max(...
                [currentresults;best_results(iCdd,:)],[],1);
        end  
    end
end


% print
for iCdd = 1:nCdd
    fprintf('%s: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %g\n',...
        candidates{iCdd},best_results(iCdd,:)); 
end
% find the best of each metric
[M,I] = max(best_results,[],1);
fprintf('\nbest: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %g\n',M);
fprintf('best: %s %s %s %s %s %s %s\n',...
    candidates{I(1)},candidates{I(2)},candidates{I(3)},candidates{I(4)},...
    candidates{I(5)},candidates{I(6)},candidates{I(7)});

% plot
% find the range of the results
minmin = min(min(best_results(:,1:end-1)));
maxmax = max(max(best_results(:,1:end-1)));

% as bar
subplot(1,2,1)
bar3(best_results(:,1:end-1))
xlabel('performances')
ylabel('options')
zlabel('zzlabel')
xticklabels({'ACC','NMI','PUR','PRE','REC','FSC','ARI'})
yticklabels(candidates)
zlim([minmin maxmax])

% as surface
subplot(1,2,2)
surf(best_results(:,1:end-1))
xlabel('performances')
ylabel('options')
zlabel('zzlabel')
xticklabels({'ACC','NMI','PUR','PRE','REC','FSC','ARI'})
yticklabels(candidates)
zlim([minmin maxmax])
