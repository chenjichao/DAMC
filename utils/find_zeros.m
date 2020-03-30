% close all
% clear
% clc
% % restoredefaultpath
function find_zeros(filename)


    addpath(genpath('logs'));

%     filename = 'DAMC_wDfE-MSRCv1-20200330_135600';
    filename = sprintf('%s.log',filename)
    fileID = fopen(filename,'r');
    % fileID = fopen('ELMDFAN_fixed-MSRCv1-20200322_140952.log','r');
    formatSpec = '%c';
    A = fscanf(fileID,formatSpec);


    % A = split(A,'Done experiment');
    A = split(A,'Done');

    % count = 0;
    % for i = 1:length(A)
    %     if startsWith(A{i},' experiment')
    %         count = count+1;
    %     end
    % end
    % TF = startsWith(str,pattern)



    paraname = {};
    paravalue = {};
    nPrm = 0;
    newStr = splitlines(A{1});
    for i = 1:length(newStr)
    if ~isempty(strfind(newStr{i},': ['))
        nPrm = nPrm+1;
        paraset = strtrim(split(newStr{i},":"));
        paraname{nPrm} = paraset{1};
        temp = split(paraset{2},["[","]"]);
        paravalue{nPrm} = split(temp{2})';
    end
    end

    longvalue = 0;
    for iPrm = 1:nPrm
    temp = length(paravalue{iPrm});
    if temp > longvalue
        longvalue = temp;
    end
    end
    results = zeros(length(paraname),longvalue,3);

    % A = split(A,'Done experiment');

    for iPrm = 1:nPrm
    %     fprintf('paraname: %s\n# of paravalue: %d\n',...
    %         paraname{iPrm},length(paravalue{iPrm}))
    for iVal = 1:length(paravalue{iPrm})
        pattern = sprintf('%s: %s',paraname{iPrm},paravalue{iPrm}{iVal});

        for i = 1:length(A)
            if startsWith(A{i},' experiment')
                if ~isempty(strfind(A{i},pattern))
                    results(iPrm,iVal,2) = results(iPrm,iVal,2)+1;
                    if ~isempty(strfind(A{i},'accuracy: 0.0000'))
                        results(iPrm,iVal,1) = results(iPrm,iVal,1)+1;
                    end
                end
            end
        end
    end
    end




    % parse results
    highest_fail_rate = 0;
    highest_fail_para = '';
    for iPrm = 1:nPrm
    fprintf('\n%s: ',paraname{iPrm});
    for iVal = 1:length(paravalue{iPrm})
        fprintf('%s ',paravalue{iPrm}{iVal});
        if results(iPrm,iVal,2)~=0
            fail_rate = results(iPrm,iVal,1)/results(iPrm,iVal,2);
            results(iPrm,iVal,3) = fail_rate;
            current_para = sprintf('%s: %s\n',paraname{iPrm},paravalue{iPrm}{iVal});
    %             fprintf('%s: %s\nfail_rate:%0.2f\n',paraname{iPrm},paravalue{iPrm}{iVal},fail_rate);
            if fail_rate > highest_fail_rate
                highest_fail_rate = fail_rate;
                highest_fail_para = current_para;
            end
        end
    end
    end
    highest_fail_para
    highest_fail_rate

end
