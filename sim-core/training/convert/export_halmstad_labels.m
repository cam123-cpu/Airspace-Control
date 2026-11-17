% Export the Halmstad drone dataset's MATLAB groundTruth labels to labels.csv
% (video,frame,class,x,y,w,h; frame 1-based, pixels) for training/convert/halmstad_csv.py.
% Usage in MATLAB:  export_halmstad_labels('/data/halmstad')
function export_halmstad_labels(root)
    files = dir(fullfile(root, '**', '*_LABELS.mat'));
    fid = fopen(fullfile(root, 'labels.csv'), 'w');
    fprintf(fid, 'video,frame,class,x,y,w,h\n');
    for k = 1:numel(files)
        S = load(fullfile(files(k).folder, files(k).name));
        fn = fieldnames(S); gt = S.(fn{1});
        video = erase(files(k).name, '_LABELS.mat');
        T = gt.LabelData;                 % timetable: one column per label class
        classes = T.Properties.VariableNames;
        for f = 1:height(T)
            wrote = false;
            for c = 1:numel(classes)
                boxes = T{f, c};
                if iscell(boxes), boxes = boxes{1}; end
                for b = 1:size(boxes, 1)
                    fprintf(fid, '%s,%d,%s,%g,%g,%g,%g\n', video, f, lower(classes{c}), boxes(b, :));
                    wrote = true;
                end
            end
            if ~wrote, fprintf(fid, '%s,%d,,0,0,0,0\n', video, f); end
        end
    end
    fclose(fid);
end
