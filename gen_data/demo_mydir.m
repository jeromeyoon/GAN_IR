function [names,onlynames] = demo_mydir(path)

    [folder, name, ext] = fileparts(path);
    if isempty(folder)
        folder = '.';
    end;
    names = dir(path);
    for i = 1: length(names),
        onlynames{i} = names(i).name ;
        names(i).name = sprintf('%s/%s', folder,names(i).name);
    end
    

end
