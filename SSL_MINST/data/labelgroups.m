function [lgroups, labels] = labelgroups(filename)
  lgroups = zeros(10,1000);
  fp = fopen(filename, 'rb');
  assert(fp ~= -1, ['Could not open ', filename, '']);

  magic = fread(fp, 1, 'int32', 0, 'ieee-be');
  assert(magic == 2049, ['Bad magic number in ', filename, '']);

  numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
  labels = fread(fp, inf, 'unsigned char');
  fclose(fp);
  
  k = [1,1,1,1,1,1,1,1,1,1];
  
  for i = 1:10000
    lab = labels(i);
    lgroups(lab+1, k(lab+1)) = i;
    k(lab+1) = k(lab+1)+1;
  end
  
end
