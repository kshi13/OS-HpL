function images = loadMNISTImages(filename)
  
%%Modified version of loadMNISTImages from
%%http://ufldl.stanford.edu/wiki/index.php/Using_the_MNIST_Dataset  

%loadMNISTImages returns a 28*28x[number of MNIST images] matrix containing
%the raw MNIST images

fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open, ', filename,...
                  'please download the MNIST dataset from ',...  
                  'http://yann.lecun.com/exdb/mnist/ and put the ', ...
                  'corresponding files in this folder.']);

magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2051, ['Bad magic number in ', filename, '']);

numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
numRows = fread(fp, 1, 'int32', 0, 'ieee-be');
numCols = fread(fp, 1, 'int32', 0, 'ieee-be');

images = fread(fp, inf, 'unsigned char');
images = reshape(images, numCols, numRows, numImages);
images = permute(images,[2 1 3]);

fclose(fp);

% Reshape to #pixels x #examples
images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));

end

