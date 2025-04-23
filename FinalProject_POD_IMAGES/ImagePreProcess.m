% recortar_grid_imagen.m
% Divide una imagen en 25 subimágenes (5x5) y las guarda por separado

% Cargar imagen
image_path = 'Grid_6x6.png'; % Ruta de la imagen
img = imread(image_path);
if size(img,3) == 3
    img = rgb2gray(img);  % Convertir a escala de grises si es necesario
end
img = im2double(img);  % Convertir a tipo double para procesamiento

% Definir tamaño del grid
grid_size = 6;
[full_height, full_width] = size(img);
cell_height = floor(full_height / grid_size);
cell_width = floor(full_width / grid_size);

% Crear carpeta de salida
output_dir = 'dataset_faces_SET2';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Recortar y guardar cada subimagen
counter = 1;
for row = 0:grid_size-1
    for col = 0:grid_size-1
        y1 = row * cell_height + 1;
        y2 = y1 + cell_height - 1;
        x1 = col * cell_width + 1;
        x2 = x1 + cell_width - 1;
        
        subimg = img(y1:y2, x1:x2);
        filename = fullfile(output_dir, sprintf('face_%d.png', counter));
        imwrite(subimg, filename);
        fprintf('Guardado: %s\n', filename);
        counter = counter + 1;
    end
end
