#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "Model.h"

Model ModelBase::volume = {	NULL,
							0,
							{0, 0, 0},
							{-1, -1, -1}			
							};

float ModelBase::histogram[256];

void ModelBase::compute_histogram() {
	unsigned int int_histogram[256];
	float max_value = 0;
	for (int i = 0; i<256; i++) 
		int_histogram[i] = 0;
	for (unsigned int i = 0; i<volume.size; i++)
		int_histogram[volume.data[i]]++;
	for (int i = 0; i<256; i++) {
		histogram[i] = sqrt(sqrt((float)int_histogram[i]));
		if (histogram[i] > max_value)
			max_value = histogram[i];
	}
	for (int i = 0; i<256; i++) 
		histogram[i] = histogram[i] / max_value;
}

int ModelBase::load_model(const char* file_name) {
	//free predosly
    const char *dot = strrchr(file_name, '.');
	const char supported_ext[2][10] = {".raw", ".pvm"};
	if (!dot || (strcmp(dot, supported_ext[0]) != 0 && strcmp(dot, supported_ext[1]) != 0)) {
		fprintf(stderr, "File error: Unsupported file extension (.raw|.pvm allowed)\n");
		return 1;
	}
	unsigned char *loaded_data;
	unsigned int width, height, depth, size, components = 1;
	if (strcmp(dot, supported_ext[1]) == 0) {
		printf("Reading PVM file...\n");
		float scale_x, scale_y, scale_z;
		unsigned char *description, *courtesy, *parameters, *comment;
		loaded_data = readPVMvolume(file_name, &width, &height, &depth, 
			&components, &scale_x, &scale_y, &scale_z, &description, &courtesy, &parameters, &comment);
		if (loaded_data == NULL) { 
			fprintf(stderr, "Error: File not found: %s\n", file_name);
			return 1;
		}
		printf("PVM file volume metadata:\nDimensions: width = %d height = %d depth = %d components = %d\n",
			width, height, depth, components);
		if (scale_x!=1.0f || scale_y!=1.0f || scale_z!=1.0f)
			printf("Real dimensions: %g %g %g\n", scale_x, scale_y, scale_z);
		if (description!=NULL)
			printf("Object description:\n%s\n",description);
		if (courtesy!=NULL)
			printf("Courtesy information:\n%s\n",courtesy);
		if (parameters!=NULL)
			printf("Scan parameters:\n%s\n",parameters);
		if (comment!=NULL)
			printf("Additonal comments:\n%s\n",comment);
		printf("\n");
		size = width * height * depth;				
		//float max_scale = MAXIMUM(scale_x, MAXIMUM(scale_y, scale_z));	// scaling - nasledne treba upravit indexovanie v modeli: (pos.x*(1/bound.x)) 
		//volume.bound = (-1) * make_float3(scale_x / max_scale, scale_y / max_scale, scale_z / max_scale);
	}
	if (strcmp(dot, supported_ext[0]) == 0) {
		printf("Reading RAW file...\n");
		loaded_data = readRAWfile(file_name, &size);
		if (loaded_data == NULL) { 
			fprintf(stderr, "Error: File not found: %s\n", file_name);
			return 1;
		}
		printf("Enter RAW file volume dimensions (width, height, depth): ");   
		scanf("%d %d %d",&width, &height, &depth);
		if (width * height * depth != size) {
			printf("Enter RAW file volume components (bytes per voxel): ");   
			scanf("%d", &components);
			if (width * height * depth * components != size) {
				fprintf(stderr, "Error: Incorrect RAW file volume parameters\n");
				free(loaded_data);
				return 1;
			}
		}
	}
	if (components > 2) {
		fprintf(stderr, "Error: Unsupported number of components (1|2 allowed)\n");
		free(loaded_data);
		return 1;
	}
	if (components == 2) {
		printf("Quantizing 16 bit volume to 8 bit using a non-linear mapping...\n");
		loaded_data = quantize(loaded_data, width, height, depth);
	}
	volume.dims = make_ushort3(width, height, depth);
	volume.size = size;
	volume.data = loaded_data;
	printf("File loaded: %s, volume size: %.2f MB\n\n", file_name, volume.size / (float)(1024 * 1024));
	compute_histogram();

	return 0;
}


