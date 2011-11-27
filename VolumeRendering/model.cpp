#include <stdio.h>
#include <stdlib.h>
#include "model.h"

static Volume_model volume = {	NULL,
								0,
								{0, 0, 0},
								{-1, -1, -1},
								{1, 1, 1},
								0.06f					// upravit podla poctu voxelov 
							};

int load_file(const char *file_name, unsigned char **result, size_t *file_size) {
	*file_size = 0;
	FILE *f = fopen(file_name, "rb");
	if (f == NULL) 
	{ 
		*result = NULL;
		return -1;				// chyba otvarania
	} 
	fseek(f, 0, SEEK_END);
	*file_size = (size_t) ftell(f);
	fseek(f, 0, SEEK_SET);
	*result = (unsigned char *)malloc(*file_size);
	if (*file_size != fread(*result, sizeof(unsigned char), *file_size, f)) 
	{ 
		free(*result);
		return -2;				// chyba citania
	} 
	fclose(f);
	return 0;
}

int load_model(const char* file_name) {
	int result = load_file(file_name, &volume.data, &volume.size);
	if (volume.size == 0 || result != 0) 
		return -1;
	printf("File loaded: %s. Size: %u B.\n", file_name, volume.size);
	volume.dims = make_cudaExtent(32, 32, 32);													// nacita sa z hlavicky, zatial explicitne
	int max_size = MAXIMUM(volume.dims.width, MAXIMUM(volume.dims.height, volume.dims.depth));	// dlzka najvacsej hrany je 2 a stred kvadra v [0,0,0]
	volume.max_bound = make_float3(volume.dims.width / (float) max_size, volume.dims.height / (float) max_size, volume.dims.depth / (float) max_size);
	volume.min_bound = -volume.max_bound;
	return 0;
}

Volume_model get_model() {
	return volume;
}


