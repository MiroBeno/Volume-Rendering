#include <stdio.h>
#include <stdlib.h>
#include "model.h"

static Volume_model volume = {	NULL,
								0,
								{0, 0, 0},
								{-1, -1, -1},
								{1, 1, 1}			
							};



int load_model(const char* file_name) {
	volume.data = readRAWfile(file_name, &volume.size);
	if (volume.data == NULL) 
		return 1;
	printf("File loaded: %s. Size: %u B.\n", file_name, volume.size);
	// skontrolovat dims -> inak zla alokacia
//	volume.dims = make_int3(32, 32, 32);
//	volume.dims = make_int3(256, 256, 256);
	volume.dims = make_int3(128, 256, 256);
//	volume.dims = make_int3(512, 499, 512);
	int max_size = MAXIMUM(volume.dims.x, MAXIMUM(volume.dims.y, volume.dims.z));	// dlzka najvacsej hrany je 2 a stred kvadra v [0,0,0]
//	volume.max_bound = make_float3(volume.dims.x / (float) max_size, volume.dims.y / (float) max_size, volume.dims.z / (float) max_size);
//	volume.min_bound = -volume.max_bound;
	return 0;
}

Volume_model get_model() {
	return volume;
}


