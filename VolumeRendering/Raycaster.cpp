#include <stdio.h>
#include <stdlib.h>
#include "raycaster.h"

static Raycaster raycaster = {	Volume_model(),
								View(),
								0.06f,
								0.95f,
								0.0f,
								{0.5f, 0.5f, 0.5f}
							};

float4 transfer_fn_lol[256];

Raycaster *get_raycaster() {
	return &raycaster;
}

void change_tf_offset(float offset, bool reset) {
	raycaster.tf_offset = CLAMP(reset ? offset : raycaster.tf_offset + offset, 0, 0.9f);
	printf("Transfer funcion offset: %4.3f\n", raycaster.tf_offset);
}

void change_ray_step(float step, bool reset) {
	raycaster.ray_step = CLAMP(reset ? step : raycaster.ray_step + step, 0.001f, 1);
	printf("Ray sampling step: %4.4f\n", raycaster.ray_step);
}

void change_ray_threshold(float threshold, bool reset) {
	raycaster.ray_threshold = CLAMP(reset ? threshold : raycaster.ray_threshold + threshold, 0.25f, 1);
	printf("Ray accumulation threshold: %4.3f\n", raycaster.ray_threshold);
}

void set_raycaster_model(Volume_model model) {
	raycaster.model = model;
	int max_size = MAXIMUM(model.dims.x, MAXIMUM(model.dims.y, model.dims.z));
	raycaster.ray_step = 2.0f / max_size;  // dlzka najvacsej hrany je 2 
	raycaster.ray_step -= raycaster.ray_step / max_size;
	/**/
	for (int i =0; i < 256; i++) {
		transfer_fn_lol[i] = make_float4(i <= 85 ? (i*3)/255.0f : 0.0f, 
										(i > 85) && (i <= 170) ? ((i-85)*3)/255.0f : 0.0f, 
										i > 170 ? ((i-170)*3)/255.0f : 0.0f, 
										i/255.0f);
	}
}

void set_raycaster_view(View view) {
	raycaster.view = view;
}
