#include <stdio.h>
#include "Raycaster.h"

Raycaster RaycasterBase::raycaster = {	
							Model(),
							View(),
							0.06f,
							0.95f,
							0.0f
						};

float4 RaycasterBase::transfer_fn[256];

void RaycasterBase::change_tf_offset(float offset, bool reset) {
	raycaster.tf_offset = CLAMP(reset ? offset : raycaster.tf_offset + offset, 0, 0.9f);
	printf("Transfer funcion offset: %.3f\n", raycaster.tf_offset);
}

void RaycasterBase::change_ray_step(float step, bool reset) {
	raycaster.ray_step = CLAMP(reset ? step : raycaster.ray_step + step, 0.001f, 1);
	printf("Ray sampling step: %.4f (approx %.1f sampling points)\n", raycaster.ray_step, 2/raycaster.ray_step);
}

void RaycasterBase::change_ray_threshold(float threshold, bool reset) {
	raycaster.ray_threshold = CLAMP(reset ? threshold : raycaster.ray_threshold + threshold, 0.25f, 1);
	printf("Ray accumulation threshold: %.3f\n", raycaster.ray_threshold);
}

void RaycasterBase::set_volume(Model model) {
	raycaster.model = model;
	int max_size = MAXIMUM(model.dims.x, MAXIMUM(model.dims.y, model.dims.z));
	raycaster.ray_step = 2.0f / max_size;  // dlzka najvacsej hrany je 2 
	raycaster.ray_step -= raycaster.ray_step / max_size;
}
