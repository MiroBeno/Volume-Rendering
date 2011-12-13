#include <stdio.h>
#include <stdlib.h>
#include "raycaster.h"

static Raycaster raycaster = {	NULL,				// opravit inicializaciu, asi sa neda pri vnorenej strukture
								0,
								{0, 0, 0},
								{-1, -1, -1},
								{1, 1, 1},
								0.06f,
								0.95f,
								0.0f
							};

Raycaster get_raycaster() {
	return raycaster;
}

void set_tf_offset(float offset) {
	raycaster.tf_offset = MINIMUM(MAXIMUM(raycaster.tf_offset+offset, 0), 0.9f);
	printf("Transfer funcion offset: %4.3f\n", raycaster.tf_offset);
}

void set_ray_step(float step) {
	raycaster.ray_step = MINIMUM(MAXIMUM(raycaster.ray_step+step, 0.001f), 1);
	printf("Ray sampling step: %4.4f\n", raycaster.ray_step);
}

void set_ray_threshold(float threshold) {
	raycaster.ray_thershold = MINIMUM(MAXIMUM(raycaster.ray_thershold+threshold, 0.25f), 1);
	printf("Ray accumulation threshold: %4.3f\n", raycaster.ray_thershold);
}

void set_raycaster_model(Volume_model model) {
	raycaster.model = model;
}
