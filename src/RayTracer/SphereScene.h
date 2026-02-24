#pragma once
#include "shaders/commons.h"
#include <vulkan/vulkan.h>
#include <vector>
#include <cstdint>

struct SphereSceneConfig {
	uint32_t total_sphere_count  = 50000;
	uint32_t light_sphere_count  = 300;
	float    min_light_intensity = 2000.0f;
	float    max_light_intensity = 2000.0f;
	float    min_radius          = 2.0f;
	float    max_radius          = 3.0f;
	float    lower_plane_y       = -10.0f;
	float    upper_plane_y       = 50.0f;
	float    scene_extent        = 2000.0f;
	float    plane_margin        = 1.0f;
	uint32_t plane_tessellation  = 10;
	uint32_t seed                = 42;
};

// All geometry and light data for the sphere scene, in Lumen-native formats.
// generate() is the single entry point; no GPU resources are created here.
struct SphereSceneData {
	std::vector<SpherePrimitive>    spheres;
	std::vector<VkAabbPositionsKHR> aabbs;

	// sphere_mats[0..N-1] then plane_mat[0] (lower) then plane_mat[1] (upper)
	std::vector<Material> materials;

	std::vector<Light>    gpu_lights;
	uint32_t              total_light_triangle_cnt = 0;
	float                 total_light_area         = 0.0f;

	// Both plane instances share this single mesh (different instance transforms).
	std::vector<glm::vec3> plane_positions;   // positions-only → vertex_buffer (BLAS build)
	std::vector<Vertex>    plane_vertices;    // full Vertex    → compact_vertices_buffer
	std::vector<uint32_t>  plane_indices;

	static SphereSceneData generate(const SphereSceneConfig& config);
};
