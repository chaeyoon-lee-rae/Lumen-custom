#include "LumenPCH.h"
#include "SphereScene.h"
#include <random>
#include <algorithm>
#include <cmath>
#include <cstdio>

static constexpr float kPi = 3.14159265358979323846f;

static void generate_plane_mesh(const SphereSceneConfig& cfg,
                                std::vector<glm::vec3>& positions,
                                std::vector<Vertex>&    vertices,
                                std::vector<uint32_t>&  indices) {
	const uint32_t tess   = cfg.plane_tessellation;
	const float    extent = cfg.scene_extent * 2.0f;
	const float    step   = extent / tess;
	const float    half   = extent * 0.5f;

	const uint32_t vtx_count = (tess + 1) * (tess + 1);
	positions.reserve(vtx_count);
	vertices.reserve(vtx_count);
	indices.reserve(tess * tess * 6);

	// Grid at Y=0, normal pointing up; instance transforms place each plane.
	for (uint32_t zi = 0; zi <= tess; ++zi) {
		for (uint32_t xi = 0; xi <= tess; ++xi) {
			float x = -half + xi * step;
			float z = -half + zi * step;
			float u = float(xi) / tess;
			float v = float(zi) / tess;

			glm::vec3 pos{x, 0.0f, z};
			positions.push_back(pos);

			Vertex vtx;
			vtx.pos    = pos;
			vtx.normal = glm::vec3(0.0f, 1.0f, 0.0f);
			vtx.uv0    = glm::vec2(u, v);
			vertices.push_back(vtx);
		}
	}

	// Two triangles per quad, CCW from above (normal up).
	for (uint32_t zi = 0; zi < tess; ++zi) {
		for (uint32_t xi = 0; xi < tess; ++xi) {
			uint32_t tl = zi * (tess + 1) + xi;
			uint32_t tr = tl + 1;
			uint32_t bl = (zi + 1) * (tess + 1) + xi;
			uint32_t br = bl + 1;
			indices.push_back(tl); indices.push_back(bl); indices.push_back(tr);
			indices.push_back(tr); indices.push_back(bl); indices.push_back(br);
		}
	}
}

SphereSceneData SphereSceneData::generate(const SphereSceneConfig& cfg) {
	SphereSceneData data;

	std::mt19937 rng(cfg.seed);
	std::uniform_real_distribution<float> radius_dist(cfg.min_radius, cfg.max_radius);
	std::uniform_real_distribution<float> xz_dist(-cfg.scene_extent, cfg.scene_extent);
	std::uniform_real_distribution<float> color_dist(0.01f, 1.0f);
	std::uniform_real_distribution<float> intensity_dist(cfg.min_light_intensity, cfg.max_light_intensity);
	std::uniform_real_distribution<float> roughness_dist(0.0f, 1.0f);
	std::uniform_real_distribution<float> metallic_dist(0.0f, 1.0f);

	// ── Sphere placement with collision detection ─────────────────────────
	data.spheres.reserve(cfg.total_sphere_count);
	const uint32_t max_attempts = cfg.total_sphere_count * 100;
	for (uint32_t att = 0;
	     data.spheres.size() < cfg.total_sphere_count && att < max_attempts;
	     ++att) {
		float r = radius_dist(rng);
		float x = xz_dist(rng);
		float z = xz_dist(rng);
		float min_y = cfg.lower_plane_y + r + cfg.plane_margin;
		float max_y = cfg.upper_plane_y - r - cfg.plane_margin;
		if (min_y >= max_y) continue;

		std::uniform_real_distribution<float> y_dist(min_y, max_y);
		float y = y_dist(rng);

		bool ok = true;
		for (const auto& s : data.spheres) {
			float dx = x - s.center.x, dy = y - s.center.y, dz = z - s.center.z;
			float min_d = r + s.radius;
			if (dx * dx + dy * dy + dz * dz < min_d * min_d) { ok = false; break; }
		}
		if (ok) data.spheres.push_back({glm::vec3(x, y, z), r});
	}
	printf("SphereScene: generated %zu spheres (requested %u)\n",
	       data.spheres.size(), cfg.total_sphere_count);

	// ── Select emissive spheres ───────────────────────────────────────────
	const uint32_t N            = (uint32_t)data.spheres.size();
	const uint32_t num_emissive = std::min(cfg.light_sphere_count, N);

	std::vector<uint32_t> order(N);
	for (uint32_t i = 0; i < N; ++i) order[i] = i;
	std::shuffle(order.begin(), order.end(), rng);

	std::vector<bool> is_emissive(N, false);
	for (uint32_t i = 0; i < num_emissive; ++i) is_emissive[order[i]] = true;

	// ── Materials: sphere_mat[0..N-1], lower_plane, upper_plane ──────────
	data.materials.reserve(N + 2);
	for (uint32_t i = 0; i < N; ++i) {
		Material mat{};
		mat.bsdf_type  = BSDF_TYPE_DIFFUSE;
		mat.bsdf_props = BSDF_FLAG_DIFFUSE | BSDF_FLAG_REFLECTION;
		mat.texture_id = -1;
		mat.ior        = 1.5f;

		if (is_emissive[i]) {
			float surf  = 4.0f * kPi * data.spheres[i].radius * data.spheres[i].radius;
			float power = intensity_dist(rng);
			float rad   = power / surf;
			float r = color_dist(rng), g = color_dist(rng), b = color_dist(rng);
			float mag = std::sqrt(r * r + g * g + b * b);
			mat.emissive_factor = glm::vec3(r, g, b) * (rad / mag);
			mat.albedo          = glm::vec3(0.0f);
		} else {
			// BIM-matched BRDF: principled GGX with per-material roughness/metallic
			// Ref: BIM scene_spheres.cpp:126-127 (random), generateSamples.rgen:121 (evalBRDF)
			mat.bsdf_type       = BSDF_TYPE_PRINCIPLED;
			mat.bsdf_props      = BSDF_FLAG_DIFFUSE | BSDF_FLAG_REFLECTION;
			mat.albedo          = glm::vec3(color_dist(rng), color_dist(rng), color_dist(rng));
			mat.emissive_factor = glm::vec3(0.0f);
			//mat.roughness       = roughness_dist(rng);
			mat.roughness		= 1.0f;
			//mat.metallic        = metallic_dist(rng);
			mat.metallic		= 0.0f;
			mat.spec_trans      = 0.0f;
			mat.clearcoat       = 0.0f;
		}
		data.materials.push_back(mat);
	}

	// BIM-matched plane materials: principled GGX, roughness=1, metallic=0
	{
		Material pm{};
		pm.bsdf_type  = BSDF_TYPE_PRINCIPLED;
		pm.bsdf_props = BSDF_FLAG_DIFFUSE | BSDF_FLAG_REFLECTION;
		pm.texture_id = -1;
		pm.ior        = 1.5f;
		pm.albedo     = glm::vec3(0.8f);
		pm.roughness  = 1.0f;
		pm.metallic   = 0.0f;
		pm.spec_trans = 0.0f;
		pm.clearcoat  = 0.0f;
		data.materials.push_back(pm);   // index N   = lower plane
		data.materials.push_back(pm);   // index N+1 = upper plane
	}

	// ── GPU light structs ─────────────────────────────────────────────────
	data.gpu_lights.reserve(num_emissive);
	for (uint32_t i = 0; i < num_emissive; ++i) {
		uint32_t si = order[i];
		const SpherePrimitive& s = data.spheres[si];
		const Material&        m = data.materials[si];
		float surf = 4.0f * kPi * s.radius * s.radius;

		Light l{};
		l.world_matrix  = glm::mat4(1.0f);
		l.pos           = s.center;
		l.prim_mesh_idx = si;           // used for MIS triangle_idx matching
		l.num_triangles = 1;            // one sample point per sphere
		l.L             = m.emissive_factor;
		l.light_flags   = LIGHT_SPHERE | (1 << 4);  // sphere + finite
		l.world_radius  = s.radius;                 // repurposed: sphere radius
		l.world_center  = s.center;

		data.gpu_lights.push_back(l);
		data.total_light_area += surf;
	}
	data.total_light_triangle_cnt = num_emissive;

	printf("SphereScene: %u emissive, %u diffuse\n",
	       num_emissive, N - num_emissive);

	// ── AABBs ─────────────────────────────────────────────────────────────
	data.aabbs.reserve(N);
	for (const auto& s : data.spheres) {
		VkAabbPositionsKHR ab{
			s.center.x - s.radius, s.center.y - s.radius, s.center.z - s.radius,
			s.center.x + s.radius, s.center.y + s.radius, s.center.z + s.radius
		};
		data.aabbs.push_back(ab);
	}

	// ── Plane mesh ────────────────────────────────────────────────────────
	generate_plane_mesh(cfg, data.plane_positions, data.plane_vertices, data.plane_indices);
	printf("SphereScene: plane mesh %zu verts, %zu tris\n",
	       data.plane_vertices.size(), data.plane_indices.size() / 3);

	return data;
}
