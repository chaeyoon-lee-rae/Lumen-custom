#include "Framework/RenderGraph.h"
#include "LumenPCH.h"
#include "ReSTIRDISphere.h"
#include "Framework/VkUtils.h"
#include "Framework/AccelerationStructure.h"
#include "Framework/PersistentResourceManager.h"

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static const char* rgen_paths[3] = {
	"src/shaders/integrators/restir/di/temporal_pass.rgen",
	"src/shaders/integrators/restir/di/spatial_pass.rgen",
	"src/shaders/integrators/restir/di/output.rgen",
};

// Shader list shared across all passes (only rgen differs).
// Hit-group ordering (after rgen + 2 miss shaders):
//   sphere.rchit + sphere.rint  → procedural HG 0  (sphere primary)
//   sphere.rchit + sphere.rint  → procedural HG 1  (sphere shadow)
//   ray.rchit   + ray.rahit     → triangle  HG 2  (plane  primary)
//   ray.rchit                   → triangle  HG 3  (plane  shadow, TerminateOnFirstHit)
static std::vector<vk::Shader> make_sphere_shaders(const std::string& rgen) {
	return {
		vk::Shader(rgen),
		vk::Shader("src/shaders/ray.rmiss"),
		vk::Shader("src/shaders/ray_shadow.rmiss"),
		vk::Shader("src/shaders/sphere.rchit"),
		vk::Shader("src/shaders/sphere.rint"),
		vk::Shader("src/shaders/sphere.rchit"),
		vk::Shader("src/shaders/sphere.rint"),
		vk::Shader("src/shaders/ray.rchit"),
		vk::Shader("src/shaders/ray.rahit"),
		vk::Shader("src/shaders/ray.rchit"),
	};
}

// ─────────────────────────────────────────────────────────────────────────────
// init
// ─────────────────────────────────────────────────────────────────────────────

void ReSTIRDISphere::init() {
	// ── 1. Generate procedural scene ──────────────────────────────────────
	sphere_data = SphereSceneData::generate(sphere_config);
	const uint32_t N = (uint32_t)sphere_data.spheres.size();

	// ── 2. Replace lumen_scene geometry buffers ───────────────────────────
	prm::remove(lumen_scene->vertex_buffer);
	prm::remove(lumen_scene->index_buffer);
	prm::remove(lumen_scene->compact_vertices_buffer);
	prm::remove(lumen_scene->materials_buffer);
	prm::remove(lumen_scene->prim_lookup_buffer);

	lumen_scene->vertex_buffer = prm::get_buffer({
		.name        = "Sphere Scene Vertex (positions)",
		.usage       = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
		               VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
		               VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
		.memory_type = vk::BufferType::GPU,
		.size        = sphere_data.plane_positions.size() * sizeof(glm::vec3),
		.data        = sphere_data.plane_positions.data(),
	});

	lumen_scene->index_buffer = prm::get_buffer({
		.name        = "Sphere Scene Index",
		.usage       = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
		               VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
		               VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
		.memory_type = vk::BufferType::GPU,
		.size        = sphere_data.plane_indices.size() * sizeof(uint32_t),
		.data        = sphere_data.plane_indices.data(),
	});

	lumen_scene->compact_vertices_buffer = prm::get_buffer({
		.name        = "Sphere Scene Compact Vertices",
		.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
		.memory_type = vk::BufferType::GPU,
		.size        = sphere_data.plane_vertices.size() * sizeof(Vertex),
		.data        = sphere_data.plane_vertices.data(),
	});

	lumen_scene->materials_buffer = prm::get_buffer({
		.name        = "Sphere Scene Materials",
		.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
		.memory_type = vk::BufferType::GPU,
		.size        = sphere_data.materials.size() * sizeof(Material),
		.data        = sphere_data.materials.data(),
	});

	// prim_lookup: two entries (one per plane instance), indexed by instanceCustomIndex.
	std::vector<PrimMeshInfo> prim_lookup(2);
	prim_lookup[0].index_offset   = 0;
	prim_lookup[0].vertex_offset  = 0;
	prim_lookup[0].material_index = N;
	prim_lookup[0].min_pos = glm::vec4(-sphere_config.scene_extent, sphere_config.lower_plane_y,
	                                    -sphere_config.scene_extent, 0.0f);
	prim_lookup[0].max_pos = glm::vec4( sphere_config.scene_extent, sphere_config.lower_plane_y,
	                                     sphere_config.scene_extent, 0.0f);

	prim_lookup[1].index_offset   = 0;
	prim_lookup[1].vertex_offset  = 0;
	prim_lookup[1].material_index = N + 1;
	prim_lookup[1].min_pos = glm::vec4(-sphere_config.scene_extent, sphere_config.upper_plane_y,
	                                    -sphere_config.scene_extent, 0.0f);
	prim_lookup[1].max_pos = glm::vec4( sphere_config.scene_extent, sphere_config.upper_plane_y,
	                                     sphere_config.scene_extent, 0.0f);

	lumen_scene->prim_lookup_buffer = prm::get_buffer({
		.name        = "Sphere Scene Prim Lookup",
		.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
		.memory_type = vk::BufferType::GPU,
		.size        = prim_lookup.size() * sizeof(PrimMeshInfo),
		.data        = prim_lookup.data(),
	});

	// ── 3. Lights ─────────────────────────────────────────────────────────
	lumen_scene->gpu_lights               = sphere_data.gpu_lights;
	lumen_scene->total_light_triangle_cnt = sphere_data.total_light_triangle_cnt;
	lumen_scene->total_light_area         = sphere_data.total_light_area;

	if (!sphere_data.gpu_lights.empty()) {
		lumen_scene->mesh_lights_buffer = prm::get_buffer({
			.name        = "Sphere Lights",
			.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
			.memory_type = vk::BufferType::GPU,
			.size        = sphere_data.gpu_lights.size() * sizeof(Light),
			.data        = sphere_data.gpu_lights.data(),
		});
	}

	// ── 4. Scene dimensions ───────────────────────────────────────────────
	float ext = sphere_config.scene_extent * 2.0f;
	float ht  = sphere_config.upper_plane_y - sphere_config.lower_plane_y;
	lumen_scene->m_dimensions.radius =
		0.5f * std::sqrt(ext * ext + ht * ht + ext * ext);
	lumen_scene->m_dimensions.center =
		glm::vec3(0.0f, (sphere_config.lower_plane_y + sphere_config.upper_plane_y) * 0.5f, 0.0f);

	// ── 5. Base integrator init (output texture, UBO, callbacks) ──────────
	Integrator::init();

	// Override camera to match BIM renderer: yaw=-22°, no pitch.
	lumen_scene->camera->rotation = glm::vec3(0.0f, -22.0f, 0.0f);
	update_uniform_buffers();

	// ── 6. Sphere primitive buffer ────────────────────────────────────────
	sphere_buffer = prm::get_buffer({
		.name        = "Sphere Primitives",
		.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
		.memory_type = vk::BufferType::GPU,
		.size        = sphere_data.spheres.size() * sizeof(SpherePrimitive),
		.data        = sphere_data.spheres.data(),
	});

	// ── 7. AABB buffer (kept alive for BLAS build in create_accel) ────────
	aabb_buffer = prm::get_buffer({
		.name        = "Sphere AABBs",
		.usage       = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
		               VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
		.memory_type = vk::BufferType::GPU,
		.size        = sphere_data.aabbs.size() * sizeof(VkAabbPositionsKHR),
		.data        = sphere_data.aabbs.data(),
	});

	// ── 8. ReSTIR DI buffers ──────────────────────────────────────────────
	g_buffer = prm::get_buffer({
		.name        = "G-Buffer",
		.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
		               VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		.memory_type = vk::BufferType::GPU,
		.size        = Window::width() * Window::height() * sizeof(RestirGBufferData),
	});

	temporal_reservoir_buffer = prm::get_buffer({
		.name        = "Temporal Reservoirs",
		.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
		               VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		.memory_type = vk::BufferType::GPU,
		.size        = Window::width() * Window::height() * sizeof(RestirReservoir),
	});

	passthrough_reservoir_buffer = prm::get_buffer({
		.name        = "Passthrough Reservoirs",
		.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
		               VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		.memory_type = vk::BufferType::GPU,
		.size        = Window::width() * Window::height() * sizeof(RestirReservoir),
	});

	spatial_reservoir_buffer = prm::get_buffer({
		.name        = "Spatial Reservoirs",
		.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
		               VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		.memory_type = vk::BufferType::GPU,
		.size        = Window::width() * Window::height() * sizeof(RestirReservoir),
	});

	tmp_col_buffer = prm::get_buffer({
		.name        = "Temporary Color",
		.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
		.memory_type = vk::BufferType::GPU,
		.size        = Window::width() * Window::height() * sizeof(float) * 3,
	});

	// ── 9. SceneDesc ──────────────────────────────────────────────────────
	SceneDesc desc{};
	desc.compact_vertices_addr      = lumen_scene->compact_vertices_buffer->get_device_address();
	desc.index_addr                 = lumen_scene->index_buffer->get_device_address();
	desc.material_addr              = lumen_scene->materials_buffer->get_device_address();
	desc.prim_info_addr             = lumen_scene->prim_lookup_buffer->get_device_address();
	desc.g_buffer_addr              = g_buffer->get_device_address();
	desc.temporal_reservoir_addr    = temporal_reservoir_buffer->get_device_address();
	desc.passthrough_reservoir_addr = passthrough_reservoir_buffer->get_device_address();
	desc.spatial_reservoir_addr     = spatial_reservoir_buffer->get_device_address();
	desc.color_storage_addr         = tmp_col_buffer->get_device_address();
	desc.sphere_prims_addr          = sphere_buffer->get_device_address();

	lumen_scene->scene_desc_buffer = prm::get_buffer({
		.name        = "Scene Desc",
		.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
		.memory_type = vk::BufferType::GPU,
		.size        = sizeof(SceneDesc),
		.data        = &desc,
	});

	// ── 10. Register buffer addresses with render graph ───────────────────
	pc_ray.total_light_area = 0;
	frame_num               = 0;

	assert(vk::render_graph()->settings.shader_inference == true);
	lumen::RenderGraph* rg = vk::render_graph();
	REGISTER_BUFFER_WITH_ADDRESS(SceneDesc, desc, compact_vertices_addr,      lumen_scene->compact_vertices_buffer, rg);
	REGISTER_BUFFER_WITH_ADDRESS(SceneDesc, desc, index_addr,                 lumen_scene->index_buffer,            rg);
	REGISTER_BUFFER_WITH_ADDRESS(SceneDesc, desc, material_addr,              lumen_scene->materials_buffer,        rg);
	REGISTER_BUFFER_WITH_ADDRESS(SceneDesc, desc, prim_info_addr,             lumen_scene->prim_lookup_buffer,      rg);
	REGISTER_BUFFER_WITH_ADDRESS(SceneDesc, desc, g_buffer_addr,              g_buffer,                             rg);
	REGISTER_BUFFER_WITH_ADDRESS(SceneDesc, desc, temporal_reservoir_addr,    temporal_reservoir_buffer,            rg);
	REGISTER_BUFFER_WITH_ADDRESS(SceneDesc, desc, passthrough_reservoir_addr, passthrough_reservoir_buffer,         rg);
	REGISTER_BUFFER_WITH_ADDRESS(SceneDesc, desc, spatial_reservoir_addr,     spatial_reservoir_buffer,             rg);
	REGISTER_BUFFER_WITH_ADDRESS(SceneDesc, desc, color_storage_addr,         tmp_col_buffer,                       rg);
	REGISTER_BUFFER_WITH_ADDRESS(SceneDesc, desc, sphere_prims_addr,          sphere_buffer,                        rg);
}

// ─────────────────────────────────────────────────────────────────────────────
// create_accel  (identical to ReSTIRGISphere)
// ─────────────────────────────────────────────────────────────────────────────

void ReSTIRDISphere::create_accel(vk::BVH& tlas_out, std::vector<vk::BVH>& blases) {
	// ── BLAS 0: sphere AABBs (procedural geometry) ────────────────────────
	VkAccelerationStructureGeometryKHR aabb_geom{};
	aabb_geom.sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
	aabb_geom.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
	aabb_geom.flags        = VK_GEOMETRY_OPAQUE_BIT_KHR;
	aabb_geom.geometry.aabbs.sType              = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
	aabb_geom.geometry.aabbs.data.deviceAddress = aabb_buffer->get_device_address();
	aabb_geom.geometry.aabbs.stride             = sizeof(VkAabbPositionsKHR);

	VkAccelerationStructureBuildRangeInfoKHR sphere_range{};
	sphere_range.primitiveCount = (uint32_t)sphere_data.spheres.size();

	vk::BlasInput sphere_blas_input;
	sphere_blas_input.as_geom.push_back(aabb_geom);
	sphere_blas_input.as_build_offset_info.push_back(sphere_range);

	// ── BLAS 1: plane triangles ───────────────────────────────────────────
	LumenPrimMesh plane_mesh{};
	plane_mesh.vtx_offset   = 0;
	plane_mesh.first_idx    = 0;
	plane_mesh.idx_count    = (uint32_t)sphere_data.plane_indices.size();
	plane_mesh.vtx_count    = (uint32_t)sphere_data.plane_positions.size();
	plane_mesh.world_matrix = glm::mat4(1.0f);

	VkDeviceAddress vertex_addr = lumen_scene->vertex_buffer->get_device_address();
	VkDeviceAddress index_addr  = lumen_scene->index_buffer->get_device_address();
	vk::BlasInput plane_blas_input = vk::to_vk_geometry(plane_mesh, vertex_addr, index_addr);

	vk::build_blas(blases, {sphere_blas_input, plane_blas_input},
	               VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);

	// ── TLAS: 3 instances ─────────────────────────────────────────────────
	std::vector<VkAccelerationStructureInstanceKHR> instances(3);

	// Instance 0: sphere BLAS
	instances[0].transform               = vk::to_vk_matrix(glm::mat4(1.0f));
	instances[0].instanceCustomIndex     = 0xFF;
	instances[0].mask                    = 0xFF;
	instances[0].instanceShaderBindingTableRecordOffset = 0;
	instances[0].flags                   = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
	instances[0].accelerationStructureReference = blases[0].get_blas_device_address();

	// Instance 1: lower plane
	glm::mat4 lower_transform = glm::mat4(1.0f);
	lower_transform[3][1] = sphere_config.lower_plane_y;
	instances[1].transform               = vk::to_vk_matrix(lower_transform);
	instances[1].instanceCustomIndex     = 0;
	instances[1].mask                    = 0xFF;
	instances[1].instanceShaderBindingTableRecordOffset = 2;
	instances[1].flags                   = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
	instances[1].accelerationStructureReference = blases[1].get_blas_device_address();

	// Instance 2: upper plane (Y-flip + translate)
	glm::mat4 upper_transform = glm::mat4(1.0f);
	upper_transform[1][1] = -1.0f;
	upper_transform[3][1] = sphere_config.upper_plane_y;
	instances[2].transform               = vk::to_vk_matrix(upper_transform);
	instances[2].instanceCustomIndex     = 1;
	instances[2].mask                    = 0xFF;
	instances[2].instanceShaderBindingTableRecordOffset = 2;
	instances[2].flags                   = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
	instances[2].accelerationStructureReference = blases[1].get_blas_device_address();

	vk::build_tlas(tlas_out, instances,
	               VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
}

// ─────────────────────────────────────────────────────────────────────────────
// render
// ─────────────────────────────────────────────────────────────────────────────

void ReSTIRDISphere::render() {
	pc_ray.size_x               = Window::width();
	pc_ray.size_y               = Window::height();
	pc_ray.num_lights           = (int)lumen_scene->gpu_lights.size();
	pc_ray.time                 = rand() % UINT_MAX;
	pc_ray.max_depth            = 1;   // BIM-matched: direct illumination only (no multi-bounce PT)
	pc_ray.sky_col              = glm::vec3(0.0f);
	pc_ray.do_spatiotemporal    = do_spatiotemporal;
	pc_ray.random_num           = rand() % UINT_MAX;
	pc_ray.total_light_area     = lumen_scene->total_light_area;
	pc_ray.light_triangle_count = (int)lumen_scene->total_light_triangle_cnt;
	pc_ray.enable_accumulation  = enable_accumulation;
	pc_ray.frame_num            = frame_num;

	const std::initializer_list<lumen::ResourceBinding> rt_bindings = {
		output_tex,
		scene_ubo_buffer,
		lumen_scene->scene_desc_buffer,
	};

	// ── Temporal pass + initial path tracing ──────────────────────────────
	vk::render_graph()
		->add_rt("ReSTIRDISphere - Temporal",
		         {.shaders = make_sphere_shaders(rgen_paths[0]),
		          .dims    = {Window::width(), Window::height()}})
		.push_constants(&pc_ray)
		.zero(g_buffer)
		.zero(spatial_reservoir_buffer)
		.zero(temporal_reservoir_buffer, !do_spatiotemporal)
		.bind(rt_bindings)
		.bind(lumen_scene->mesh_lights_buffer)
		.bind_texture_array(lumen_scene->scene_textures)
		.bind_tlas(tlas);

	// ── Spatial pass ──────────────────────────────────────────────────────
	vk::render_graph()
		->add_rt("ReSTIRDISphere - Spatial",
		         {.shaders = make_sphere_shaders(rgen_paths[1]),
		          .dims    = {Window::width(), Window::height()}})
		.push_constants(&pc_ray)
		.bind(rt_bindings)
		.bind(lumen_scene->mesh_lights_buffer)
		.bind_texture_array(lumen_scene->scene_textures)
		.bind_tlas(tlas);

	// ── Output pass ───────────────────────────────────────────────────────
	vk::render_graph()
		->add_rt("ReSTIRDISphere - Output",
		         {.shaders = make_sphere_shaders(rgen_paths[2]),
		          .dims    = {Window::width(), Window::height()}})
		.push_constants(&pc_ray)
		.bind(rt_bindings)
		.bind(lumen_scene->mesh_lights_buffer)
		.bind_texture_array(lumen_scene->scene_textures)
		.bind_tlas(tlas);

	if (!do_spatiotemporal)
		do_spatiotemporal = true;
}

// ─────────────────────────────────────────────────────────────────────────────
// update / gui / destroy
// ─────────────────────────────────────────────────────────────────────────────

bool ReSTIRDISphere::update() {
	frame_num++;
	bool updated = Integrator::update();
	if (updated) {
		frame_num = 0;
	}
	return updated;
}

bool ReSTIRDISphere::gui() {
	bool result = false;
	result |= ImGui::Checkbox("Enable accumulation", &enable_accumulation);
	return result;
}

void ReSTIRDISphere::destroy() {
	Integrator::destroy();
	auto buffer_list = {
		aabb_buffer,
		sphere_buffer,
		g_buffer,
		temporal_reservoir_buffer,
		passthrough_reservoir_buffer,
		spatial_reservoir_buffer,
		tmp_col_buffer,
	};
	for (vk::Buffer* b : buffer_list) {
		prm::remove(b);
	}
	// lumen_scene geometry buffers freed by LumenScene::destroy()
}
