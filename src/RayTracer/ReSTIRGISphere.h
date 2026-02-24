#pragma once
#include "Integrator.h"
#include "SphereScene.h"
#include "shaders/integrators/restir/gi/restirgi_commons.h"

// ReSTIR GI integrator for a fully procedural sphere scene.
// Geometry is generated at runtime (no mesh file for spheres).
// The scene JSON only provides camera and config; all geometry is synthetic.
//
// SBT hit-group layout (matches create_accel instance SBT offsets):
//   HG 0: sphere.rchit + sphere.rint  – sphere primary rays   (instanceSBT=0, sbtOffset=0)
//   HG 1: sphere.rchit + sphere.rint  – sphere shadow  rays   (instanceSBT=0, sbtOffset=1)
//   HG 2: ray.rchit   + ray.rahit     – plane  primary rays   (instanceSBT=2, sbtOffset=0)
//   HG 3: ray.rchit                   – plane  shadow  rays   (instanceSBT=2, sbtOffset=1)
//
// NOTE: switching integrators in the GUI while this scene is active is not
// supported; restart the application with a different scene JSON instead.
class ReSTIRGISphere : public Integrator {
   public:
	ReSTIRGISphere(LumenScene* lumen_scene, const vk::BVH& tlas)
		: Integrator(lumen_scene, tlas) {}

	virtual void init() override;
	virtual void render() override;
	virtual bool update() override;
	virtual bool gui() override;
	virtual void destroy() override;
	virtual void create_accel(vk::BVH& tlas, std::vector<vk::BVH>& blases) override;

   private:
	SphereSceneConfig sphere_config;
	SphereSceneData   sphere_data;

	vk::Buffer* aabb_buffer              = nullptr;

	vk::Buffer* restir_samples_buffer    = nullptr;
	vk::Buffer* restir_samples_old_buffer = nullptr;
	vk::Buffer* temporal_reservoir_buffer = nullptr;
	vk::Buffer* spatial_reservoir_buffer  = nullptr;
	vk::Buffer* tmp_col_buffer            = nullptr;

	vk::Buffer* sphere_buffer = nullptr;   // binding 10 – sphere primitives

	PCReSTIRGI pc_ray{};
	bool do_spatiotemporal   = false;
	bool enable_accumulation = true;
};
