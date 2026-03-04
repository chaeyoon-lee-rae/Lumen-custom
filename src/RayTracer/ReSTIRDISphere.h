#pragma once
#include "Integrator.h"
#include "SphereScene.h"
#include "shaders/integrators/restir/di/restirdi_commons.h"

// ReSTIR DI integrator for a fully procedural sphere scene.
// Reuses the existing DI shaders (temporal_pass/spatial_pass/output.rgen)
// with the sphere SBT hit-group layout from ReSTIRGISphere.
//
// SBT hit-group layout (matches create_accel instance SBT offsets):
//   HG 0: sphere.rchit + sphere.rint  – sphere primary rays   (instanceSBT=0, sbtOffset=0)
//   HG 1: sphere.rchit + sphere.rint  – sphere shadow  rays   (instanceSBT=0, sbtOffset=1)
//   HG 2: ray.rchit   + ray.rahit     – plane  primary rays   (instanceSBT=2, sbtOffset=0)
//   HG 3: ray.rchit                   – plane  shadow  rays   (instanceSBT=2, sbtOffset=1)
class ReSTIRDISphere : public Integrator {
   public:
	ReSTIRDISphere(LumenScene* lumen_scene, const vk::BVH& tlas)
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

	vk::Buffer* aabb_buffer                  = nullptr;
	vk::Buffer* sphere_buffer                = nullptr;  // binding 10 – sphere primitives

	vk::Buffer* g_buffer                     = nullptr;
	vk::Buffer* passthrough_reservoir_buffer = nullptr;
	vk::Buffer* temporal_reservoir_buffer    = nullptr;
	vk::Buffer* spatial_reservoir_buffer     = nullptr;
	vk::Buffer* tmp_col_buffer               = nullptr;

	PCReSTIR pc_ray{};
	bool do_spatiotemporal   = false;   // BIM-matched: reuseSamples → temporal reservoir reuse across frames
	bool enable_accumulation = false;  // BIM reuseSamples does not use frame color averaging
};
