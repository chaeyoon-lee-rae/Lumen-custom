#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require

#include "commons.h"
#include "utils.glsl"

layout(location = 0) rayPayloadInEXT HitPayload payload;
hitAttributeEXT SphereHitAttribs sphere_attribs;

layout(set = 0, binding = 2, scalar) buffer SceneDesc_ { SceneDesc scene_desc; };
layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer SpherePrims { SpherePrimitive spheres[]; };

#define PI 3.14159265359

void main() {
	SpherePrims sphere_buf = SpherePrims(scene_desc.sphere_prims_addr);
	SpherePrimitive s = sphere_buf.spheres[gl_PrimitiveID];

	vec3 world_pos = gl_WorldRayOriginEXT + gl_HitTEXT * gl_WorldRayDirectionEXT;
	vec3 n = sphere_attribs.normal;

	float u = (atan(n.z, n.x) + PI) / (2.0 * PI);
	float v = acos(clamp(n.y, -1.0, 1.0)) / PI;

	float area = 4.0 * PI * s.radius * s.radius;

	payload.n_g           = n;
	payload.n_s           = n;
	payload.pos           = world_pos;
	payload.uv            = vec2(u, v);
	payload.material_idx  = sphere_attribs.materialIndex;
	payload.triangle_idx  = gl_PrimitiveID;
	payload.instance_idx  = gl_InstanceCustomIndexEXT;
	payload.area          = area;
	payload.dist          = gl_RayTminEXT + gl_HitTEXT;
	payload.hit_kind      = gl_HitKindEXT;
}
