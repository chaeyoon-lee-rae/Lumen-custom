#ifndef PT_COMMONS
#define PT_COMMONS
vec3 uniform_sample_light(inout uvec4 seed, const Material mat, vec3 pos, const bool side, const vec3 n_s,
						  const vec3 wo, out bool visible) {
	vec3 res = vec3(0);
	// Sample light
	vec3 wi;
	float wi_len;
	float pdf_light_w;
	float pdf_light_a;
	LightRecord record;
	float cos_from_light;
	const vec3 Le =
		sample_light_Li(rand4(seed), pos, pc.num_lights, pdf_light_w, wi, wi_len, pdf_light_a, cos_from_light, record);
	const vec3 p = offset_ray2(pos, n_s);
	float bsdf_pdf;
	float cos_x = dot(n_s, wi);
	vec3 f = eval_bsdf(n_s, wo, mat, 1, side, wi, bsdf_pdf);
	float pdf_light;
	any_hit_payload.hit = 1;
	float shadow_tmax = wi_len - EPS;
	if (get_light_type(record.flags) == LIGHT_SPHERE) {
		// Prevent false self-occlusion: the shadow ray toward a sampled surface point
		// on the sphere would also hit the sphere's near face at t ≈ wi_len - 2*radius,
		// which is accepted as an occluder. Instead, stop just before the entry point.
		Light sph_light = lights[record.light_idx];
		vec3 oc = p - sph_light.pos;
		float b = dot(oc, wi);
		float disc = b * b - (dot(oc, oc) - sph_light.world_radius * sph_light.world_radius);
		shadow_tmax = max(-b - sqrt(max(disc, 0.0)) - EPS, 0.0);
	}
	traceRayEXT(tlas, gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT, 0x1, 1, 0, 1, p, 0, wi,
				shadow_tmax, 1);
	visible = any_hit_payload.hit == 0;
	if (visible && pdf_light_w > 0) {
		const float mis_weight = is_light_delta(record.flags) ? 1 : 1 / (1 + bsdf_pdf / pdf_light_w);
		res += mis_weight * f * abs(cos_x) * Le / pdf_light_w;
	}
	if (get_light_type(record.flags) == LIGHT_AREA || get_light_type(record.flags) == LIGHT_SPHERE) {
		// Sample BSDF
		f = sample_bsdf(n_s, wo, mat, 1, side, wi, bsdf_pdf, cos_x, seed);
		if (bsdf_pdf != 0) {
			traceRayEXT(tlas, flags, 0x1, 0, 0, 0, p, tmin, wi, tmax, 0);
			if (payload.triangle_idx == record.triangle_idx && payload.instance_idx == record.instance_idx) {
				const float wi_len = length(payload.pos - pos);
				const float g = abs(dot(payload.n_s, -wi)) / (wi_len * wi_len);
				const float mis_weight = 1. / (1 + pdf_light_a / (g * bsdf_pdf));
				res += f * mis_weight * abs(cos_x) * Le / bsdf_pdf;
			}
		}
	}
	return res;
}

vec3 uniform_sample_light(inout uvec4 seed, const Material mat, vec3 pos, const bool side, const vec3 n_s,
						  const vec3 wo) {
	bool unused;
	return uniform_sample_light(seed, mat, pos, side, n_s, wo, unused);
}
#endif