/*
 * nr86_capture — ReShade add-on.
 * Dumps color + (if Generic Depth is loaded) linearized depth + previous
 * color for Farneback mvec. Does not open NVIDIA Neural Rendering blobs.
 * Offline single-player only. Hide the HUD: this hook is post-UI LDR.
 *
 * F10 = one frame. F9 = toggle burst (needed for motion vectors).
 *
 * SPDX-License-Identifier: MIT
 */

#include <reshade.hpp>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace reshade::api;

struct __declspec(uuid("7c6363c7-f94e-437a-9160-141782c44a98")) generic_depth_data
{
	resource selected_depth_stencil = { 0 };
	resource override_depth_stencil = { 0 };
	resource_view selected_shader_resource = { 0 };
	bool using_backup_texture = false;
};

struct __declspec(uuid("a7e4c19d-6b52-4e08-9c31-0d86a2b5e7f1")) capture_state
{
	uint32_t index = 0;
	bool burst = false;
	std::vector<uint8_t> prev_bgr;
	uint32_t prev_w = 0;
	uint32_t prev_h = 0;
};

static fs::path capture_root()
{
	char base[MAX_PATH] = {};
	size_t n = sizeof(base);
	reshade::get_reshade_base_path(base, &n);
	fs::path root = fs::path(base) / "nr86_capture";
	std::error_code ec;
	fs::create_directories(root, ec);
	return root;
}

static const char *color_format_name(format fmt)
{
	switch (fmt)
	{
	case format::b8g8r8a8_unorm:
	case format::b8g8r8a8_unorm_srgb:
		return "b8g8r8a8";
	case format::b8g8r8x8_unorm:
	case format::b8g8r8x8_unorm_srgb:
		return "b8g8r8x8";
	case format::r8g8b8a8_unorm:
	case format::r8g8b8a8_unorm_srgb:
	case format::r8g8b8x8_unorm:
	case format::r8g8b8x8_unorm_srgb:
		return "r8g8b8a8";
	case format::r10g10b10a2_unorm:
		return "r10g10b10a2";
	case format::r16g16b16a16_float:
		return "r16g16b16a16_float";
	case format::r32g32b32a32_float:
		return "r32g32b32a32_float";
	default:
		return "unsupported";
	}
}

static const char *depth_format_name(format fmt)
{
	switch (fmt)
	{
	case format::d32_float:
	case format::r32_float:
		return "d32_float";
	case format::d24_unorm_s8_uint:
	case format::d24_unorm_x8_uint:
	case format::r24_unorm_x8_uint:
	case format::r24_g8_typeless:
		return "d24_unorm";
	case format::d16_unorm:
		return "d16_unorm";
	case format::d32_float_s8_uint:
	case format::r32_float_x8_uint:
		return "d32_float_s8";
	default:
		return "unsupported";
	}
}

static float f16_to_f32(uint16_t h)
{
	const uint32_t s = (h >> 15) & 1u;
	uint32_t e = (h >> 10) & 0x1fu;
	uint32_t m = h & 0x3ffu;
	uint32_t out;
	if (e == 0)
	{
		if (m == 0)
			out = s << 31;
		else
		{
			e = 1;
			while ((m & 0x400u) == 0)
			{
				m <<= 1;
				--e;
			}
			m &= 0x3ffu;
			out = (s << 31) | ((e + 127 - 15) << 23) | (m << 13);
		}
	}
	else if (e == 31)
		out = (s << 31) | 0x7f800000u | (m << 13);
	else
		out = (s << 31) | ((e + 127 - 15) << 23) | (m << 13);
	float f;
	std::memcpy(&f, &out, 4);
	return f;
}

static uint8_t to_u8(float x)
{
	if (x < 0.f)
		return 0;
	if (x > 1.f)
		return 255;
	return static_cast<uint8_t>(x * 255.f + 0.5f);
}

static bool unpack_color_bgr(
	format fmt,
	uint32_t w,
	uint32_t h,
	const uint8_t *src,
	uint32_t pitch,
	std::vector<uint8_t> &bgr)
{
	bgr.assign(static_cast<size_t>(w) * h * 3, 0);
	for (uint32_t y = 0; y < h; ++y)
	{
		const uint8_t *row = src + y * pitch;
		uint8_t *dst = bgr.data() + static_cast<size_t>(y) * w * 3;
		switch (fmt)
		{
		case format::b8g8r8a8_unorm:
		case format::b8g8r8a8_unorm_srgb:
		case format::b8g8r8x8_unorm:
		case format::b8g8r8x8_unorm_srgb:
			for (uint32_t x = 0; x < w; ++x)
			{
				dst[x * 3 + 0] = row[x * 4 + 0];
				dst[x * 3 + 1] = row[x * 4 + 1];
				dst[x * 3 + 2] = row[x * 4 + 2];
			}
			break;
		case format::r8g8b8a8_unorm:
		case format::r8g8b8a8_unorm_srgb:
		case format::r8g8b8x8_unorm:
		case format::r8g8b8x8_unorm_srgb:
			for (uint32_t x = 0; x < w; ++x)
			{
				dst[x * 3 + 0] = row[x * 4 + 2];
				dst[x * 3 + 1] = row[x * 4 + 1];
				dst[x * 3 + 2] = row[x * 4 + 0];
			}
			break;
		case format::r10g10b10a2_unorm:
			for (uint32_t x = 0; x < w; ++x)
			{
				uint32_t p;
				std::memcpy(&p, row + x * 4, 4);
				const float r = float(p & 0x3ffu) / 1023.f;
				const float g = float((p >> 10) & 0x3ffu) / 1023.f;
				const float b = float((p >> 20) & 0x3ffu) / 1023.f;
				dst[x * 3 + 0] = to_u8(b);
				dst[x * 3 + 1] = to_u8(g);
				dst[x * 3 + 2] = to_u8(r);
			}
			break;
		case format::r16g16b16a16_float:
			for (uint32_t x = 0; x < w; ++x)
			{
				uint16_t hx[4];
				std::memcpy(hx, row + x * 8, 8);
				dst[x * 3 + 0] = to_u8(f16_to_f32(hx[2]));
				dst[x * 3 + 1] = to_u8(f16_to_f32(hx[1]));
				dst[x * 3 + 2] = to_u8(f16_to_f32(hx[0]));
			}
			break;
		case format::r32g32b32a32_float:
			for (uint32_t x = 0; x < w; ++x)
			{
				float px[4];
				std::memcpy(px, row + x * 16, 16);
				dst[x * 3 + 0] = to_u8(px[2]);
				dst[x * 3 + 1] = to_u8(px[1]);
				dst[x * 3 + 2] = to_u8(px[0]);
			}
			break;
		default:
			return false;
		}
	}
	return true;
}

static bool write_bmp_bgr24(const fs::path &path, uint32_t w, uint32_t h, const uint8_t *bgr)
{
	const uint32_t row_out = (w * 3 + 3) & ~3u;
	const uint32_t img_size = row_out * h;
	const uint32_t file_size = 54 + img_size;
	std::ofstream out(path, std::ios::binary);
	if (!out)
		return false;
	uint8_t hdr[54] = {};
	hdr[0] = 'B';
	hdr[1] = 'M';
	std::memcpy(hdr + 2, &file_size, 4);
	uint32_t off = 54;
	std::memcpy(hdr + 10, &off, 4);
	uint32_t dib = 40;
	std::memcpy(hdr + 14, &dib, 4);
	int32_t wi = static_cast<int32_t>(w);
	int32_t hi = -static_cast<int32_t>(h);
	std::memcpy(hdr + 18, &wi, 4);
	std::memcpy(hdr + 22, &hi, 4);
	uint16_t planes = 1, bpp = 24;
	std::memcpy(hdr + 26, &planes, 2);
	std::memcpy(hdr + 28, &bpp, 2);
	std::memcpy(hdr + 34, &img_size, 4);
	out.write(reinterpret_cast<char *>(hdr), 54);
	std::vector<uint8_t> row(row_out, 0);
	for (uint32_t y = 0; y < h; ++y)
	{
		std::memcpy(row.data(), bgr + static_cast<size_t>(y) * w * 3, w * 3);
		out.write(reinterpret_cast<char *>(row.data()), row_out);
	}
	return true;
}

static bool unpack_depth(
	format fmt,
	uint32_t w,
	uint32_t h,
	const uint8_t *src,
	uint32_t pitch,
	std::vector<float> &depth)
{
	depth.assign(static_cast<size_t>(w) * h, 0.f);
	for (uint32_t y = 0; y < h; ++y)
	{
		const uint8_t *row = src + y * pitch;
		float *dst = depth.data() + static_cast<size_t>(y) * w;
		switch (fmt)
		{
		case format::d32_float:
		case format::r32_float:
			for (uint32_t x = 0; x < w; ++x)
				std::memcpy(dst + x, row + x * 4, 4);
			break;
		case format::d24_unorm_s8_uint:
		case format::d24_unorm_x8_uint:
		case format::r24_unorm_x8_uint:
		case format::r24_g8_typeless:
			for (uint32_t x = 0; x < w; ++x)
			{
				uint32_t p;
				std::memcpy(&p, row + x * 4, 4);
				dst[x] = float(p & 0x00ffffffu) / 16777215.f;
			}
			break;
		case format::d16_unorm:
			for (uint32_t x = 0; x < w; ++x)
			{
				uint16_t p;
				std::memcpy(&p, row + x * 2, 2);
				dst[x] = float(p) / 65535.f;
			}
			break;
		case format::d32_float_s8_uint:
		case format::r32_float_x8_uint:
			for (uint32_t x = 0; x < w; ++x)
				std::memcpy(dst + x, row + x * 8, 4);
			break;
		default:
			return false;
		}
	}
	return true;
}

static bool copy_to_host(
	device *dev,
	command_queue *queue,
	resource src,
	resource_usage src_state,
	const resource_desc &desc,
	resource &host)
{
	resource_desc host_desc = desc;
	host_desc.heap = memory_heap::readback;
	host_desc.usage = resource_usage::copy_dest;
	host_desc.flags = resource_flags::none;
	if (!dev->create_resource(host_desc, nullptr, resource_usage::copy_dest, &host))
		return false;
	command_list *cmd = queue->get_immediate_command_list();
	cmd->barrier(src, src_state, resource_usage::copy_source);
	cmd->copy_resource(src, host);
	cmd->barrier(src, resource_usage::copy_source, src_state);
	queue->flush_immediate_command_list();
	queue->wait_idle();
	return true;
}

static void dump_frame(effect_runtime *runtime, resource back)
{
	capture_state &st = *runtime->get_private_data<capture_state>();
	device *dev = runtime->get_device();
	command_queue *queue = runtime->get_command_queue();
	if (back.handle == 0)
		return;
	const resource_desc desc = dev->get_resource_desc(back);
	resource host = {};
	if (!copy_to_host(dev, queue, back, resource_usage::render_target, desc, host))
	{
		reshade::log::message(reshade::log::level::error, "nr86: color copy failed");
		return;
	}
	subresource_data mapped = {};
	if (!dev->map_texture_region(host, 0, nullptr, map_access::read_only, &mapped))
	{
		dev->destroy_resource(host);
		return;
	}

	const uint32_t w = desc.texture.width;
	const uint32_t h = desc.texture.height;
	std::vector<uint8_t> bgr;
	const char *cf = color_format_name(desc.texture.format);
	if (!unpack_color_bgr(desc.texture.format, w, h, static_cast<const uint8_t *>(mapped.data), mapped.row_pitch, bgr))
	{
		char err[128];
		std::snprintf(err, sizeof(err), "nr86: unsupported color format %u (%s)", static_cast<unsigned>(desc.texture.format), cf);
		reshade::log::message(reshade::log::level::error, err);
		dev->unmap_texture_region(host, 0);
		dev->destroy_resource(host);
		return;
	}
	dev->unmap_texture_region(host, 0);
	dev->destroy_resource(host);

	char idbuf[32];
	std::snprintf(idbuf, sizeof(idbuf), "%06u", st.index++);
	const fs::path dir = capture_root() / idbuf;
	std::error_code ec;
	fs::create_directories(dir, ec);
	write_bmp_bgr24(dir / "color.bmp", w, h, bgr.data());

	bool have_prev = false;
	if (!st.prev_bgr.empty() && st.prev_w == w && st.prev_h == h)
	{
		write_bmp_bgr24(dir / "color_prev.bmp", w, h, st.prev_bgr.data());
		have_prev = true;
	}
	st.prev_bgr.swap(bgr);
	st.prev_w = w;
	st.prev_h = h;

	bool have_depth = false;
	const char *df = "none";
	uint32_t dw = 0, dh = 0;
	if (generic_depth_data *gd = runtime->get_private_data<generic_depth_data>())
	{
		if (gd->selected_depth_stencil.handle != 0)
		{
			const resource_desc dd = dev->get_resource_desc(gd->selected_depth_stencil);
			dw = dd.texture.width;
			dh = dd.texture.height;
			df = depth_format_name(dd.texture.format);
			resource dhost = {};
			resource_usage src_state = gd->using_backup_texture ? resource_usage::shader_resource : resource_usage::depth_stencil;
			if (copy_to_host(dev, queue, gd->selected_depth_stencil, src_state, dd, dhost))
			{
				subresource_data dm = {};
				if (dev->map_texture_region(dhost, 0, nullptr, map_access::read_only, &dm))
				{
					std::vector<float> depth;
					if (unpack_depth(dd.texture.format, dw, dh, static_cast<const uint8_t *>(dm.data), dm.row_pitch, depth))
					{
						std::ofstream raw(dir / "depth.f32", std::ios::binary);
						raw.write(reinterpret_cast<const char *>(depth.data()), static_cast<std::streamsize>(depth.size() * 4));
						have_depth = true;
					}
					else
					{
						char err[128];
						std::snprintf(err, sizeof(err), "nr86: unsupported depth format %u (%s)", static_cast<unsigned>(dd.texture.format), df);
						reshade::log::message(reshade::log::level::error, err);
					}
					dev->unmap_texture_region(dhost, 0);
				}
				dev->destroy_resource(dhost);
			}
		}
	}

	std::ofstream meta(dir / "meta.json");
	meta << "{\n";
	meta << "  \"id\": \"" << idbuf << "\",\n";
	meta << "  \"width\": " << w << ",\n";
	meta << "  \"height\": " << h << ",\n";
	meta << "  \"color\": \"color.bmp\",\n";
	meta << "  \"color_format\": \"" << cf << "\",\n";
	meta << "  \"prev_color\": " << (have_prev ? "\"color_prev.bmp\"" : "null") << ",\n";
	meta << "  \"depth\": " << (have_depth ? "\"depth.f32\"" : "null") << ",\n";
	meta << "  \"depth_format\": \"" << df << "\",\n";
	meta << "  \"depth_width\": " << dw << ",\n";
	meta << "  \"depth_height\": " << dh << ",\n";
	meta << "  \"note\": \"post-ui LDR. hide HUD. not a NVIDIA NR dump. burst F9 for mvec.\"\n";
	meta << "}\n";
	reshade::log::message(reshade::log::level::info, ("nr86: captured " + dir.string()).c_str());
}

static void on_init(effect_runtime *runtime)
{
	runtime->create_private_data<capture_state>();
}

static void on_destroy(effect_runtime *runtime)
{
	runtime->destroy_private_data<capture_state>();
}

static void on_present(effect_runtime *runtime, command_list *, resource_view rtv, resource_view)
{
	capture_state &st = *runtime->get_private_data<capture_state>();
	device *dev = runtime->get_device();
	const resource back = dev->get_resource_from_view(rtv);
	if (runtime->is_key_pressed(VK_F10))
		dump_frame(runtime, back);
	if (runtime->is_key_pressed(VK_F9))
	{
		st.burst = !st.burst;
		reshade::log::message(
			reshade::log::level::info,
			st.burst ? "nr86: burst ON (mvec)" : "nr86: burst OFF");
	}
	if (st.burst)
		dump_frame(runtime, back);
}

extern "C" __declspec(dllexport) const char *NAME = "nr86 capture";
extern "C" __declspec(dllexport) const char *DESCRIPTION =
	"Dump color/depth/prev-color for the Ampere student. Not a DLSS 5 injector. Hide HUD.";

extern "C" __declspec(dllexport) bool AddonInit(HMODULE addon_module, HMODULE reshade_module)
{
	if (!reshade::register_addon(addon_module, reshade_module))
		return false;
	reshade::register_event<reshade::addon_event::init_effect_runtime>(on_init);
	reshade::register_event<reshade::addon_event::destroy_effect_runtime>(on_destroy);
	reshade::register_event<reshade::addon_event::reshade_finish_effects>(on_present);
	return true;
}

extern "C" __declspec(dllexport) void AddonUninit(HMODULE addon_module, HMODULE reshade_module)
{
	reshade::unregister_addon(addon_module, reshade_module);
}
