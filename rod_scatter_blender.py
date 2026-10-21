"""
QUANTUM SCATTERING — a 3D wave packet hits a rod (Blender 4.2+ / 5.x)
@AlexCarten_

How to use:  File > New > General  ->  Scripting  ->  Open this file  ->  Run Script.
First run solves the Schrodinger equation (~1 min, watch Window > Toggle System Console),
then the scene appears. Render > Render Animation (Ctrl+F12) writes ONE mp4 file:
~/OrbitalRenders/rod_scatter/rod_scatter.mp4

Physics: the rod is infinite along z, so the 3D wave function splits exactly into
psi(x,y,t) * g(z,t). psi(x,y,t) is solved numerically (split-step Fourier, hbar = m = 1),
g(z,t) is a free Gaussian. The orange surface is the exact 3D isosurface |psi|^2 = const,
the white stripes are the phase of the wave.

Crash-free: no Python runs during playback or render. Each frame of the solution is an EXR
file; Geometry Nodes read the right file for the current frame on their own.
"""
import bpy, math, os
import numpy as np

# ---------------- CONFIG ----------------
SECONDS   = 14
FPS       = 30
K0        = 3.0      # packet momentum along +x
SIGMA     = 2.2      # packet radius
X0        = -9.0     # start position
ROD_R     = 1.2      # rod radius
ROD_V     = 60.0     # rod potential (impenetrable)
T_TOTAL   = 5.4      # simulated time
ISO       = 0.035     # isosurface level, as a fraction of the starting peak density
N, L      = 320, 40.0    # simulation grid
CROP      = 14.0     # half-size of the visible region
VOX_XY    = 200      # voxel resolution across x and y (higher = smoother, slower)
VOX_Z     = 90       # voxel resolution along z
ZMAX      = 6.5      # vertical extent of the volume
CAMERA_SHAKE = 0.03
ENGINE    = "EEVEE"  # "EEVEE" or "CYCLES"
SAMPLES   = 64
RES_SCALE = 100
OUTPUT    = os.path.join(os.path.expanduser("~"), "OrbitalRenders", "rod_scatter")

COL = dict(
    packet=(0.85, 0.22, 0.015),    # burnt orange
    stripe=(0.95, 0.92, 0.85),     # cream
    rod=(0.04, 0.32, 0.07),        # deep green
    sky_top=(0.004, 0.008, 0.04),
    sky_bottom=(0.06, 0.11, 0.32),
)

NF = SECONDS * FPS
DATA_DIR = os.path.join(OUTPUT, "psi_frames")

# ---------------- physics ----------------
def simulate():
    print("[1/3] Solving the Schrodinger equation (2D x free z)...")
    os.makedirs(DATA_DIR, exist_ok=True)
    x = np.linspace(-L / 2, L / 2, N, endpoint=False); dx = x[1] - x[0]
    X, Y = np.meshgrid(x, x, indexing="xy")
    k = 2 * np.pi * np.fft.fftfreq(N, dx); KX, KY = np.meshgrid(k, k, indexing="xy")
    R = np.hypot(X, Y)
    V = ROD_V * 0.5 * (1 - np.tanh((R - ROD_R) / 0.12))
    psi = (1 / (math.sqrt(2 * math.pi) * SIGMA)) * np.exp(-((X - X0)**2 + Y**2) / (4 * SIGMA**2) + 1j * K0 * X)
    SUB = 8; dt = T_TOTAL / (NF * SUB)
    edge = np.clip((np.abs(X) - 16) / 4, 0, None)**2 + np.clip((np.abs(Y) - 16) / 4, 0, None)**2
    half_v = np.exp((-0.5j * V - 15 * edge) * dt)
    kin = np.exp(-0.5j * (KX**2 + KY**2) * dt)
    sel = np.where(np.abs(x) <= CROP)[0]; i0, i1 = sel[0], sel[-1] + 1
    T = i1 - i0
    peak0 = (2 * math.pi * SIGMA**2) ** -1.5
    c = ISO * peak0
    img = bpy.data.images.new("_psi_tmp", T, T, alpha=True, float_buffer=True)
    img.colorspace_settings.name = 'Non-Color'
    for f in range(NF):
        if f:
            for _ in range(SUB):
                psi = half_v * np.fft.ifft2(kin * np.fft.fft2(half_v * psi))
        t = f * SUB * dt
        sz = SIGMA * math.sqrt(1 + (t / (2 * SIGMA**2))**2)        # free spreading along z
        p = psi[i0:i1, i0:i1]
        D = np.abs(p)**2
        arg = D / (c * math.sqrt(2 * math.pi) * sz)
        q = 2 * sz**2 * np.log(np.maximum(arg, 1e-12))             # height^2 of the isosurface (>0 inside)
        ph = np.angle(p)
        a = np.zeros((T, T, 4), np.float32)
        a[..., 0] = np.clip(q, -50, None); a[..., 1] = np.cos(ph); a[..., 2] = np.sin(ph); a[..., 3] = 1
        img.pixels.foreach_set(a.ravel())
        img.filepath_raw = os.path.join(DATA_DIR, f"psi_{f + 1:04d}.exr"); img.file_format = 'OPEN_EXR'
        img.save()
        if f % 60 == 0: print(f"    frame {f + 1}/{NF}")
    bpy.data.images.remove(img)
    return T

# ---------------- helpers ----------------
def clear_scene():
    for h in list(bpy.app.handlers.frame_change_pre):
        if getattr(h, "__name__", "") in ("tunnel_frame", "update"):
            bpy.app.handlers.frame_change_pre.remove(h)
    for ob in list(bpy.data.objects): bpy.data.objects.remove(ob, do_unlink=True)
    for coll in (bpy.data.meshes, bpy.data.curves, bpy.data.materials, bpy.data.lights,
                 bpy.data.cameras, bpy.data.actions, bpy.data.images):
        for b in list(coll):
            if b.users == 0: coll.remove(b)
    for ng in list(bpy.data.node_groups):
        if ng.users == 0 and ng.bl_idname == 'GeometryNodeTree': bpy.data.node_groups.remove(ng)

def link(ob):
    bpy.context.scene.collection.objects.link(ob); return ob

def principled(name, rgb, rough=0.35, coat=0.0):
    m = bpy.data.materials.new(name); m.use_nodes = True
    bs = [n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED"][0]
    bs.inputs["Base Color"].default_value = (*rgb, 1)
    bs.inputs["Roughness"].default_value = rough
    for nm in ("Coat Weight", "Clearcoat"):
        if nm in bs.inputs: bs.inputs[nm].default_value = coat; break
    return m, bs

def packet_material():
    m, bs = principled("Packet", COL["packet"], rough=0.28, coat=0.6)
    N, L = m.node_tree.nodes, m.node_tree.links
    at = N.new("ShaderNodeAttribute"); at.attribute_type = 'GEOMETRY'; at.attribute_name = "phase"
    mr = N.new("ShaderNodeMapRange"); mr.inputs[1].default_value = -1; mr.inputs[2].default_value = 1
    ramp = N.new("ShaderNodeValToRGB"); cr = ramp.color_ramp
    cr.elements[0].position = 0.0; cr.elements[0].color = (*COL["packet"], 1)
    cr.elements[1].position = 0.90; cr.elements[1].color = (*COL["packet"], 1)
    e = cr.elements.new(0.955); e.color = (*COL["stripe"], 1)
    e = cr.elements.new(1.0); e.color = (*COL["stripe"], 1)
    L.new(at.outputs["Fac"], mr.inputs[0]); L.new(mr.outputs[0], ramp.inputs[0])
    L.new(ramp.outputs[0], bs.inputs["Base Color"])
    return m

# ---------------- geometry nodes surface ----------------
def build_packet(T):
    first = os.path.join(DATA_DIR, "psi_0001.exr")
    img = bpy.data.images.load(first); img.source = 'SEQUENCE'
    img.colorspace_settings.name = 'Non-Color'
    ng = bpy.data.node_groups.new("PacketIso", 'GeometryNodeTree')
    ng.interface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    ng.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    N, L = ng.nodes, ng.links
    out = N.new("NodeGroupOutput")
    def math_(op, a, b=None):
        n = N.new("ShaderNodeMath"); n.operation = op
        for i, v in enumerate((a, b)):
            if v is None: continue
            if isinstance(v, (int, float)): n.inputs[i].default_value = v
            else: L.new(v, n.inputs[i])
        return n.outputs[0]
    def sample(pos_socket):
        sep = N.new("ShaderNodeSeparateXYZ"); L.new(pos_socket, sep.inputs[0])
        sc = (T - 1) / (T * 2 * CROP); off = 0.5 / T + (T - 1) / (2 * T)
        u = math_('MULTIPLY_ADD', sep.outputs["X"], sc); u.node.inputs[2].default_value = off
        v = math_('MULTIPLY_ADD', sep.outputs["Y"], sc); v.node.inputs[2].default_value = off
        uv = N.new("ShaderNodeCombineXYZ"); L.new(u, uv.inputs["X"]); L.new(v, uv.inputs["Y"])
        tex = N.new("GeometryNodeImageTexture"); tex.inputs["Image"].default_value = img
        tex.interpolation = 'Linear'; tex.extension = 'EXTEND'
        st = N.new("GeometryNodeInputSceneTime")
        fr = math_('MINIMUM', math_('MAXIMUM', st.outputs["Frame"], 1.0), float(NF))
        L.new(fr, tex.inputs["Frame"]); L.new(uv.outputs[0], tex.inputs["Vector"])
        cs = N.new("FunctionNodeSeparateColor"); L.new(tex.outputs["Color"], cs.inputs[0])
        return cs.outputs["Red"], cs.outputs["Green"], sep.outputs["Z"]
    # density = q(x,y) - z^2 : positive inside the exact 3D isosurface
    pos = N.new("GeometryNodeInputPosition")
    q, _, z = sample(pos.outputs[0])
    dens = math_('SUBTRACT', q, math_('MULTIPLY', z, z))
    cube = N.new("GeometryNodeVolumeCube")
    L.new(dens, cube.inputs["Density"])
    cube.inputs["Background"].default_value = -50.0
    cube.inputs["Min"].default_value = (-CROP, -CROP, -ZMAX); cube.inputs["Max"].default_value = (CROP, CROP, ZMAX)
    cube.inputs["Resolution X"].default_value = VOX_XY; cube.inputs["Resolution Y"].default_value = VOX_XY
    cube.inputs["Resolution Z"].default_value = VOX_Z
    v2m = N.new("GeometryNodeVolumeToMesh")
    if hasattr(v2m, "resolution_mode"): v2m.resolution_mode = 'GRID'          # Blender 4.x
    else:                                                                    # Blender 5.x: menu socket
        for val in ('Grid', 'GRID'):
            try: v2m.inputs["Resolution Mode"].default_value = val; break
            except Exception: pass
    v2m.inputs["Threshold"].default_value = 0.0; v2m.inputs["Adaptivity"].default_value = 0.0
    L.new(cube.outputs[0], v2m.inputs["Volume"])
    # phase does not depend on z: sample it at the mesh points
    pos2 = N.new("GeometryNodeInputPosition")
    _, cosph, _ = sample(pos2.outputs[0])
    store = N.new("GeometryNodeStoreNamedAttribute"); store.data_type = 'FLOAT'; store.domain = 'POINT'
    store.inputs["Name"].default_value = "phase"
    L.new(v2m.outputs[0], store.inputs["Geometry"]); L.new(cosph, store.inputs["Value"])
    smooth = N.new("GeometryNodeSetShadeSmooth"); L.new(store.outputs[0], smooth.inputs["Geometry"])
    mat = N.new("GeometryNodeSetMaterial"); mat.inputs["Material"].default_value = packet_material()
    L.new(smooth.outputs[0], mat.inputs["Geometry"]); L.new(mat.outputs[0], out.inputs[0])
    me = bpy.data.meshes.new("Packet"); ob = link(bpy.data.objects.new("Packet", me))
    ob.modifiers.new("iso", 'NODES').node_group = ng
    return ob

def build_rod():
    import bmesh
    me = bpy.data.meshes.new("Rod"); bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=96, radius1=ROD_R * 0.92, radius2=ROD_R * 0.92, depth=14)
    bm.to_mesh(me); bm.free()
    for p in me.polygons: p.use_smooth = len(p.vertices) == 4
    ob = link(bpy.data.objects.new("Rod", me))
    ob.location = (0, 0, 0)
    m, _ = principled("Rod", COL["rod"], rough=0.3, coat=0.4); me.materials.append(m)

def fcurves(ob):
    ad = ob.animation_data
    if not ad or not ad.action: return []
    act = ad.action
    if hasattr(act, "fcurves"):
        try:
            if len(act.fcurves): return list(act.fcurves)
        except Exception: pass
    out = []
    for layer in getattr(act, "layers", []):
        for strip in layer.strips:
            for cb in strip.channelbags: out += list(cb.fcurves)
    return out

def build_camera():
    cd = bpy.data.cameras.new("Camera"); cd.lens = 32
    cam = link(bpy.data.objects.new("Camera", cd)); bpy.context.scene.camera = cam
    tgt = link(bpy.data.objects.new("CamTarget", None))
    c = cam.constraints.new('TRACK_TO'); c.target = tgt; c.track_axis = 'TRACK_NEGATIVE_Z'; c.up_axis = 'UP_Y'
    # reference framing: front view from above, rod in the middle,
    # the packet comes in from the back-left; slow push-in
    keys = [(0.00, (7, -20, 21), (-2, 0, -1)),
            (0.45, (5, -18, 19), (0, 0, -1)),
            (1.00, (4, -17, 18), (1, 0, -1))]
    for u_, loc, look in keys:
        f = max(1, int(u_ * NF))
        cam.location = loc; cam.keyframe_insert("location", frame=f)
        tgt.location = look; tgt.keyframe_insert("location", frame=f)
    if CAMERA_SHAKE > 0:
        for i, fc in enumerate(fcurves(cam)):
            nm = fc.modifiers.new('NOISE'); nm.scale = 50; nm.strength = CAMERA_SHAKE; nm.phase = i * 11.3

def build_world_and_lights():
    w = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = w; w.use_nodes = True
    N, L = w.node_tree.nodes, w.node_tree.links
    for n in list(N): N.remove(n)
    tc = N.new("ShaderNodeTexCoord"); sep = N.new("ShaderNodeSeparateXYZ")
    ramp = N.new("ShaderNodeValToRGB"); bg = N.new("ShaderNodeBackground"); out = N.new("ShaderNodeOutputWorld")
    ramp.color_ramp.elements[0].position = 0.25; ramp.color_ramp.elements[0].color = (*COL["sky_bottom"], 1)
    ramp.color_ramp.elements[1].position = 0.75; ramp.color_ramp.elements[1].color = (*COL["sky_top"], 1)
    L.new(tc.outputs["Window"], sep.inputs[0]); L.new(sep.outputs["Y"], ramp.inputs[0])
    L.new(ramp.outputs[0], bg.inputs["Color"]); L.new(bg.outputs[0], out.inputs[0])
    sun = link(bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", 'SUN')))
    sun.data.energy = 3.5; sun.data.angle = math.radians(8)
    sun.rotation_euler = (math.radians(35), math.radians(-20), math.radians(40))
    fill = link(bpy.data.objects.new("Fill", bpy.data.lights.new("Fill", 'AREA')))
    fill.data.energy = 3000; fill.data.size = 20; fill.location = (-10, -20, 12)
    fill.rotation_euler = (math.radians(60), 0, math.radians(-25))

def setup_render(scene):
    r = scene.render
    r.resolution_x, r.resolution_y, r.resolution_percentage = 1080, 1920, RES_SCALE
    r.fps = FPS; scene.frame_start, scene.frame_end = 1, NF
    if ENGINE == "CYCLES":
        r.engine = 'CYCLES'; scene.cycles.samples = SAMPLES
        try:
            prefs = bpy.context.preferences.addons['cycles'].preferences
            for backend in ('OPTIX', 'CUDA', 'HIP', 'METAL', 'ONEAPI'):
                try:
                    prefs.compute_device_type = backend; prefs.get_devices()
                    if any(d.type == backend for d in prefs.devices):
                        for d in prefs.devices: d.use = d.type == backend
                        scene.cycles.device = 'GPU'; break
                except Exception: pass
        except Exception: pass
    else:
        for eng in ('BLENDER_EEVEE', 'BLENDER_EEVEE_NEXT'):
            try: r.engine = eng; break
            except Exception: pass
        try: scene.eevee.taa_render_samples = SAMPLES
        except Exception: pass
    try:
        scene.view_settings.view_transform = 'AgX'
        for lk in ('AgX - Punchy', 'Punchy'):
            try: scene.view_settings.look = lk; break
            except Exception: pass
    except Exception: pass
    r.filepath = os.path.join(OUTPUT, "rod_scatter.mp4")
    ims = r.image_settings
    try: ims.media_type = 'VIDEO'
    except Exception: pass
    ims.file_format = 'FFMPEG'
    r.ffmpeg.format = 'MPEG4'; r.ffmpeg.codec = 'H264'
    r.ffmpeg.constant_rate_factor = 'HIGH'; r.ffmpeg.ffmpeg_preset = 'GOOD'
    r.use_lock_interface = True

def build():
    scene = bpy.context.scene
    clear_scene()
    T = simulate()
    print("[2/3] Building the scene...")
    build_packet(T); build_rod(); build_camera(); build_world_and_lights(); setup_render(scene)
    scene.frame_set(1)
    print(f"[3/3] Ready: {NF} frames. Ctrl+F12 -> {scene.render.filepath}")

def _deferred():
    try: build()
    except Exception:
        import traceback; traceback.print_exc()
    return None

if bpy.app.background:
    build()
else:
    print("Building... (Window > Toggle System Console to watch)")
    bpy.app.timers.register(_deferred, first_interval=0.2)
