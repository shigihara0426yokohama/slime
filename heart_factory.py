"""White slime with a heart inside (8 kinds). Usage: python3 heart_factory.py [ids...]"""
import sys, math, copy, numpy as np
from skimage.measure import marching_cubes
import slime_factory as SF
from slime_factory import A, sd_sphere, sd_ellipsoid, sd_round_cone, smin, smax, basis_from_normal, ellipsoid_mesh, merge

def body_heart(p):
    # the white slime's body with a slightly flatter front, so the heart can sit under the face
    d = sd_ellipsoid(p, A(0, 0.70, 0), A(0.80, 0.74, 0.70))
    d = smin(d, sd_ellipsoid(p, A(0, 0.17, 0.0), A(1.22, 0.21, 0.92)), 0.35)
    for c, r in [((-0.80, 0.12, 0.52), 0.24), ((0.82, 0.12, 0.46), 0.24),
                 ((-1.16, 0.12, -0.08), 0.22), ((1.16, 0.12, -0.12), 0.22), ((-0.55, 0.12, -0.72), 0.22), ((0.6, 0.12, -0.7), 0.22)]:
        d = smin(d, sd_sphere(p, A(*c), r), 0.26)
    return smax(d, -p[..., 1], 0.06)

def heart_sdf(p, depth=0.5):
    q = p * A(1, 1, 1 / depth)                                    # flatten front-to-back
    d = smin(sd_sphere(q, A(-0.075, 0.03, 0), 0.085), sd_sphere(q, A(0.075, 0.03, 0), 0.085), 0.02)
    for s in (-1, 1):
        d = smin(d, sd_round_cone(q, A(s * 0.07, 0.0, 0), A(0, -0.135, 0), 0.075, 0.012), 0.03)
    return d * depth

def heart_mesh(depth=0.5, step=0.008, facet=False):
    g = np.arange(-0.2, 0.2 + step, step); X, Y, Z = np.meshgrid(g, g, g, indexing='ij')
    vol = heart_sdf(np.stack([X, Y, Z], -1), depth)
    v, f, n, _ = marching_cubes(vol, 0.0, spacing=(step, step, step)); v += g[0]
    f = f[:, ::-1]
    if facet:   # gem: unshare vertices so every face is flat
        v = v[f.reshape(-1)]; f = np.arange(len(v)).reshape(-1, 3)
    return v, f

def place(v, center, normal, scale):
    R = basis_from_normal(normal)              # columns: u, v, n  -> local z faces out of the body
    return center + (v * scale) @ R.T

def heart_extra(kind):
    def extra(C, hit, parts, wings, stroke):
        c, _ = hit((0.0, 0.40, 2.0)); n = np.array([0, 0.48, 0.87]); n /= np.linalg.norm(n)
        S = 1.08; ctr = c + n * 0.016                              # set into the jelly; the front of the heart shows
        if 'gloss' in parts:                                        # no shine blob over the heart
            g = parts['gloss']; cen = g['v'][g['f']].mean(1); keep = np.linalg.norm(cen - ctr, axis=1) > 0.3
            g['f'] = g['f'][keep]
        b = parts['body']; cen = b['v'][b['f']].mean(1)                # a thin clear jelly skin over the heart -> it looks inside
        sel = b['f'][(np.linalg.norm((cen - ctr) * A(1, 1.25, 1), axis=1) < 0.17) & (cen[:, 2] > 0.3)]
        used = np.unique(sel); remap = -np.ones(len(b['v']), int); remap[used] = np.arange(len(used))
        parts['jelly_cap'] = dict(v=b['v'][used] + b['n'][used] * 0.045, f=remap[sel], color=(0.96, 0.97, 1.0), rough=0.08, alpha=0.38)
        def put(name, vf, color, rough, alpha=1.0, vcol=None):
            v, f = vf; d = dict(v=place(v, ctr, n, S), f=f, color=color, rough=rough, alpha=alpha)
            if vcol is not None: d['vcol'] = vcol
            parts[name] = d
        if kind in ('purple', 'green', 'red'):                      # gummy / glass heart with a soft highlight
            col = dict(purple=(0.52, 0.22, 0.86), green=(0.06, 0.55, 0.28), red=(0.86, 0.06, 0.10))[kind]
            put('heart', heart_mesh(0.55), col, 0.08 if kind == 'red' else 0.18, 0.92)
            put('heart_shine', (ellipsoid_mesh(A(-0.06, 0.06, 0.05), A(0.03, 0.018, 0.01), seg=12, rings=6)), (1, 1, 1), 0.1, 0.8)
        elif kind == 'gold':
            put('heart_metal_gold', heart_mesh(0.55), (1.0, 0.78, 0.25), 0.22)
        elif kind in ('pink', 'blue'):                              # cut gem: flat facets
            col = dict(pink=(0.98, 0.55, 0.68), blue=(0.10, 0.25, 0.85))[kind]
            put('heart_gem', heart_mesh(0.5, step=0.03, facet=True), col, 0.04, 0.95)
        elif kind == 'cookie':
            v, f = heart_mesh(0.3, step=0.005); r = np.linalg.norm(v[:, :2] * A(1, 1.15)[:2], axis=1)
            col = np.tile([0.86, 0.62, 0.32], (len(v), 1))
            edge = heart_sdf(v * A(1, 1, 0), 0.3) > -0.007                         # toasted rim
            col[edge] = [0.70, 0.45, 0.20]
            ang = np.arctan2(v[:, 1], v[:, 0]); front = v[:, 2] > 0.02
            dots = front & (np.abs((r - 0.07) % 0.045 - 0.022) < 0.006) & (np.sin(ang * 12) > 0.3)   # sugar-icing dots
            col[dots] = [0.98, 0.93, 0.80]
            put('heart_cookie', (v, f), (1, 1, 1), 0.85, vcol=col)
        elif kind == 'circuit':
            v, f = heart_mesh(0.35, step=0.005); col = np.tile([0.16, 0.20, 0.26], (len(v), 1))
            front = v[:, 2] > 0.0
            rim = heart_sdf(v * A(1, 1, 0), 0.35) > -0.005
            grid = front & ((np.abs((v[:, 0] + 0.2) % 0.045 - 0.022) < 0.004) & (np.sin(v[:, 1] * 60) > -0.2) | (np.abs((v[:, 1] + 0.2) % 0.05 - 0.025) < 0.004) & (np.sin(v[:, 0] * 50) > 0.1))
            col[grid] = [0.86, 0.66, 0.26]; col[rim] = [0.92, 0.72, 0.30]
            put('heart_board', (v, f), (1, 1, 1), 0.45, vcol=col)
            rv, rf = heart_mesh(0.38, step=0.005); rv = rv * A(1.03, 1.03, 0.85)
            band = np.abs(rv[rf].mean(1)[:, 2]) < 0.012                      # only the rim band around the edge
            put('heart_metal_rim', (rv, rf[band]), (0.95, 0.74, 0.30), 0.25)
            chip = ellipsoid_mesh(A(0, -0.005, 0.055), A(0.045, 0.04, 0.012), seg=4, rings=2)
            put('heart_glow_chip', chip, (0.35, 0.65, 1.0), 0.3)
    return extra

KINDS = dict(purple='むらさきハート', gold='きんのハート', red='あかいハート', pink='ピンクの宝石ハート', cookie='クッキーハート',
             circuit='きかいのハート', green='みどりハート', blue='青い宝石ハート')
for k, nm in KINDS.items():
    c = copy.deepcopy(SF.SLIMES['slime_white']); c['name'] = 'スライム（' + nm + '）'; c['extra'] = heart_extra(k); c['mouth_y'] = 0.585; c['eye_pos'] = (0.29, 0.73); c['body'] = body_heart
    SF.SLIMES['heart_' + k] = c

if __name__ == '__main__':
    for k in (sys.argv[1:] or list(KINDS)):
        SF.write_slime('heart_' + k, f'/home/claude/heart_{k}_slime.glb')
