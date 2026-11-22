#!/usr/bin/env python3
"""Generate the SVG figures in img/ (no dependencies beyond the stdlib)."""
import math
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "img"
OUT.mkdir(exist_ok=True)

INK = "#1f2933"
MUTED = "#7b8794"
GRID = "#e4e7eb"
A = "#2563eb"   # blue
B = "#dc2626"   # red
C = "#059669"   # green
FONT = "font-family='-apple-system,Segoe UI,Helvetica,Arial,sans-serif'"


def svg(w, h, body):
    return (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {w} {h}' "
            f"width='{w}' height='{h}' {FONT}>\n"
            f"<rect width='{w}' height='{h}' fill='#ffffff' rx='12'/>\n{body}\n</svg>\n")


def text(x, y, s, size=14, fill=INK, anchor="start", weight="normal"):
    return (f"<text x='{x:.1f}' y='{y:.1f}' font-size='{size}' fill='{fill}' "
            f"text-anchor='{anchor}' font-weight='{weight}' stroke='#ffffff' "
            f"stroke-width='4' paint-order='stroke' stroke-linejoin='round'>{s}</text>")


def poly(pts, stroke, width=2, dash=None, fill="none", opacity=1):
    d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    da = f" stroke-dasharray='{dash}'" if dash else ""
    return (f"<polyline points='{d}' fill='{fill}' stroke='{stroke}' "
            f"stroke-width='{width}'{da} stroke-linejoin='round' "
            f"stroke-linecap='round' opacity='{opacity}'/>")


def dot(x, y, r, fill, stroke="none"):
    return f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{r}' fill='{fill}' stroke='{stroke}' stroke-width='2'/>"


# ---------------------------------------------------------------- 1. CBDR
def cbdr():
    W, H = 880, 420
    out = []

    def panel(ox, title, other_v, verdict, colour):
        out.append(text(ox + 200, 36, title, 18, weight="bold", anchor="middle"))
        # own boat goes up, other boat starts right
        own0, own_v = (ox + 120, 360), (0, -42)
        oth0 = (ox + 360, 240)
        for k in range(6):
            t = k
            o = (own0[0] + own_v[0] * t, own0[1] + own_v[1] * t)
            p = (oth0[0] + other_v[0] * t, oth0[1] + other_v[1] * t)
            op = 0.25 + 0.75 * k / 5
            out.append(poly([o, p], colour, 1.5, "5 4", opacity=op))
            out.append(dot(*o, 6, A))
            out.append(dot(*p, 6, B))
            out.append(text(o[0] - 12, o[1] + 4, f"t{k}", 11, MUTED, "end"))
        out.append(text(ox + 200, 400, verdict, 14, colour, "middle", "bold"))

    # collision: other velocity such that both meet at t=6 at (120, 108)
    meet = (120, 360 - 42 * 6)
    panel(0, "Constant bearing → collision", ((meet[0] - 360) / 6, (meet[1] - 240) / 6),
          "Sight lines stay parallel; range shrinks", B)
    out.append(dot(meet[0], meet[1], 10, "none", B))
    out.append(text(meet[0] + 16, meet[1] + 5, "impact", 13, B))
    panel(440, "Drifting bearing → pass", (-45, -35),
          "Sight lines rotate; the boats miss", C)
    out.append(f"<line x1='440' y1='60' x2='440' y2='380' stroke='{GRID}' stroke-width='2'/>")
    legend = [dot(30, 70, 6, A), text(42, 75, "you", 13),
              dot(90, 70, 6, B), text(102, 75, "other vessel", 13)]
    out += legend
    (OUT / "cbdr.svg").write_text(svg(W, H, "\n".join(out)))


# ------------------------------------------------- 2. elastic energy transfer
def elastic():
    W, H = 760, 400
    L, R, T, Bm = 80, 720, 50, 330
    out = [text(W / 2, 30, "Energy transferred in a head-on elastic collision", 18,
                anchor="middle", weight="bold")]
    xs = [10 ** (-2 + 4 * i / 400) for i in range(401)]

    def X(r): return L + (math.log10(r) + 2) / 4 * (R - L)
    def Y(f): return Bm - f * (Bm - T)

    for f in (0, 0.25, 0.5, 0.75, 1):
        out.append(f"<line x1='{L}' x2='{R}' y1='{Y(f):.1f}' y2='{Y(f):.1f}' stroke='{GRID}'/>")
        out.append(text(L - 10, Y(f) + 5, f"{int(f*100)}%", 12, MUTED, "end"))
    for r, lab in ((0.01, "1/100"), (0.1, "1/10"), (1, "1"), (10, "10"), (100, "100")):
        out.append(f"<line y1='{T}' y2='{Bm}' x1='{X(r):.1f}' x2='{X(r):.1f}' stroke='{GRID}'/>")
        out.append(text(X(r), Bm + 22, lab, 12, MUTED, "middle"))
    pts = [(X(r), Y(4 * r / (1 + r) ** 2)) for r in xs]
    out.append(poly(pts, A, 3))
    out.append(dot(X(1), Y(1), 7, A))
    out.append(text(X(2.6), Y(0.9), "m₁ = m₂ → 100% (Newton's cradle)", 13, A))
    out.append(text((L + R) / 2, Bm + 50, "mass ratio m₁ / m₂ (log scale)", 14, INK, "middle"))
    out.append(text(L + 6, T + 18, "4m₁m₂ / (m₁+m₂)²", 14, INK, weight="bold"))
    (OUT / "elastic.svg").write_text(svg(W, H, "\n".join(out)))


# ------------------------------------------------------------- 3. rotation
def rotation():
    W, H = 880, 400
    out = [text(220, 34, "Off-centre hit on a spinning body", 18, anchor="middle", weight="bold")]
    # dumbbell molecule
    cx, cy = 260, 210
    out.append(f"<line x1='{cx}' y1='{cy-80}' x2='{cx}' y2='{cy+80}' stroke='{INK}' stroke-width='6'/>")
    out.append(dot(cx, cy - 80, 26, "#94a3b8", INK))
    out.append(dot(cx, cy + 80, 26, "#94a3b8", INK))
    out.append(dot(cx, cy, 5, INK))
    out.append(text(cx + 12, cy + 18, "centre of mass", 12, MUTED))
    out.append(text(cx + 36, cy + 120, "m₂, I", 14, INK))
    # incoming particle
    hy = cy - 60
    out.append(dot(70, hy, 12, B))
    out.append(text(70, hy - 20, "m₁", 14, B, "middle"))
    out.append(f"<line x1='86' y1='{hy}' x2='{cx-34}' y2='{hy}' stroke='{B}' stroke-width='2.5' marker-end='url(#arr)'/>")
    out.append(text(150, hy + 22, "u₁", 14, B, "middle"))
    # lever arm b
    bx = cx - 60
    out.append(f"<line x1='{bx}' y1='{hy}' x2='{bx}' y2='{cy}' stroke='{C}' stroke-width='2' stroke-dasharray='4 3'/>")
    out.append(f"<line x1='{bx}' y1='{cy}' x2='{cx}' y2='{cy}' stroke='{C}' stroke-width='1' stroke-dasharray='2 3'/>")
    out.append(text(bx - 8, (hy + cy) / 2 + 5, "b", 16, C, "end", "bold"))
    # spin arrow
    out.append(f"<path d='M {cx+70} {cy-40} A 80 80 0 0 1 {cx+70} {cy+40}' fill='none' stroke='{A}' stroke-width='2.5' marker-end='url(#arrb)'/>")
    out.append(text(cx + 100, cy + 5, "ω", 18, A))
    out.append("<defs><marker id='arr' markerWidth='10' markerHeight='10' refX='8' refY='5' orient='auto'>"
               f"<path d='M0,0 L10,5 L0,10 z' fill='{B}'/></marker>"
               "<marker id='arrb' markerWidth='10' markerHeight='10' refX='8' refY='5' orient='auto'>"
               f"<path d='M0,0 L10,5 L0,10 z' fill='{A}'/></marker></defs>")
    out.append(text(220, 370, "M = m₂I / (I + m₂b²)", 18, INK, "middle", "bold"))

    # right panel: M/m2 and rotation share vs b (I = m2 * k^2, k = 1)
    L, R, T, Bm = 500, 850, 60, 320
    out.append(f"<line x1='460' y1='50' x2='460' y2='380' stroke='{GRID}' stroke-width='2'/>")
    out.append(text((L + R) / 2, 34, "Effect of lever arm (I = m₂k²)", 16, anchor="middle", weight="bold"))
    def X(b): return L + b / 3 * (R - L)
    def Y(f): return Bm - f * (Bm - T)
    for f in (0, 0.5, 1):
        out.append(f"<line x1='{L}' x2='{R}' y1='{Y(f):.1f}' y2='{Y(f):.1f}' stroke='{GRID}'/>")
        out.append(text(L - 8, Y(f) + 4, f"{f:g}", 12, MUTED, "end"))
    for b in (0, 1, 2, 3):
        out.append(text(X(b), Bm + 20, f"{b}k", 12, MUTED, "middle"))
    bs = [3 * i / 200 for i in range(201)]
    out.append(poly([(X(b), Y(1 / (1 + b * b))) for b in bs], A, 3))
    out.append(poly([(X(b), Y(b * b / (1 + b * b))) for b in bs], C, 3))
    out.append(text(X(2.2), Y(0.17) - 10, "M / m₂", 14, A, weight="bold"))
    out.append(text(X(2.05), Y(0.83) - 10, "share of energy", 14, C, weight="bold"))
    out.append(text(X(2.05), Y(0.83) + 8, "into rotation", 14, C, weight="bold"))
    out.append(text((L + R) / 2, Bm + 48, "lever arm b", 14, INK, "middle"))
    (OUT / "rotation.svg").write_text(svg(W, H, "\n".join(out)))


# ----------------------------------------------------------- 4. diffusion
def diffusion():
    W, H = 760, 420
    L, R, T, Bm = 70, 730, 50, 370
    N = 400
    out = [text(W / 2, 30, "Random walk: spread grows like √n, not n", 18,
                anchor="middle", weight="bold")]
    ymax = 60
    def X(n): return L + n / N * (R - L)
    def Y(y): return (T + Bm) / 2 - y / ymax * (Bm - T) / 2
    out.append(f"<line x1='{L}' x2='{R}' y1='{Y(0):.1f}' y2='{Y(0):.1f}' stroke='{GRID}'/>")
    for y in (-60, -30, 30, 60):
        out.append(text(L - 8, Y(y) + 4, str(y), 12, MUTED, "end"))
    for n in (0, 100, 200, 300, 400):
        out.append(text(X(n), Bm + 20, str(n), 12, MUTED, "middle"))
    rng = random.Random(7)
    palette = ["#93c5fd", "#a5b4fc", "#c4b5fd", "#99f6e4", "#fcd34d", "#fda4af"]
    for i in range(18):
        y, pts = 0, [(X(0), Y(0))]
        for n in range(1, N + 1):
            y += rng.choice((1, -1))
            pts.append((X(n), Y(max(-ymax, min(ymax, y)))))
        out.append(poly(pts, palette[i % len(palette)], 1.3, opacity=0.9))
    for s in (1, -1):
        out.append(poly([(X(n), Y(s * math.sqrt(n))) for n in range(N + 1)], A, 3))
        out.append(poly([(X(n), Y(s * min(n, ymax))) for n in range(N + 1)], B, 2, "6 5"))
    out.append(text(X(398), Y(-math.sqrt(400)) + 24, "±√n  (diffusive, proved in Lean)", 14, A, "end", "bold"))
    out.append(text(X(66), Y(60) + 18, "±n  (ballistic, no collisions)", 14, B, weight="bold"))
    out.append(text((L + R) / 2, Bm + 44, "steps n", 14, INK, "middle"))
    (OUT / "diffusion.svg").write_text(svg(W, H, "\n".join(out)))


# ------------------------------------------- 5. random walk -> heat kernel
def heat():
    W, H = 760, 400
    L, R, T, Bm = 60, 730, 55, 340
    out = [text(W / 2, 30, "Random-walk probabilities approach the Gaussian heat kernel", 17,
                anchor="middle", weight="bold")]
    xmax, pmax = 30, 0.27
    def X(x): return L + (x + xmax) / (2 * xmax) * (R - L)
    def Y(p): return Bm - p / pmax * (Bm - T)
    out.append(f"<line x1='{L}' x2='{R}' y1='{Bm}' y2='{Bm}' stroke='{MUTED}'/>")
    for x in (-30, -20, -10, 0, 10, 20, 30):
        out.append(text(X(x), Bm + 20, str(x), 12, MUTED, "middle"))
    for n, col in ((10, B), (40, A), (160, C)):
        for k in range(n + 1):
            x = 2 * k - n
            if abs(x) > xmax:
                continue
            p = math.comb(n, k) / 2 ** n
            bw = (R - L) / (2 * xmax) * 1.2
            out.append(f"<rect x='{X(x)-bw/2:.1f}' y='{Y(p):.1f}' width='{bw:.1f}' "
                       f"height='{Bm-Y(p):.1f}' fill='{col}' opacity='0.35'/>")
        pts = [(X(x / 10), Y(2 / math.sqrt(2 * math.pi * n) * math.exp(-(x / 10) ** 2 / (2 * n))))
               for x in range(-10 * xmax, 10 * xmax + 1)]
        out.append(poly(pts, col, 2.5))
        peak = 2 / math.sqrt(2 * math.pi * n)
        out.append(text(X(0) + 14 + n ** 0.5 * 9, Y(peak * 0.75), f"n = {n}", 14, col, weight="bold"))
    out.append(text((L + R) / 2, Bm + 44, "position x", 14, INK, "middle"))
    out.append(text(L + 4, T + 6, "bars: exact path counts / 2ⁿ", 13, MUTED))
    out.append(text(L + 4, T + 24, "curves: exp(−x²/2n) / √n  (heat kernel)", 13, MUTED))
    (OUT / "heat.svg").write_text(svg(W, H, "\n".join(out)))


# ------------------------------------------- 6. heat vs Schrödinger modes
def quantum():
    W, H = 880, 360
    out = []
    def panel(ox, title, sub, col, fn):
        L, R, mid, amp = ox + 40, ox + 400, 200, 90
        out.append(text(ox + 220, 34, title, 17, anchor="middle", weight="bold"))
        out.append(f"<line x1='{L}' x2='{R}' y1='{mid}' y2='{mid}' stroke='{GRID}'/>")
        for j, t in enumerate((0, 0.5, 1.0, 1.5)):
            pts = [(L + i / 200 * (R - L), mid - amp * fn(i / 200 * 4 * math.pi, t)) for i in range(201)]
            out.append(poly(pts, col, 2.5, opacity=1 - 0.24 * j))
            out.append(text(R - 6, 54 + 13 * j, f"t = {t:g}", 12, col, "end"))
        out.append(text(ox + 220, 335, sub, 14, col, "middle", "bold"))
    panel(0, "Heat mode: real D", "amplitude decays: exp(−Dk²t)", B,
          lambda x, t: math.exp(-t) * math.cos(x))
    panel(440, "Schrödinger mode: D = iħ/2m", "amplitude constant: the wave just moves", A,
          lambda x, t: math.cos(x - 1.3 * t))
    out.append(f"<line x1='440' y1='50' x2='440' y2='310' stroke='{GRID}' stroke-width='2'/>")
    (OUT / "quantum.svg").write_text(svg(W, H, "\n".join(out)))


# ------------------------------------------------------------- 7. Bohr
def bohr():
    W, H = 760, 400
    cx, cy = 230, 218
    out = [text(W / 2, 30, "Bohr model: force balance + quantised angular momentum", 17,
                anchor="middle", weight="bold")]
    out.append(dot(cx, cy, 6, B))
    out.append(dot(40, 70, 6, B) + text(52, 75, "proton", 13) + dot(120, 70, 6, A) + text(132, 75, "electron", 13))
    for n in (1, 2, 3):
        r = 17 * n * n
        out.append(f"<circle cx='{cx}' cy='{cy}' r='{r}' fill='none' stroke='{A}' stroke-width='2' opacity='{1.1-0.25*n:.2f}'/>")
        a = -0.6 - 0.35 * n
        out.append(dot(cx + r * math.cos(a), cy + r * math.sin(a), 6, A))
        out.append(text(cx + r * math.cos(0.5) + 6, cy + r * math.sin(0.5) + 4, f"n={n}", 13, A, weight="bold"))
    x0 = 470
    rows = [("n", "radius", "speed", "energy"),
            ("1", "a₀", "c/137", "−13.6 eV"),
            ("2", "4a₀", "c/274", "−3.40 eV"),
            ("3", "9a₀", "c/411", "−1.51 eV")]
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            out.append(text(x0 + (0, 40, 110, 185)[j], 110 + 30 * i, cell, 14,
                            INK if i else MUTED, weight="bold" if i == 0 else "normal"))
    out.append(text(x0, 260, "r ∝ n²  ·  v ∝ 1/n  ·  E ∝ −1/n²", 15, A, weight="bold"))
    out.append(text(x0, 290, "m v²/r = K/r²  and  m v r = nħ", 13, MUTED))
    out.append(text(x0, 345, "A picture, not reality: real electrons", 12, MUTED))
    out.append(text(x0, 362, "are standing waves, not orbiting dots.", 12, MUTED))
    (OUT / "bohr.svg").write_text(svg(W, H, "\n".join(out)))


# ------------------------------------------------------------- 8. spin test
def spin():
    W, H = 880, 400
    out = [text(215, 34, "The model: a helix at total speed c", 17, anchor="middle", weight="bold")]
    # helix drawn as a projected coil
    L, R, mid, amp = 40, 400, 190, 55
    back, front = [], []
    turns = 6
    for i in range(601):
        a = i / 600 * turns * 2 * math.pi
        x = L + 20 + i / 600 * (R - L - 40) + 30 * math.cos(a)
        y = mid - amp * math.sin(a)
        (front if math.cos(a) >= 0 else back).append((i, x, y))
    def runs(pts):
        seg, prev = [], None
        for i, x, y in pts:
            if prev is not None and i != prev + 1:
                yield seg; seg = []
            seg.append((x, y)); prev = i
        if seg: yield seg
    for seg in runs(back):
        out.append(poly(seg, A, 2, opacity=0.3))
    out.append(f"<line x1='{L}' y1='{mid}' x2='{R+20}' y2='{mid}' stroke='{MUTED}' stroke-width='1.5' stroke-dasharray='5 4' marker-end='url(#av)'/>")
    for seg in runs(front):
        out.append(poly(seg, A, 3))
    out.append("<defs><marker id='av' markerWidth='10' markerHeight='10' refX='8' refY='5' orient='auto'>"
               f"<path d='M0,0 L10,5 L0,10 z' fill='{MUTED}'/></marker></defs>")
    out.append(text(R + 22, mid + 20, "v", 15, MUTED, "end", "bold"))
    out.append(text(L + 6, mid - amp - 14, "u (circulation)", 14, A, weight="bold"))
    out.append(text(215, 300, "v² + u² = c²", 18, INK, "middle", "bold"))
    out.append(text(215, 326, "spin ħ/2 at speed c  ⇒  radius ħ/2mc", 14, MUTED, "middle"))
    out.append(text(215, 346, "= 1.9 × 10⁻¹³ m", 14, MUTED, "middle"))
    out.append(f"<line x1='440' y1='50' x2='440' y2='380' stroke='{GRID}' stroke-width='2'/>")
    # right: magnetic moment bars
    out.append(text(660, 34, "The test: magnetic moment", 17, anchor="middle", weight="bold"))
    bx, top, base = 520, 80, 310
    scale = (base - top) / 1.1
    for i, (lab, val, col, note) in enumerate((
            ("model", 0.5, B, "0.500 μB"),
            ("measured", 1.00116, C, "1.00116 μB"))):
        x = bx + i * 150
        hgt = val * scale
        out.append(f"<rect x='{x}' y='{base-hgt:.1f}' width='90' height='{hgt:.1f}' fill='{col}' rx='4'/>")
        out.append(text(x + 45, base + 22, lab, 14, INK, "middle"))
        out.append(text(x + 45, base - hgt - 10, note, 14, col, "middle", "bold"))
    out.append(f"<line x1='{bx-20}' x2='{bx+260}' y1='{base}' y2='{base}' stroke='{MUTED}'/>")
    out.append(text(660, 362, "Any circulating point charge gives g = 1.", 13, MUTED, "middle"))
    out.append(text(660, 380, "The electron has g ≈ 2. Proved: no radius fixes this.", 13, MUTED, "middle"))
    (OUT / "spin.svg").write_text(svg(W, H, "\n".join(out)))

if __name__ == "__main__":
    cbdr(); elastic(); rotation(); diffusion(); heat(); quantum(); bohr(); spin()
    docs = OUT.parent / "docs" / "img"
    docs.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("*.svg"):
        (docs / f.name).write_text(f.read_text())
    print("wrote", sorted(p.name for p in OUT.glob("*.svg")))
