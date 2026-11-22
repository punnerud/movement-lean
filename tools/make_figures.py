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


if __name__ == "__main__":
    cbdr(); elastic(); rotation(); diffusion()
    print("wrote", sorted(p.name for p in OUT.glob("*.svg")))
