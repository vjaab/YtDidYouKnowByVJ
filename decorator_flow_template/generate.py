import base64
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).parent
FONTS = ROOT / "assets" / "fonts"
ICONS = ROOT / "assets" / "icons"
OUT_HTML = ROOT / "output.html"
OUT_PNG = ROOT / "output.png"

CANVAS_W, CANVAS_H = 1080, 1350  # Instagram 4:5 portrait


def font_data_uri(filename: str) -> str:
    data = (FONTS / filename).read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:font/woff2;base64,{b64}"


def icon_svg(name: str) -> str:
    """Read a lucide icon and strip the license comment / class attr so it
    inherits color via CSS (stroke: var(--accent))."""
    raw = (ICONS / f"{name}.svg").read_text()
    lines = [l for l in raw.splitlines() if not l.strip().startswith("<!--")]
    svg = "\n".join(lines)
    svg = svg.replace('class="lucide lucide-' + name + '"', "")
    return svg


# ---- content: mirrors the JSON schema a real pipeline step would produce ----
STEPS = [
    {
        "accent": "#C879E6",
        "icon": "function-square",
        "title": "Define Decorator Function",
        "desc": "Takes a function as an argument.",
    },
    {
        "accent": "#5FB3F0",
        "icon": "layers",
        "title": "Define Wrapper Function (Inner)",
        "desc": "This function will replace the original function.",
    },
    {
        "accent": "#86D9A0",
        "icon": "zap",
        "title": "Add Logic Before/After Call",
        "desc": "Wrapper executes additional code, then calls original function.",
    },
    {
        "accent": "#E8B85C",
        "icon": "corner-down-right",
        "title": "Return Wrapper Function",
        "desc": "Decorator returns the newly created wrapper.",
    },
    {
        "accent": "#58C7D6",
        "icon": "at-sign",
        "title": "Apply Decorator (@syntax)",
        "desc": "Place @decorator_name above the target function definition.",
    },
    {
        "accent": "#F07178",
        "icon": "repeat-2",
        "title": "Original Function is Replaced",
        "desc": "The decorated function now refers to the wrapper.",
    },
]

for s in STEPS:
    s["icon_svg"] = icon_svg(s["icon"])

context = {
    "canvas_w": CANVAS_W,
    "canvas_h": CANVAS_H,
    "fonts": {
        "sg600": font_data_uri("space-grotesk-latin-600-normal.woff2"),
        "sg700": font_data_uri("space-grotesk-latin-700-normal.woff2"),
        "jb500": font_data_uri("jetbrains-mono-latin-500-normal.woff2"),
        "jb600": font_data_uri("jetbrains-mono-latin-600-normal.woff2"),
        "in400": font_data_uri("inter-latin-400-normal.woff2"),
        "in500": font_data_uri("inter-latin-500-normal.woff2"),
    },
    "filename": "decorators.py",
    "topic": "PYTHON",
    "slide_current": "03",
    "slide_total": "05",
    "eyebrow": "CONCEPT WALKTHROUGH",
    "title": "How Python Decorators Work",
    "subtitle_pre": "",
    "subtitle_bold": "@decorator",
    "subtitle_post": " wraps a function without touching its source",
    "steps": STEPS,
    "arrow_svg": icon_svg("arrow-down"),
    "check_svg": icon_svg("check-circle-2"),
    "closing_line": "decorator chain complete — ready to use",
}

env = Environment(loader=FileSystemLoader(str(ROOT / "templates")))
tpl = env.get_template("decorator_flow.html.j2")
html = tpl.render(**context)
OUT_HTML.write_text(html)
print(f"wrote {OUT_HTML}")

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": CANVAS_W, "height": CANVAS_H}, device_scale_factor=2)
    page.goto(f"file://{OUT_HTML}")
    page.wait_for_timeout(150)
    page.screenshot(path=str(OUT_PNG))
    browser.close()
print(f"wrote {OUT_PNG}")
