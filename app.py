# app.py  — IA Image Enhancer + Vectorización (modo Pro) + preview inline
import hashlib, json
import os, uuid, sys, subprocess, json
from flask import Flask, send_from_directory, request, jsonify
from PIL import Image
import numpy as np
import cv2


# --- Config ---
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
MODELS_FOLDER = 'models'
BIN_FOLDER = os.path.join(os.getcwd(), 'bin')  # p.ej. C:\Users\...\ia-image-enhancer\bin
MAX_CONTENT_LENGTH = 25 * 1024 * 1024
ALLOWED_EXT = {'.png', '.jpg', '.jpeg', '.webp'}
VT_BIN = os.path.join(BIN_FOLDER, 'vtracer.exe')  # ajusta si usas Linux/Mac

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app = Flask(__name__, static_url_path='/', static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

print(f"[BOOT] Python: {sys.executable}")
try:
    import cairosvg as _cairosvg_probe
    print(f"[BOOT] cairosvg import OK (v{_cairosvg_probe.__version__})")
except Exception as e:
    print(f"[BOOT] cairosvg import FAILED: {e}")

# --- Utilidades PIL/CV ---
def pil_to_cv(img_pil):
    arr = np.array(img_pil.convert('RGB'))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def cv_to_pil(img_cv):
    rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def clip8(x): return np.clip(x, 0, 255).astype(np.uint8)

# --- Filtros base ---
def sharpen(img, strength=1.0):
    if strength <= 0: return img
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
    return cv2.addWeighted(img, 1 + float(strength), blur, -float(strength), 0)

def denoise(img, level=0):
    if level <= 0: return img
    d = 5 + int((level / 100) * 7)
    sigmaColor = 25 + int((level / 100) * 100)
    sigmaSpace = 25 + int((level / 100) * 100)
    return cv2.bilateralFilter(img, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)


# --- Super-res OpenCV dnn_superres ---
SR_ENGINES = {}
def load_sr(scale: int):
    try:
        from cv2 import dnn_superres
    except Exception:
        print("[SR] dnn_superres no disponible, usaré bicúbico.")
        return None
    key = f"espcn_x{scale}"
    if key in SR_ENGINES: return SR_ENGINES[key]
    model_path = os.path.join(MODELS_FOLDER, f'espcn_x{scale}.pb')
    if not os.path.exists(model_path):
        print(f"[SR] Modelo no encontrado: {model_path}, usaré bicúbico.")
        return None
    sr = dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel('espcn', scale)
    print(f"[SR] ESPCN x{scale} cargado.")
    SR_ENGINES[key] = sr
    return sr

# --- Avanzados ---
def apply_clahe_bgr(img_bgr, clip=2.0, tile=8):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=max(1.0, float(clip)), tileGridSize=(max(4, int(tile)),)*2)
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

def guided_denoise(img_bgr, radius=5, eps=1e-6):
    try:
        gf = cv2.ximgproc.guidedFilter
        I = img_bgr.astype(np.float32) / 255.0
        p = I
        r = max(2, int(radius))
        out = np.zeros_like(I)
        for c in range(3):
            out[:,:,c] = gf(guide=I[:,:,c], src=p[:,:,c], radius=r, eps=float(eps))
        return clip8(out * 255.0)
    except Exception:
        return cv2.bilateralFilter(img_bgr, d=7, sigmaColor=25, sigmaSpace=25)

def laplacian_boost(img_bgr, amount=0.3):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, ddepth=cv2.CV_16S, ksize=3)
    lap = cv2.convertScaleAbs(lap)
    lap_bgr = cv2.cvtColor(lap, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(img_bgr, 1.0, lap_bgr, float(amount), 0)

def resize_lanczos(img_bgr, scale=1.0):
    if scale == 1.0: return img_bgr
    h, w = img_bgr.shape[:2]
    return cv2.resize(img_bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LANCZOS4)

# --- Deconvolución Wiener ---
def gaussian_psf(size, sigma):
    k = cv2.getGaussianKernel(size, sigma)
    psf = k @ k.T
    psf /= psf.sum()
    return psf

def wiener_deblur_channel(ch, psf, K=0.01):
    psf_pad = np.zeros_like(ch)
    kh, kw = psf.shape
    psf_pad[:kh, :kw] = psf
    psf_pad = np.roll(psf_pad, -kh//2, axis=0)
    psf_pad = np.roll(psf_pad, -kw//2, axis=1)
    Ch = np.fft.fft2(ch)
    Ph = np.fft.fft2(psf_pad)
    Xh = (np.conj(Ph) / (np.abs(Ph)**2 + K)) * Ch
    out = np.real(np.fft.ifft2(Xh))
    return np.clip(out, 0.0, 1.0)

def wiener_deblur(img_bgr, radius=1.5, lam=0.01):
    size = int(max(3, round(radius*6)//2*2+1))
    sigma = float(radius)
    psf = gaussian_psf(size, sigma)
    f = img_bgr.astype(np.float32) / 255.0
    chans = cv2.split(f)
    outs = [wiener_deblur_channel(c, psf, K=float(lam)) for c in chans]
    out = cv2.merge(outs)
    return clip8(out * 255.0)

# --- Nitidez multiescala ---
def multiscale_sharpen(img_bgr, amount=0.6):
    if amount <= 0: return img_bgr
    f = img_bgr.astype(np.float32)
    b1 = cv2.GaussianBlur(f, (0,0), 1.0)
    b2 = cv2.GaussianBlur(f, (0,0), 2.0)
    b3 = cv2.GaussianBlur(f, (0,0), 4.0)
    d1 = f - b1; d2 = b1 - b2; d3 = b2 - b3
    out = f + (1.20*amount)*d1 + (0.75*amount)*d2 + (0.35*amount)*d3
    return clip8(out)

# --- Anti-halo / papel ---
def local_range_gray(gray):
    k = np.ones((3,3), np.uint8)
    maxf = cv2.dilate(gray, k)
    minf = cv2.erode(gray, k)
    rng = maxf.astype(np.int16) - minf.astype(np.int16)
    return rng.astype(np.float32)

def edge_clip_sharpen(img_bgr, amount=0.6, clip_rel=0.015, sigma=1.0):
    f = img_bgr.astype(np.float32)
    base = cv2.GaussianBlur(f, (0,0), sigma)
    hp = f - base
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    rng = local_range_gray(gray) + 1e-6
    clip_abs = (clip_rel * rng * 255.0).astype(np.float32)
    hp = np.clip(hp, -clip_abs[...,None], clip_abs[...,None])
    out = f + amount * hp
    return clip8(out)

def paper_smooth(img_bgr, strength=0.3):
    s = float(max(0.0, min(1.0, strength)))
    if s <= 0: return img_bgr
    sm = cv2.edgePreservingFilter(img_bgr, flags=1, sigma_s=40, sigma_r=0.1)
    return cv2.addWeighted(img_bgr, 1.0 - 0.5*s, sm, 0.5*s, 0)

def nlm_colored(img_bgr, strength=10):
    s = float(max(0, min(30, strength)))
    hY = 8.0 * (s / 30.0); hC = 6.0 * (s / 30.0)
    return cv2.fastNlMeansDenoisingColored(img_bgr, None, hY, hC, 7, 21)

def nlm_chroma_only(img_bgr, strength=12):
    s = float(max(0, min(30, strength)))
    hC = 10.0 * (s / 30.0)
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    cr = cv2.fastNlMeansDenoising(cr, None, hC, 7, 21)
    cb = cv2.fastNlMeansDenoising(cb, None, hC, 7, 21)
    out = cv2.merge([y, cr, cb])
    return cv2.cvtColor(out, cv2.COLOR_YCrCb2BGR)

def anti_ringing(img_bgr, amount=0.25):
    a = float(max(0.0, min(1.0, amount)))
    if a <= 0.0: return img_bgr
    return cv2.edgePreservingFilter(img_bgr, flags=1, sigma_s=20, sigma_r=0.25 + 0.2*a)

# --- Pipelines de mejora (se mantienen) ---
def enhance_image(pil_img, scale=1, denoise_level=0, sharpen_level=1.0,
                  deblur=False, deblur_radius=1.5, deblur_lambda=0.01,
                  ms_amount=0.4, nlm_strength=0, nlm_chroma=False, antiring=0.0,
                  edgeclip=False, edgeclip_amt=0.6, edgeclip_clip=0.015, paper_flat=0.0):
    img = pil_to_cv(pil_img)
    if scale in (2, 4):
        sr = load_sr(scale)
        if sr is not None:
            try: img = sr.upsample(img); print(f"[SR] IA x{scale}")
            except Exception as e: print(f"[SR] fallo {e}, bicúbico"); h, w = img.shape[:2]; img = cv2.resize(img,(w*scale,h*scale),cv2.INTER_CUBIC)
        else:
            h, w = img.shape[:2]; img = cv2.resize(img,(w*scale,h*scale),cv2.INTER_CUBIC)
    if deblur: img = wiener_deblur(img, radius=deblur_radius, lam=deblur_lambda)
    if nlm_strength > 0: img = nlm_colored(img, nlm_strength)
    if nlm_chroma:       img = nlm_chroma_only(img, max(8, nlm_strength))
    img = denoise(img, denoise_level)
    img = sharpen(img, sharpen_level)
    if ms_amount > 0: img = multiscale_sharpen(img, amount=ms_amount)
    if edgeclip:  img = edge_clip_sharpen(img, amount=edgeclip_amt, clip_rel=edgeclip_clip, sigma=1.0)
    if paper_flat > 0: img = paper_smooth(img, strength=paper_flat)
    if antiring > 0:   img = anti_ringing(img, amount=antiring)
    return cv_to_pil(img)

def enhance_text_mode(pil_img, scale=2, clahe_clip=2.5, clahe_tile=8,
                      guided_radius=6, guided_eps=1e-6,
                      sharpen_level=0.6, lap_amount=0.25,
                      final_downscale=1.0,
                      deblur=False, deblur_radius=1.5, deblur_lambda=0.01,
                      ms_amount=0.5, nlm_strength=0, nlm_chroma=False, antiring=0.0,
                      edgeclip=False, edgeclip_amt=0.6, edgeclip_clip=0.015, paper_flat=0.0):
    img = pil_to_cv(pil_img)
    if scale in (2, 4):
        sr = load_sr(scale)
        if sr is not None:
            try: img = sr.upsample(img); print(f"[TEXT] IA x{scale}")
            except Exception as e: print(f"[TEXT] SR fallo {e}, bicúbico"); h,w=img.shape[:2]; img=cv2.resize(img,(w*scale,h*scale),cv2.INTER_CUBIC)
        else:
            h,w=img.shape[:2]; img=cv2.resize(img,(w*scale,h*scale),cv2.INTER_CUBIC)
    if deblur: img = wiener_deblur(img, radius=deblur_radius, lam=deblur_lambda)
    img = apply_clahe_bgr(img, clip=clahe_clip, tile=clahe_tile)
    img = guided_denoise(img, radius=guided_radius, eps=guided_eps)
    if nlm_strength > 0: img = nlm_colored(img, nlm_strength)
    if nlm_chroma:       img = nlm_chroma_only(img, max(8, nlm_strength))
    if sharpen_level > 0: img = sharpen(img, strength=float(sharpen_level))
    if lap_amount > 0:    img = laplacian_boost(img, amount=float(lap_amount))
    if ms_amount > 0:     img = multiscale_sharpen(img, amount=ms_amount)
    if edgeclip:  img = edge_clip_sharpen(img, amount=edgeclip_amt, clip_rel=edgeclip_clip, sigma=1.0)
    if final_downscale and final_downscale != 1.0: img = resize_lanczos(img, scale=float(final_downscale))
    if paper_flat > 0: img = paper_smooth(img, strength=paper_flat)
    if antiring > 0:   img = anti_ringing(img, amount=antiring)
    return cv_to_pil(img)

# --- Microtexto (se mantiene) ---
def build_text_mask(img_bgr):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (0,0), 1.0)
    edges = cv2.Canny(g, 60, 140)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    return edges

def enhance_microtext_mode(pil_img, scale=4,
                           clahe_clip=3.2, clahe_tile=8,
                           guided_radius=7, guided_eps=1e-6,
                           usm_strength=0.8, lap_amount=0.28,
                           final_downscale=0.5,
                           deblur=True, deblur_radius=1.6, deblur_lambda=0.01,
                           ms_amount=0.6, nlm_strength=12, nlm_chroma=True, antiring=0.25,
                           edgeclip=True, edgeclip_amt=0.55, edgeclip_clip=0.015, paper_flat=0.25):
    img = pil_to_cv(pil_img)
    if scale in (2, 4):
        sr = load_sr(scale)
        if sr is not None:
            try: img = sr.upsample(img); print(f"[MICRO] IA x{scale}")
            except Exception as e: print(f"[MICRO] SR fallo {e}, bicúbico"); h,w=img.shape[:2]; img=cv2.resize(img,(w*scale,h*scale),cv2.INTER_CUBIC)
        else:
            h,w=img.shape[:2]; img=cv2.resize(img,(w*scale,h*scale),cv2.INTER_CUBIC)
    if deblur: img = wiener_deblur(img, radius=deblur_radius, lam=deblur_lambda)
    img = apply_clahe_bgr(img, clip=clahe_clip, tile=clahe_tile)
    img = guided_denoise(img, radius=guided_radius, eps=guided_eps)
    if nlm_strength > 0: img = nlm_colored(img, nlm_strength)
    if nlm_chroma:       img = nlm_chroma_only(img, max(8, nlm_strength))
    mask = build_text_mask(img); mask_f = (mask.astype(np.float32)/255.0)[..., None]
    sharp_usm = sharpen(img, strength=float(usm_strength))
    sharp_lap = laplacian_boost(sharp_usm, amount=float(lap_amount))
    mix = (img.astype(np.float32) * (1.0 - mask_f) + sharp_lap.astype(np.float32) * mask_f).astype(np.uint8)
    if ms_amount > 0: mix = multiscale_sharpen(mix, amount=ms_amount)
    if edgeclip:  mix = edge_clip_sharpen(mix, amount=edgeclip_amt, clip_rel=edgeclip_clip, sigma=1.0)
    if final_downscale and final_downscale != 1.0: mix = resize_lanczos(mix, scale=float(final_downscale))
    if paper_flat > 0: mix = paper_smooth(mix, strength=paper_flat)
    if antiring > 0:   mix = anti_ringing(mix, amount=antiring)
    return cv_to_pil(mix)

# =========================
# ===  VECTORIZACIÓN    ===
# =========================
def _contours_to_path(contour):
    pts = contour.reshape(-1, 2)
    if len(pts) == 0: return ""
    d = [f"M {pts[0,0]} {pts[0,1]}"]
    for p in pts[1:]: d.append(f"L {p[0]} {p[1]}")
    d.append("Z")
    return " ".join(d)

def _write_svg(width, height, paths, fills, outfile):
    from xml.sax.saxutils import escape
    with open(outfile, 'w', encoding='utf-8') as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n')
        f.write('<g fill-rule="evenodd">\n')
        for d, fill in zip(paths, fills):
            f.write(f'  <path d="{escape(d)}" fill="{escape(fill)}" stroke="none"/>\n')
        f.write('</g>\n</svg>\n')

def vectorize_mono(img_bgr, svg_smooth=2.0, min_area=25):
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    bw = cv2.medianBlur(bw, 3)
    contours, _ = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    paths, fills = [], []
    eps = float(svg_smooth)
    for cnt in contours:
        area = abs(cv2.contourArea(cnt))
        if area < float(min_area): continue
        approx = cv2.approxPolyDP(cnt, eps, True)
        d = _contours_to_path(approx)
        if not d: continue
        fills.append("#000000"); paths.append(d)
    return w, h, paths, fills

def vectorize_color(img_bgr, k=6, svg_smooth=2.0, min_area=25):
    h, w = img_bgr.shape[:2]
    Z = img_bgr.reshape((-1,3)).astype(np.float32)
    k = int(max(2, min(16, k)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _ret, labels, centers = cv2.kmeans(Z, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    centers = centers.astype(np.uint8); labels = labels.flatten()
    paths, fills = [], []; eps = float(svg_smooth); min_area = float(min_area)
    for i in range(k):
        color = tuple(int(c) for c in centers[i])  # BGR
        mask = (labels == i).astype(np.uint8).reshape((h, w)) * 255
        mask = cv2.medianBlur(mask, 3)
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        conts, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if not conts: continue
        hex_fill = "#{:02x}{:02x}{:02x}".format(color[2], color[1], color[0])
        for cnt in conts:
            area = abs(cv2.contourArea(cnt))
            if area < min_area: continue
            approx = cv2.approxPolyDP(cnt, eps, True)
            d = _contours_to_path(approx)
            if not d: continue
            paths.append(d); fills.append(hex_fill)
    return w, h, paths, fills

# --- Guardado de SVG/PDF desde paths ---
def save_svg_or_pdf(width, height, paths, fills, base_name, out_format='svg'):
    svg_name = f"{base_name}.svg"
    svg_path = os.path.join(RESULTS_FOLDER, svg_name)
    _write_svg(width, height, paths, fills, svg_path)
    if out_format == 'pdf':
        e1 = None
        try:
            import cairosvg
            pdf_name = f"{base_name}.pdf"
            pdf_path = os.path.join(RESULTS_FOLDER, pdf_name)
            cairosvg.svg2pdf(url=svg_path, write_to=pdf_path)
            return pdf_name, 'application/pdf', None
        except Exception as _err1:
            e1 = _err1; print(f"[vectorize] cairosvg no usable: {e1}")
        try:
            from svglib.svglib import svg2rlg
            from reportlab.graphics import renderPDF
            drawing = svg2rlg(svg_path)
            pdf_name = f"{base_name}.pdf"
            pdf_path = os.path.join(RESULTS_FOLDER, pdf_name)
            renderPDF.drawToFile(drawing, pdf_path)
            return pdf_name, 'application/pdf', None
        except Exception as e2:
            print(f"[vectorize] svglib/reportlab PDF FAILED: {e2}")
            return None, None, ("PDF_FAILED", f"cairosvg_err={e1}; svglib_err={e2}")
    return svg_name, 'image/svg+xml', None

# --- Guardado de PDF/SVG desde un SVG existente (VTracer) ---
def save_svg_or_pdf_from_file(svg_path, base_name, out_format='svg'):
    if out_format == 'pdf':
        e1 = None
        try:
            import cairosvg
            pdf_name = f"{base_name}.pdf"
            pdf_path = os.path.join(RESULTS_FOLDER, pdf_name)
            cairosvg.svg2pdf(url=svg_path, write_to=pdf_path)
            return pdf_name, 'application/pdf', None
        except Exception as _err1:
            e1 = _err1; print(f"[vtracer] cairosvg no usable: {e1}")
        try:
            from svglib.svglib import svg2rlg
            from reportlab.graphics import renderPDF
            drawing = svg2rlg(svg_path)
            pdf_name = f"{base_name}.pdf"
            pdf_path = os.path.join(RESULTS_FOLDER, pdf_name)
            renderPDF.drawToFile(drawing, pdf_path)
            return pdf_name, 'application/pdf', None
        except Exception as e2:
            print(f"[vtracer] svglib/reportlab PDF FAILED: {e2}")
            return None, None, ("PDF_FAILED", f"cairosvg_err={e1}; svglib_err={e2}")
    # devolver el SVG original dentro de results (mueve/copia si está fuera)
    if os.path.dirname(svg_path) != os.path.abspath(RESULTS_FOLDER):
        new_svg = os.path.join(RESULTS_FOLDER, f"{base_name}.svg")
        try: os.replace(svg_path, new_svg)
        except Exception:
            import shutil; shutil.copyfile(svg_path, new_svg)
        svg_path = new_svg
    name = os.path.basename(svg_path)
    return name, 'image/svg+xml', None

# --- Preprocesado Pro para vectorizar ---
def preprocess_for_vectorize(pil_img,
                             upsample=2,               # 1 o 2
                             smooth_strength=1.0,      # 0..2
                             posterize_colors=None,    # None o int
                             edge_boost=0.6,           # 0..1.2
                             for_mono=False):
    img = pil_to_cv(pil_img)

    # Upscale
    if upsample in (2, 4):
        sr = load_sr(upsample)
        if sr is not None:
            try: img = sr.upsample(img)
            except Exception:
                h, w = img.shape[:2]
                img = cv2.resize(img, (w*upsample, h*upsample), interpolation=cv2.INTER_LANCZOS4)
        else:
            h, w = img.shape[:2]
            img = cv2.resize(img, (w*upsample, h*upsample), interpolation=cv2.INTER_LANCZOS4)

    # Aplanado edge-aware
    if smooth_strength > 0:
        try: img = cv2.pyrMeanShiftFiltering(img, sp=8, sr=16, maxLevel=1)
        except Exception: pass
        d = 7; sc = int(40 * smooth_strength); ss = int(40 * smooth_strength)
        img = cv2.bilateralFilter(img, d=d, sigmaColor=max(10, sc), sigmaSpace=max(10, ss))

    # Posterize (solo color)
    if posterize_colors and posterize_colors > 1 and not for_mono:
        h, w = img.shape[:2]
        Z = img.reshape((-1,3)).astype(np.float32)
        k = int(max(2, min(16, posterize_colors)))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _ret, labels, centers = cv2.kmeans(Z, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        centers = centers.astype(np.uint8)
        img = centers[labels.flatten()].reshape((h, w, 3))

    # Edge boost
    if edge_boost > 0:
        img = edge_clip_sharpen(img, amount=float(edge_boost), clip_rel=0.015, sigma=1.0)

    # Mono
    if for_mono:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = cv2.equalizeHist(g)
        bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 6)
        bw = cv2.medianBlur(bw, 3)
        kernel = np.ones((3,3), np.uint8)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
        img = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)

    return cv_to_pil(img)

# --- Frontend ---
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/favicon.ico')
def favicon():
    return ('', 204)

# --- Enhance API ---
@app.route('/api/enhance', methods=['POST'])
def api_enhance():
    if 'image' not in request.files: return jsonify({"error": "Falta archivo 'image'"}), 400
    f = request.files['image']
    ext = os.path.splitext(f.filename.lower())[-1]
    if ext not in ALLOWED_EXT and f.mimetype != 'image/png':
        return jsonify({"error": f"Extensión no permitida: {ext}"}), 400

    try:
        scale = int(request.form.get('scale', 1))
        denoise_level = int(request.form.get('denoise', 0))
        sharpen_level = float(request.form.get('sharpen', 1.0))
    except Exception:
        return jsonify({"error": "Parámetros inválidos"}), 400
    out_format = request.form.get('format', 'png').lower()

    text_mode = request.form.get('text_mode', '0') in ('1', 'true', 'True')
    microtext_mode = request.form.get('microtext_mode', '0') in ('1', 'true', 'True')

    clahe_clip = float(request.form.get('clahe_clip', 2.5))
    clahe_tile = int(request.form.get('clahe_tile', 8))
    guided_radius = int(request.form.get('guided_radius', 6))
    guided_eps = float(request.form.get('guided_eps', 1e-6))
    text_sharpen = float(request.form.get('text_sharpen', 0.6))
    lap_amount = float(request.form.get('lap_amount', 0.25))
    final_downscale = float(request.form.get('final_downscale', 1.0))

    deblur = request.form.get('deblur', '0') in ('1', 'true', 'True')
    deblur_radius = float(request.form.get('deblur_radius', 1.5))
    deblur_lambda = float(request.form.get('deblur_lambda', 0.01))
    ms_amount = float(request.form.get('ms_amount', 0.5))

    nlm_strength = int(request.form.get('nlm', 0))
    nlm_chroma   = request.form.get('nlm_chroma', '0') in ('1','true','True')
    antiring     = float(request.form.get('antiring', 0.0))
    edgeclip     = request.form.get('edgeclip', '0') in ('1','true','True')
    edgeclip_amt = float(request.form.get('edgeclip_amt', 0.6))
    edgeclip_clip= float(request.form.get('edgeclip_clip', 0.015))
    paper_flat   = float(request.form.get('paper_smooth', 0.0))

    uid = uuid.uuid4().hex
    up_path = os.path.join(UPLOAD_FOLDER, f"{uid}{ext if ext in ALLOWED_EXT else '.png'}")
    f.save(up_path)
    pil_in = Image.open(up_path).convert('RGB')

    if microtext_mode:
        pil_out = enhance_microtext_mode(
            pil_in,
            scale=scale if scale in (1, 2, 4) else 4,
            clahe_clip=clahe_clip, clahe_tile=clahe_tile,
            guided_radius=guided_radius, guided_eps=guided_eps,
            usm_strength=text_sharpen, lap_amount=lap_amount,
            final_downscale=final_downscale if final_downscale > 0 else 0.5,
            deblur=deblur, deblur_radius=deblur_radius, deblur_lambda=deblur_lambda,
            ms_amount=ms_amount, nlm_strength=nlm_strength, nlm_chroma=nlm_chroma, antiring=antiring,
            edgeclip=edgeclip, edgeclip_amt=edgeclip_amt, edgeclip_clip=edgeclip_clip, paper_flat=paper_flat
        )
    elif text_mode:
        pil_out = enhance_text_mode(
            pil_in,
            scale=scale if scale in (1, 2, 4) else 2,
            clahe_clip=clahe_clip, clahe_tile=clahe_tile,
            guided_radius=guided_radius, guided_eps=guided_eps,
            sharpen_level=text_sharpen, lap_amount=lap_amount,
            final_downscale=final_downscale,
            deblur=deblur, deblur_radius=deblur_radius, deblur_lambda=deblur_lambda,
            ms_amount=ms_amount, nlm_strength=nlm_strength, nlm_chroma=nlm_chroma, antiring=antiring,
            edgeclip=edgeclip, edgeclip_amt=edgeclip_amt, edgeclip_clip=edgeclip_clip, paper_flat=paper_flat
        )
    else:
        pil_out = enhance_image(
            pil_in,
            scale=scale,
            denoise_level=denoise_level,
            sharpen_level=sharpen_level,
            deblur=deblur, deblur_radius=deblur_radius, deblur_lambda=deblur_lambda,
            ms_amount=ms_amount, nlm_strength=nlm_strength, nlm_chroma=nlm_chroma, antiring=antiring,
            edgeclip=edgeclip, edgeclip_amt=edgeclip_amt, edgeclip_clip=edgeclip_clip, paper_flat=paper_flat
        )

    out_name = f"enh_{uid}.{out_format}"
    out_path = os.path.join(RESULTS_FOLDER, out_name)
    if out_format == 'jpg':
        pil_out.save(out_path, format='JPEG', quality=95); mime = 'image/jpeg'
    elif out_format == 'webp':
        pil_out.save(out_path, format='WEBP', quality=95, method=6); mime = 'image/webp'
    elif out_format == 'pdf':
        pil_out.save(out_path, format='PDF', resolution=300.0); mime = 'application/pdf'
    else:
        pil_out.save(out_path, format='PNG'); mime = 'image/png'

    # IMPORTANTE: usar /view para previsualizar, y /download solo para guardar
    return jsonify({
        "url": f"/view/{out_name}",          # compat, pero ya es una URL de vista
        "view_url": f"/view/{out_name}",
        "download_url": f"/download/{out_name}",
        "filename": out_name,
        "mime": mime
    })

# --- VTracer wrapper compatible (solo flags soportados) ---
def call_vtracer(input_path: str, out_format: str = 'svg', color_mode: str = 'color', mode: str = 'spline'):
    if not os.path.exists(VT_BIN):
        return None, None, ("VTRACER_NOT_FOUND", f"No se encontró VTracer en: {VT_BIN}")
    color_mode = (color_mode or 'color').lower()
    if color_mode not in ('color', 'gray', 'binary'): color_mode = 'color'
    uid = uuid.uuid4().hex
    svg_out = os.path.join(RESULTS_FOLDER, f"vtr_{uid}.svg")
    args = [VT_BIN, '--input', input_path, '--output', svg_out, '--mode', mode, '--colormode', color_mode]
    try:
        proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if proc.returncode != 0:
            err_snip = (proc.stderr or proc.stdout).decode(errors='ignore')[:600]
            return None, None, ("VTRACER_FAILED", f"VTracer error: {err_snip}")
        if not os.path.exists(svg_out):
            return None, None, ("VTRACER_FAILED", "VTracer terminó sin crear el SVG.")
        base = os.path.splitext(os.path.basename(svg_out))[0]
        name, mime, err = save_svg_or_pdf_from_file(svg_out, base, out_format)
        return name, mime, err
    except Exception as e:
        return None, None, ("VTRACER_EXCEPTION", str(e))

# --- API Vectorize (con Modo Pro) ---
@app.route('/api/vectorize', methods=['POST'])
def api_vectorize():
    if 'image' not in request.files: return jsonify({"error": "Falta archivo 'image'"}), 400
    f = request.files['image']
    ext = os.path.splitext(f.filename.lower())[-1]
    if ext not in ALLOWED_EXT and f.mimetype != 'image/png':
        return jsonify({"error": f"Extensión no permitida: {ext}"}), 400

    mode = (request.form.get('mode', 'mono') or 'mono').lower()  # mono | color
    colors = int(request.form.get('colors', 6))
    svg_smooth = float(request.form.get('svg_smooth', 2.0))
    min_area = float(request.form.get('min_area', 25.0))
    out_format = (request.form.get('format', 'svg') or 'svg').lower()  # svg | pdf

    engine = (request.form.get('engine', 'auto') or 'auto').lower()  # auto|vtracer|local
    if engine == 'auto':
        engine = 'vtracer' if os.path.exists(VT_BIN) else 'local'

    uid = uuid.uuid4().hex
    up_path = os.path.join(UPLOAD_FOLDER, f"{uid}{ext if ext in ALLOWED_EXT else '.png'}")
    f.save(up_path)
    pil_in = Image.open(up_path).convert('RGB')

    # --- Modo Pro defaults ---
    pro = request.form.get('pro', '1') in ('1','true','True')
    upsample = int(request.form.get('pro_upsample', 2))
    smooth_strength = float(request.form.get('pro_smooth', 1.0))
    edge_boost_amt = float(request.form.get('pro_edge', 0.6))
    posterize = colors if mode == 'color' else None

    if pro:
        pil_pre = preprocess_for_vectorize(
            pil_in,
            upsample=2 if upsample not in (1,2) else upsample,
            smooth_strength=max(0.0, min(2.0, smooth_strength)),
            posterize_colors=posterize,
            edge_boost=max(0.0, min(1.2, edge_boost_amt)),
            for_mono=(mode != 'color')
        )
        src_path = os.path.join(UPLOAD_FOLDER, f"{uid}_pro.png")
        pil_pre.save(src_path, 'PNG')
    else:
        src_path = up_path

    if engine == 'vtracer':
        color_mode = 'color' if mode == 'color' else ('binary' if mode == 'mono' else 'color')
        name, mime, err = call_vtracer(
            input_path=src_path,
            out_format=out_format,
            color_mode=color_mode,
            mode='spline'
        )
        if err: return jsonify({"error": "VTracer failed", "detail": err}), 500
        return jsonify({
            "ok": True,
            "url": f"/view/{name}",
            "view_url": f"/view/{name}",
            "download_url": f"/download/{name}",
            "filename": name,
            "mime": mime,
            "requested_format": out_format,
            "engine": "vtracer"
        })

    # --- Motor local (paths)
    img = pil_to_cv(Image.open(src_path).convert('RGB'))
    if mode == 'color':
        w, h, paths, fills = vectorize_color(img, k=colors, svg_smooth=svg_smooth, min_area=min_area)
    else:
        w, h, paths, fills = vectorize_mono(img, svg_smooth=svg_smooth, min_area=min_area)

    if not paths:
        return jsonify({"error": "No se detectaron regiones para vectorizar. Ajusta parámetros."}), 422

    base = f"vec_{uid}"
    name, mime, err = save_svg_or_pdf(w, h, paths, fills, base, out_format)
    if isinstance(err, tuple) and err and err[0] == "PDF_FAILED":
        return jsonify({"error": "No se pudo generar el PDF desde el SVG.", "detail": err[1]}), 500

    return jsonify({
        "ok": True,
        "url": f"/view/{name}",
        "view_url": f"/view/{name}",
        "download_url": f"/download/{name}",
        "filename": name,
        "mime": mime,
        "requested_format": out_format,
        "engine": "local",
        "stats": {"paths": len(paths), "width": w, "height": h, "mode": mode, "colors": colors if mode=='color' else 1}
    })

# --- Descarga y Vista inline ---
@app.route('/download/<path:fname>')
def download_file(fname):
    return send_from_directory(RESULTS_FOLDER, fname, as_attachment=True)

@app.route('/view/<path:fname>')
def view_file(fname):
    return send_from_directory(RESULTS_FOLDER, fname, as_attachment=False)

# --- Diagnóstico PDF ---
@app.route('/api/check_pdf_support')
def api_check_pdf_support():
    info = {"python_executable": sys.executable, "cairosvg_import": None, "cairosvg_version": None, "error": None}
    try:
        import cairosvg
        info["cairosvg_import"] = True
        info["cairosvg_version"] = getattr(cairosvg, "__version__", "unknown")
    except Exception as e:
        info["cairosvg_import"] = False
        info["error"] = str(e)
    return jsonify(info)

# --- JSON errors para /api/* ---
@app.errorhandler(Exception)
def json_errors(e):
    if request.path.startswith('/api/'):
        import traceback
        return jsonify({"error": f"{type(e).__name__}: {str(e)}",
                        "trace": traceback.format_exc()}), 500
    raise e

@app.after_request
def add_headers(resp):
    resp.headers['Cache-Control'] = 'no-store'
    return resp

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
