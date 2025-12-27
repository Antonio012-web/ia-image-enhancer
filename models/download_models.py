# models/download_models.py
import os, urllib.request

MODELS = {
    "espcn_x2.pb": "https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x2.pb",
    "espcn_x4.pb": "https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x4.pb",
}

os.makedirs("models", exist_ok=True)

def download(url, dst):
    print(f"⬇ Descargando {os.path.basename(dst)} ...")
    urllib.request.urlretrieve(url, dst)
    if not os.path.exists(dst) or os.path.getsize(dst) == 0:
        raise RuntimeError("Archivo vacío o no creado")
    print(f"✔ Guardado en {dst}")

if __name__ == "__main__":
    for name, url in MODELS.items():
        dst = os.path.join("models", name)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            print(f"✔ Ya existe: {dst}")
            continue
        try:
            download(url, dst)
        except Exception as e:
            print(f"✖ Error descargando {name}: {e}")
            print("👉 Descárgalo manualmente y guárdalo con ese nombre dentro de /models/")
    print("Listo.")
