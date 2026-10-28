"""Interface utilisateur locale du projet ECG PTB-XL.

Deux fenêtres, servies par un petit serveur web local (bibliothèque standard
de Python, aucune dépendance supplémentaire) :
  /utilisation      classifier un ECG comme normal ou pathologique ;
  /fonctionnement   comprendre, avec des animations, comment le réseau
                    transforme cet ECG en décisions.

Lancement, depuis la racine du projet :
  .venv\\Scripts\\python.exe -m interface.app
Le serveur n'écoute que sur la machine locale (127.0.0.1). Ctrl+C l'arrête.
"""

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
import argparse
import base64
import binascii
import json
import re
import sys
import threading
import traceback
import webbrowser

# Permet aussi l'exécution directe du fichier (bouton « Run » de VS Code).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from interface.inference import ProtocoleBinaire  # noqa: E402


STATIC_DIR = Path(__file__).resolve().parent / "static"
PAGES = {"/utilisation": "utilisation.html", "/fonctionnement": "fonctionnement.html"}
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
}
MAX_UPLOAD_BYTES = 4 * 1024 * 1024
LEARNING_CURVES_PATH = (
    Path(__file__).resolve().parents[1]
    / "results" / "binary" / "cross_validation" / "learning_curves.json"
)
TEST_ROC_PATH = Path(__file__).resolve().parents[1] / "results" / "binary" / "test_roc.json"


class Handler(BaseHTTPRequestHandler):
    protocole = None
    cache = {}
    lock = threading.Lock()

    # ------------------------------------------------------------
    # Routage
    # ------------------------------------------------------------

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/":
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", "/utilisation")
            self.end_headers()
        elif path in PAGES:
            self.send_static(PAGES[path])
        elif path.startswith("/static/"):
            self.send_static(path.removeprefix("/static/"))
        elif path == "/api/modele":
            self.answer(lambda: self.cached("modele", self.protocole.informations))
        elif path == "/api/catalogue":
            self.answer(lambda: self.cached("catalogue", self.protocole.catalogue))
        elif path == "/api/distributions":
            self.answer(lambda: self.cached("distributions", self.protocole.distributions))
        elif path == "/api/courbes-apprentissage":
            self.answer(lambda: self.cached(
                "courbes-apprentissage",
                lambda: json.loads(LEARNING_CURVES_PATH.read_text(encoding="utf-8")),
            ))
        elif path == "/api/roc":
            self.answer(lambda: self.cached(
                "roc", lambda: json.loads(TEST_ROC_PATH.read_text(encoding="utf-8")),
            ))
        elif match := re.fullmatch(r"/api/poids/(\d)", path):
            seed = int(match.group(1))
            self.answer(lambda: self.cached(f"poids{seed}", lambda: self.protocole.poids(seed)))
        elif match := re.fullmatch(r"/api/ecg/(\d{1,6})", path):
            self.answer(lambda: self.protocole.analyser_validation(int(match.group(1))))
        else:
            self.send_json({"erreur": "Adresse inconnue."}, HTTPStatus.NOT_FOUND)

    def do_POST(self):
        if urlsplit(self.path).path != "/api/analyser":
            self.send_json({"erreur": "Adresse inconnue."}, HTTPStatus.NOT_FOUND)
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_UPLOAD_BYTES:
            self.send_json({"erreur": "Fichiers trop volumineux pour un ECG de 10 s."},
                           HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            return

        def analyse():
            try:
                body = json.loads(self.rfile.read(length))
                header = base64.b64decode(body["hea"], validate=True)
                dat = base64.b64decode(body["dat"], validate=True)
            except (ValueError, KeyError, TypeError, binascii.Error):
                raise ValueError("Requête invalide : fichiers .hea et .dat attendus.") from None
            return self.protocole.analyser_fichiers(header, dat)

        self.answer(analyse)

    # ------------------------------------------------------------
    # Réponses
    # ------------------------------------------------------------

    def cached(self, key, compute):
        if key not in self.cache:
            self.cache[key] = compute()
        return self.cache[key]

    def answer(self, compute):
        try:
            with self.lock:
                result = compute()
        except PermissionError as error:
            self.send_json({"erreur": str(error)}, HTTPStatus.FORBIDDEN)
        except LookupError as error:
            self.send_json({"erreur": str(error).strip("'\"")}, HTTPStatus.NOT_FOUND)
        except (ValueError, FileNotFoundError) as error:
            self.send_json({"erreur": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception as error:  # noqa: BLE001 : erreur renvoyée à la page
            traceback.print_exc()
            self.send_json({"erreur": f"Erreur interne : {error}"},
                           HTTPStatus.INTERNAL_SERVER_ERROR)
        else:
            self.send_json(result)

    def send_json(self, data, status=HTTPStatus.OK):
        body = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_static(self, name):
        path = (STATIC_DIR / name).resolve()
        if (path.parent != STATIC_DIR or not path.is_file()
                or path.suffix not in CONTENT_TYPES):
            self.send_json({"erreur": "Fichier introuvable."}, HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", CONTENT_TYPES[path.suffix])
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        # Journal limité aux erreurs, pour garder la console lisible.
        if args and str(args[1]).startswith(("4", "5")):
            super().log_message(format, *args)


def main():
    parser = argparse.ArgumentParser(description="Interface utilisateur du projet ECG.")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--sans-navigateur", action="store_true",
                        help="Ne pas ouvrir le navigateur automatiquement.")
    args = parser.parse_args()

    print("Chargement du classifieur binaire normal/pathologique (5 modèles)...")
    Handler.protocole = ProtocoleBinaire()

    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError:
        raise SystemExit(f"Le port {args.port} est déjà utilisé : relancer avec --port 8051.")

    url = f"http://127.0.0.1:{args.port}/utilisation"
    print(f"Interface disponible : {url}")
    print(f"  Fonctionnement du réseau : http://127.0.0.1:{args.port}/fonctionnement")
    print("Ctrl+C pour arrêter.")
    if not args.sans_navigateur:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nInterface arrêtée.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
