import os
import uvicorn
import webview
import threading

from api import app


def start_server():
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")


if __name__ == '__main__':
    t = threading.Thread(target=start_server, daemon=True)
    t.start()

    window = webview.create_window(
        "XTRUNN",
        "http://127.0.0.1:8000",
        width=1200,
        height=800,
        maximized=True,
        background_color='#030508'
    )

    # Par défaut, pywebview tourne en "private_mode" (comme une fenêtre
    # de navigation privée) : localStorage, cookies etc. sont effacés à
    # chaque fermeture du logiciel. On désactive ce mode et on pointe
    # vers un dossier stable (le même que celui utilisé par la base de
    # données dans api.py) pour que tout survive vraiment aux
    # redémarrages.
    dossier_donnees = os.path.join(os.path.expanduser("~"), ".xtrunn", "webview_data")
    os.makedirs(dossier_donnees, exist_ok=True)

    webview.start(private_mode=False, storage_path=dossier_donnees, debug=True)