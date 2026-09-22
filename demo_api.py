"""
XTRUNN — Backend de la démo web publique.

Sert un unique endpoint /demo/analyser qui réutilise exactement le même
moteur de calcul que l'app de bureau (construire_resultat_analyse, importé
depuis api.py), mais :
  - n'écrit RIEN en base de données : aucune analyse, aucune stratégie,
    aucun historique n'est conservé au-delà de la réponse HTTP elle-même ;
  - applique une limite de débit par IP, pour éviter qu'un usage intensif
    (volontaire ou non) ne fasse grimper les coûts serveur ou ne ralentisse
    le service pour tout le monde.

Déployé séparément de l'app de bureau (qui, elle, tourne en local par
utilisateur via run.py et n'a besoin d'aucune limite). Les deux partagent
le même moteur de calcul en important api.py comme bibliothèque -- jamais
deux implémentations qui pourraient diverger.
"""

import os
import time
import threading
from collections import defaultdict

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from api import (
    construire_resultat_analyse,
    N_WINDOWS_DEFAULT,
    IS_RATIO_DEFAULT,
    PROTOCOLE_STANDARD_DUREE_JOURS,
)
from datetime import datetime, timezone, timedelta

# ============================================================
# LIMITE DE DÉBIT — en mémoire, par IP, fenêtre glissante.
#
# Volontairement simple pour une v1 : pas de Redis, pas d'état partagé
# entre plusieurs processus. Si le service tourne un jour avec plusieurs
# workers, chaque worker a son propre compteur -- la limite réelle devient
# alors (nombre de workers × RATE_LIMIT_MAX), pas un problème pour un
# lancement à faible trafic, mais à revoir avant une vraie montée en charge.
# ============================================================
RATE_LIMIT_MAX = 8            # analyses maximum...
RATE_LIMIT_WINDOW_S = 3600    # ...par heure glissante, par IP

_appels_par_ip: dict[str, list[float]] = defaultdict(list)
_verrou = threading.Lock()


def _ip_autorisee(ip: str) -> bool:
    maintenant = time.time()
    with _verrou:
        appels = _appels_par_ip[ip]
        appels[:] = [t for t in appels if maintenant - t < RATE_LIMIT_WINDOW_S]
        if len(appels) >= RATE_LIMIT_MAX:
            return False
        appels.append(maintenant)
        return True


app = FastAPI(title="XTRUNN — Démo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def racine():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "essayer.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/protocole-standard-dates")
async def obtenir_dates_protocole_standard():
    """Identique à l'endpoint de l'app de bureau : sans état, recalculé à
    chaque appel par rapport à aujourd'hui, jamais stocké."""
    aujourdhui = datetime.now(timezone.utc).date()
    date_debut = aujourdhui - timedelta(days=PROTOCOLE_STANDARD_DUREE_JOURS)
    return {
        "date_debut": date_debut.isoformat(),
        "date_fin": aujourdhui.isoformat(),
        "duree_jours": PROTOCOLE_STANDARD_DUREE_JOURS,
    }


@app.post("/demo/analyser")
async def demo_analyser(
    request: Request,
    trades_file: UploadFile = File(...),
    bot_name: str = Form(...),
    plateforme: str = Form("mt5"),
    capital_initial: float = Form(None),
    protocole: str = Form("personnalise"),
    n_windows: int = Form(N_WINDOWS_DEFAULT),
    is_ratio: float = Form(IS_RATIO_DEFAULT),
):
    ip = request.client.host if request.client else "inconnue"
    if not _ip_autorisee(ip):
        raise HTTPException(
            status_code=429,
            detail=(
                f"Limite de la démo atteinte ({RATE_LIMIT_MAX} analyses par heure maximum). "
                "Réessayez plus tard, ou téléchargez l'application de bureau pour un usage sans limite."
            ),
        )

    contents = await trades_file.read()

    # n_trials_testes n'est pas exposé dans la démo -- c'est un champ
    # déclaratif avancé (biais de sélection multiple), pas essentiel pour
    # un premier essai rapide sans inscription.
    resultat = construire_resultat_analyse(
        contents, trades_file.filename, bot_name, plateforme, capital_initial,
        protocole, n_windows, is_ratio, n_trials_testes=None,
    )

    # Ni id, ni strategie_id : rien n'est persisté, donc rien à identifier
    # au-delà de cette unique réponse.
    if "erreur" not in resultat:
        resultat["id"] = None
        resultat["strategie_id"] = None

    return resultat


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
