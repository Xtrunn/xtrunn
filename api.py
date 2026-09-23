import io
import os
import sys
import json
import re
import math
import sqlite3
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, FileResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# CONFIGURATION — Toute constante "magique" vit ici, à un seul
# endroit, pour que la méthodologie du score soit auditable.
# ============================================================

RANDOM_SEED = 42                 # Reproductibilité : même fichier -> même score, toujours.
N_SIMULATIONS_MC = 1000          # Simulations Monte Carlo (permutation OOS)
N_SIMULATIONS_STRESS = 1000      # Simulations du stress test de fragilité
STRESS_DROP_RATIO = 0.10         # % de trades retirés lors du stress de fragilité
COST_PCT_PER_TRADE = 0.0015      # Coût simulé par trade = 0.15% du capital (spread/slippage/commissions)
N_ROLLING_WINDOWS = 5            # Nombre de fenêtres pour l'analyse temporelle

# Seuil minimal de trades OOS en dessous duquel le score est considéré
# comme statistiquement peu fiable (échantillon trop petit).
MIN_TRADES_RELIABLE = 30
MIN_TRADES_ABSOLUTE = 3  # en dessous, on refuse carrément l'analyse
EPREUVE_JOURS_MIN_REGULARITE = 3  # en dessous, un seul jour représente forcément 100% du profit -- pas encore un vrai signal

# Protocole Standard -- un chemin recommandé où le logiciel dicte
# précisément la période à backtester (dates exactes recalculées à
# chaque visite), plutôt que de laisser deviner. Aucune exigence sur le
# nombre de trades : une stratégie à faible fréquence reste acceptée,
# juste avec une fourchette d'incertitude honnêtement plus large -- seule
# la durée est bloquante, parce que le temps ne s'accélère pour
# personne, alors que le nombre de trades ne dépend pas que de
# l'utilisateur.
PROTOCOLE_STANDARD_DUREE_JOURS = 365          # durée exacte demandée à l'affichage, et plancher de tolérance à la vérification
PROTOCOLE_STANDARD_ANCIENNETE_MAX_JOURS = 365 * 3  # le tout premier trade ne doit pas remonter à plus loin que ça
SEUIL_SCORE_VALIDATION = 70  # score_p10 (le pire cas raisonnable du bootstrap, pas le score brut) requis
                              # pour débloquer l'Épreuve de Validation -- doit rester dans la zone "positive"
                              # même dans un scénario pessimiste de rééchantillonnage

# Pondération des 4 piliers (total = 100)
POIDS_MONTE_CARLO = 30
POIDS_STABILITE_ISOOS = 30
SEUIL_FENETRES_FAIBLES_PCT = 40    # % de fenêtres faibles (<0.5) à partir duquel la cohérence inter-fenêtres est pénalisée
SEUIL_STABILITE_POUR_PENALITE = 0.5  # la pénalité ne s'applique que si la médiane elle-même reste correcte (sinon déjà pénalisée ailleurs)
MULT_PENALITE_INCOHERENCE = 0.7
POIDS_RISQUE_DRAWDOWN = 20
POIDS_CONCENTRATION = 20

# Pilier 1 (Monte Carlo) — seuils du ratio de sensibilité du drawdown à
# l'ordre des trades (pire_mdd_95 simulé / drawdown réel observé).
# <= RATIO_PATH_BON : le drawdown réel est représentatif, pas de "chance" d'ordre -> score plein
# >= RATIO_PATH_MAUVAIS : le drawdown réel a été très favorisé par son enchaînement -> score nul
RATIO_PATH_BON = 1.5
RATIO_PATH_MAUVAIS = 4.0
# Facteur de récupération (profit net / pire drawdown simulé à 95%) jugé "confortable"
RECOVERY_FACTOR_CIBLE = 3.0

# Pilier 2 — plafond de qualité absolue basé sur le Profit Factor OOS.
# Une stratégie parfaitement stable IS->OOS mais à peine rentable en absolu
# (PF proche de 1) ne doit pas obtenir le score plein du pilier stabilité.
# Paliers progressifs (pas de couperet brutal) : (seuil_pf_min, multiplicateur)
PALIERS_QUALITE_ABSOLUE = [
    (1.30, 1.00),
    (1.15, 0.85),
    (1.05, 0.60),
    (1.00, 0.35),
]

# Fourchette de confiance du score global (bootstrap avec remise sur l'OOS)
N_CI_REPLICATES = 150     # nombre de répliques bootstrap pour la fourchette
N_CI_PERMS_INNER = 40     # permutations Monte Carlo internes par réplique (allégé pour la perf)

# ============================================================
# WALK-FORWARD MULTI-FENÊTRES
#
# XTRUNN ne demande plus deux fichiers IS/OOS pré-découpés à la main par
# l'utilisateur : un seul historique chronologique complet est importé, et
# le logiciel le découpe lui-même en plusieurs fenêtres IS/OOS glissantes
# consécutives. Deux bénéfices méthodologiques directs :
#   1. L'utilisateur ne choisit plus lui-même où couper — il ne peut donc
#      plus, même involontairement, avantager la coupure.
#   2. Un split IS/OOS unique est un échantillon de taille 1 au niveau des
#      régimes de marché : rien ne dit si cet OOS a été "chanceux". Avec
#      plusieurs fenêtres indépendantes, on mesure une vraie distribution
#      de stabilité plutôt qu'un point unique.
#
# LIMITE HONNÊTE : XTRUNN ne reçoit qu'un journal de trades déjà exécutés,
# pas le code de la stratégie — il ne peut donc pas RÉ-OPTIMISER les
# paramètres par fenêtre comme le ferait un vrai moteur de walk-forward
# optimization. Le rôle "IS" de chaque fenêtre est une période de
# référence pour mesurer la stabilité dans le temps, pas une ré-
# optimisation formelle. C'est une amélioration réelle par rapport à un
# split unique, pas un remplacement d'un vrai pipeline de walk-forward.
# ============================================================

N_WINDOWS_DEFAULT = 5
N_WINDOWS_MIN = 3
N_WINDOWS_MAX = 12
IS_RATIO_DEFAULT = 0.70
IS_RATIO_MIN = 0.5
IS_RATIO_MAX = 0.9
MIN_TRADES_PER_WINDOW = 12  # trades totaux (IS+OOS) minimum par fenêtre pour qu'elle soit exploitable

# ============================================================
# BIAIS DE SÉLECTION MULTIPLE
#
# Même un résultat OOS "jamais vu par l'optimisation" reste biaisé si de
# nombreuses variantes de paramètres ont été essayées avant d'arriver à
# celle-ci : on a implicitement sélectionné celle qui a le mieux marché
# sur cette période précise, parmi plusieurs candidates. C'est le sujet
# des travaux de Bailey & López de Prado sur la "Probability of Backtest
# Overfitting" et le "Deflated Sharpe Ratio".
#
# CE SIGNAL EST PUREMENT INFORMATIF, IL N'AJUSTE PLUS LE SCORE.
# Contrairement au risque de ruine (calculé objectivement à partir des
# trades eux-mêmes), le nombre de variantes testées n'est PAS détectable
# dans les données : deux séquences de trades strictement identiques
# peuvent provenir d'une stratégie testée une seule fois, ou de la
# meilleure de 500 variantes essayées au hasard — rien dans les trades
# ne distingue les deux cas, la sélection a eu lieu AVANT que ces trades
# n'existent. Comme cette donnée est auto-déclarée et invérifiable, en
# faire un multiplicateur de score punirait uniquement les utilisateurs
# honnêtes (ceux qui déclarent un chiffre élevé) sans aucun moyen de
# dissuader une sous-déclaration — un vice de conception, pas un détail.
# Le rôle de XTRUNN se limite donc à expliquer clairement le risque,
# pas à prétendre le corriger numériquement.
# ============================================================

N_TRIALS_WARNING_THRESHOLD = 10
N_TRIALS_CRITICAL_THRESHOLD = 30


def message_biais_selection(n_trials):
    """Message contextuel, purement informatif — n'affecte jamais le score."""
    if n_trials is None or n_trials <= 1:
        return None
    return (
        f"Avec {n_trials} variantes testées avant de garder celle-ci : même un système sans aucun edge réel a "
        f"statistiquement de bonnes chances de produire, par pur hasard, au moins un résultat parmi {n_trials} qui "
        f"ressemble à celui affiché ici. Plus ce nombre est élevé, plus il faut se méfier d'un score qui semble "
        f"\"trop beau\" — mais XTRUNN ne peut pas objectivement corriger cela : cette information ne laisse aucune "
        f"trace dans les trades eux-mêmes, il faut vous faire confiance sur le chiffre déclaré."
    )

# ============================================================
# PERSISTANCE — Historique des analyses (SQLite local)
#
# La base vit dans le dossier utilisateur (~/.xtrunn/), PAS à côté de
# l'exécutable : quand le logiciel sera packagé (PyInstaller), le dossier
# de l'exécutable peut être en lecture seule et surtout est remplacé à
# chaque mise à jour — l'historique serait perdu. Le dossier utilisateur
# survit aux mises à jour et désinstallations classiques.
# ============================================================

APPDATA_DIR = os.path.join(os.path.expanduser("~"), ".xtrunn")
os.makedirs(APPDATA_DIR, exist_ok=True)
DB_PATH = os.path.join(APPDATA_DIR, "xtrunn.db")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bot_name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            score_global INTEGER NOT NULL,
            score_median REAL,
            score_p10 REAL,
            score_p90 REAL,
            oos_profit_net REAL,
            oos_profit_factor REAL,
            n_trades_oos INTEGER,
            echantillon_fiable INTEGER,
            fiabilite_evaluation TEXT,
            raisons_fiabilite TEXT,
            resultat_json TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_bot_name ON analyses(bot_name)")

    # forward_tests et l'ancien modèle d'épreuves (simple comparaison à
    # une analyse précise) sont retirés au profit d'un vrai simulateur de
    # épreuve autonome, avec ses propres règles personnalisables — plus
    # rien à voir avec l'ancien système, donc migration ponctuelle plutôt
    # qu'une évolution incrémentale. Ne s'exécute qu'une seule fois : si
    # "épreuves" existe déjà avec le NOUVEAU schéma (colonne
    # objectif_profit_pct présente), on ne touche à rien.
    conn.execute("DROP TABLE IF EXISTS forward_tests")

    # Migration : les tables s'appelaient "challenges"/"challenge_trades"
    # avant que le mode ne devienne "Épreuve" -- on les renomme plutôt que
    # de les recréer vides, pour ne perdre aucune donnée existante.
    tables_existantes = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    if 'challenges' in tables_existantes and 'epreuves' not in tables_existantes:
        conn.execute("ALTER TABLE challenges RENAME TO epreuves")
        try:
            conn.execute("ALTER TABLE epreuves RENAME COLUMN type_challenge TO type_epreuve")
        except sqlite3.OperationalError:
            pass  # colonne déjà nommée type_epreuve, ou n'existait pas sous l'ancien nom
    if 'challenge_trades' in tables_existantes and 'epreuve_trades' not in tables_existantes:
        conn.execute("ALTER TABLE challenge_trades RENAME TO epreuve_trades")
        try:
            conn.execute("ALTER TABLE epreuve_trades RENAME COLUMN challenge_id TO epreuve_id")
        except sqlite3.OperationalError:
            pass

    cols_epreuves = [r[1] for r in conn.execute("PRAGMA table_info(epreuves)").fetchall()]
    if cols_epreuves and 'objectif_profit_pct' not in cols_epreuves:
        conn.execute("DROP TABLE epreuves")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS epreuves (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nom TEXT NOT NULL,
            strategie_id INTEGER,
            analyse_id INTEGER,
            capital_initial REAL NOT NULL,
            objectif_profit_pct REAL,
            drawdown_max_pct REAL,
            perte_quotidienne_max_pct REAL,
            consistency_max_pct REAL,
            jours_min_trading INTEGER,
            duree_jours INTEGER NOT NULL,
            type_epreuve TEXT NOT NULL DEFAULT 'personnalise',
            date_debut TEXT NOT NULL,
            date_fin TEXT NOT NULL,
            statut TEXT NOT NULL DEFAULT 'en_cours',
            revele INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_epreuve_strategie ON epreuves(strategie_id)")
    # Migration : "revele" n'existait pas dans les toutes premières
    # versions de la table -- ajoutée ici pour ne pas perdre les épreuves
    # déjà enregistrées. Les épreuves déjà résolues avant cette migration
    # sont considérées comme déjà "révélées" (pas de mise en scène
    # rétroactive sur un résultat que l'utilisateur a déjà vu).
    cols_epreuves_apres = [r[1] for r in conn.execute("PRAGMA table_info(epreuves)").fetchall()]
    if 'revele' not in cols_epreuves_apres:
        conn.execute("ALTER TABLE epreuves ADD COLUMN revele INTEGER NOT NULL DEFAULT 0")
        conn.execute("UPDATE epreuves SET revele = 1 WHERE statut != 'en_cours'")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS epreuve_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            epreuve_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            profit REAL NOT NULL,
            ticket TEXT,
            symbole TEXT,
            notes TEXT,
            source TEXT NOT NULL DEFAULT 'manuel',
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_epreuve_trades_epreuve ON epreuve_trades(epreuve_id)")
    # Migration : symbole/notes n'existaient pas dans les toutes premières
    # versions de la table -- ajoutées ici pour ne pas perdre les trades
    # déjà enregistrés par les utilisateurs existants.
    cols_epreuve_trades = [r[1] for r in conn.execute("PRAGMA table_info(epreuve_trades)").fetchall()]
    if 'symbole' not in cols_epreuve_trades:
        conn.execute("ALTER TABLE epreuve_trades ADD COLUMN symbole TEXT")
    if 'notes' not in cols_epreuve_trades:
        conn.execute("ALTER TABLE epreuve_trades ADD COLUMN notes TEXT")

    # ============================================================
    # STRATÉGIES — une vraie identité persistante, distincte des analyses.
    # Avant, "bot_name" n'était qu'un texte libre répété sur chaque ligne
    # d'analyse, sans lien réel entre elles : revenir sur une stratégie
    # après avoir quitté XTRUNN obligeait à réimporter et retaper le même
    # nom. Une stratégie regroupe maintenant toutes ses analyses (chaque
    # nouveau backtest importé = une nouvelle version de la même
    # stratégie) et toutes ses épreuves, pour qu'un simple clic sur son
    # nom à l'accueil ramène directement là où l'utilisateur en était.
    # ============================================================
    conn.execute("""
        CREATE TABLE IF NOT EXISTS strategies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nom TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL
        )
    """)
    try:
        conn.execute("ALTER TABLE analyses ADD COLUMN strategie_id INTEGER")
    except sqlite3.OperationalError:
        pass  # colonne déjà présente, migration déjà appliquée précédemment
    conn.execute("CREATE INDEX IF NOT EXISTS idx_analyses_strategie ON analyses(strategie_id)")

    # Nom du fichier de trades importé -- seul repère fiable pour
    # distinguer entre elles plusieurs analyses d'une même stratégie
    # (des backtests différents), puisque le score et le nom ne suffisent
    # pas à eux seuls à savoir de quel fichier chaque résultat vient.
    try:
        conn.execute("ALTER TABLE analyses ADD COLUMN nom_fichier TEXT")
    except sqlite3.OperationalError:
        pass

    # Capital de départ et plateforme -- affichés sur la carte de
    # stratégie à l'accueil, à côté du nom, pour donner le contexte
    # d'une analyse sans avoir à l'ouvrir.
    try:
        conn.execute("ALTER TABLE analyses ADD COLUMN capital_initial REAL")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE analyses ADD COLUMN plateforme TEXT")
    except sqlite3.OperationalError:
        pass

    # Rétro-remplissage du capital pour les analyses enregistrées avant
    # l'ajout de sa colonne dédiée -- contrairement au nom de fichier et
    # à la plateforme (réellement nouveaux), le capital existait déjà
    # dans le JSON stocké depuis le tout début, pas de raison de le
    # perdre pour l'historique existant.
    a_backfill = conn.execute("SELECT id, resultat_json FROM analyses WHERE capital_initial IS NULL").fetchall()
    for row in a_backfill:
        try:
            capital = json.loads(row["resultat_json"]).get("capital_initial")
            if capital is not None:
                conn.execute("UPDATE analyses SET capital_initial = ? WHERE id = ?", (capital, row["id"]))
        except (json.JSONDecodeError, TypeError):
            pass

    # Code source de la stratégie -- collé volontairement par
    # l'utilisateur à la création, pour retrouver plus tard exactement
    # quel code a produit quel résultat. Jamais généré ni deviné, purement
    # ce que l'utilisateur choisit de coller.
    try:
        conn.execute("ALTER TABLE analyses ADD COLUMN code_source TEXT")
    except sqlite3.OperationalError:
        pass

    # Conservation du fichier brut importé -- même si l'utilisateur
    # supprime le fichier de son propre ordinateur à force d'en
    # accumuler, il reste toujours réimportable depuis ici.
    try:
        conn.execute("ALTER TABLE analyses ADD COLUMN fichier_original BLOB")
    except sqlite3.OperationalError:
        pass

    # Est-ce que la période importée a servi au réglage des paramètres de
    # la stratégie ? Un choix obligatoire à la saisie (pas de valeur par
    # défaut côté formulaire), qui ne change jamais le calcul du score
    # lui-même -- seulement le discours qui l'accompagne, puisqu'un bon
    # score sur une période déjà connue ne garantit pas ce qu'un bon
    # score sur une période neuve garantit. NULL pour les analyses
    # enregistrées avant ce changement, dont on ne peut plus savoir.
    try:
        conn.execute("ALTER TABLE analyses ADD COLUMN periode_deja_reglee INTEGER")
    except sqlite3.OperationalError:
        pass

    # Fiabilité de l'évaluation : stockée séparément du score.
    try:
        conn.execute("ALTER TABLE analyses ADD COLUMN fiabilite_evaluation TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE analyses ADD COLUMN raisons_fiabilite TEXT")
    except sqlite3.OperationalError:
        pass

    # Migration des analyses déjà enregistrées avant ce changement : une
    # stratégie créée par nom distinct (bot_name), analyses rattachées
    # rétroactivement. Idempotent — ne retouche que ce qui n'est pas
    # encore lié (strategie_id IS NULL), donc sans risque à chaque
    # démarrage.
    noms_a_migrer = conn.execute(
        "SELECT DISTINCT bot_name FROM analyses WHERE strategie_id IS NULL"
    ).fetchall()
    for row in noms_a_migrer:
        nom = row["bot_name"]
        existante = conn.execute("SELECT id FROM strategies WHERE nom = ?", (nom,)).fetchone()
        if existante:
            strategie_id = existante["id"]
        else:
            cur = conn.execute(
                "INSERT INTO strategies (nom, created_at) VALUES (?, ?)",
                (nom, datetime.now(timezone.utc).isoformat())
            )
            strategie_id = cur.lastrowid
        conn.execute(
            "UPDATE analyses SET strategie_id = ? WHERE bot_name = ? AND strategie_id IS NULL",
            (strategie_id, nom)
        )

    # Nettoyage des épreuves orphelines créées avant que la suppression en
    # cascade n'existe (analyse déjà supprimée entre-temps, épreuve restée
    # accroché à un analyse_id qui n'existe plus) — se manifestait par un
    # épreuve affichée comme "Stratégie" générique à l'accueil, impossible
    # à ouvrir ni à supprimer par les moyens normaux. Idempotent, sans
    # risque à chaque démarrage : ne supprime que ce qui n'a plus de
    # référence valide.
    conn.execute("DELETE FROM epreuves WHERE analyse_id NOT IN (SELECT id FROM analyses)")

    conn.commit()
    conn.close()


init_db()


def resoudre_dossier_ressources():
    """
    Détermine où trouver les fichiers embarqués (index.html) selon le
    mode d'exécution :
      - Script Python normal (dev)          -> dossier du script
      - PyInstaller --onedir                -> dossier de l'exécutable
      - PyInstaller --onefile                -> sys._MEIPASS (dossier
        temporaire d'extraction, différent du dossier de l'exécutable)
    Se tromper ici fait planter l'app au lancement une fois packagée,
    silencieusement en --onefile puisque le dossier existe mais est vide.
    """
    if hasattr(sys, "_MEIPASS"):
        return sys._MEIPASS
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


@app.get("/", response_class=HTMLResponse)
async def lire_interface():
    html_path = os.path.join(resoudre_dossier_ressources(), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


# ============================================================
# UTILITAIRES
# ============================================================

def calculer_max_drawdown(trades):
    cumul = np.cumsum(trades)
    max_cumul = np.maximum.accumulate(cumul)
    drawdown = max_cumul - cumul
    return float(np.max(drawdown)) if len(drawdown) > 0 else 0.0


def get_rng():
    """RNG isolé et seedé : reproductible, et n'affecte pas np.random global."""
    return np.random.default_rng(RANDOM_SEED)


PF_SENTINELLE_AUCUNE_PERTE = 0.1  # valeur de repli volontairement basse (pas l'infini) quand gains>0 et pertes=0 -- signale "suspect", pas "parfait"

def calculer_profit_factor(trades):
    gains = trades[trades > 0].sum()
    pertes = abs(trades[trades < 0].sum())
    if pertes != 0:
        return float(gains / pertes)
    return float(PF_SENTINELLE_AUCUNE_PERTE if gains > 0 else 0.0)


def qualite_absolue_multiplicateur(oos_pf):
    """
    Plafond de qualité absolue (Pilier 2) : une stratégie peut être parfaitement
    stable entre IS et OOS (ratio d'efficacité proche de 1) tout en étant à
    peine rentable dans l'absolu. Sans ce plafond, un Profit Factor OOS de
    1.02 obtiendrait le même score qu'un PF de 2.0 du moment que la stabilité
    est bonne — ce qui n'a pas de sens pour un utilisateur qui doit décider
    de passer en compte réel. Paliers progressifs plutôt qu'un couperet net.
    """
    # PF_SENTINELLE_AUCUNE_PERTE représente le cas particulier « aucun trade
    # perdant ». Il ne doit surtout pas être traité comme un PF réel de 0.1 :
    # sinon une stratégie sans perte voit artificiellement son pilier de
    # stabilité multiplié par 0, alors que le cas doit simplement être
    # évalué par les autres épreuves (concentration, coûts, trajectoire, etc.).
    if oos_pf == PF_SENTINELLE_AUCUNE_PERTE:
        return 1.0

    for seuil, mult in PALIERS_QUALITE_ABSOLUE:
        if oos_pf >= seuil:
            return mult
    return 0.0


# ============================================================
# PILIER 1 : STRESS MONTE CARLO — SENSIBILITÉ DE TRAJECTOIRE
# (permutation aléatoire des trades OOS)
#
# NOTE MÉTHODOLOGIQUE IMPORTANTE : permuter une liste ne change JAMAIS sa
# somme. Le "profit final" est donc rigoureusement identique sur les 1000
# simulations — mesurer une "probabilité de profit positif" là-dessus est
# trivial (0% ou 100% par construction, aucune information). Ce qui varie
# réellement quand on change l'ordre des trades, c'est le CHEMIN parcouru
# par l'équity, donc le drawdown. Ce pilier note donc la stratégie sur deux
# questions bien réelles :
#   (A) Sensibilité du drawdown réel à l'ordre des trades : le drawdown
#       observé dans le backtest a-t-il eu la "chance" d'un enchaînement
#       favorable, ou est-il représentatif de ce que 1000 réordonnancements
#       plausibles produisent ?
#   (B) Facteur de récupération sous stress : le profit net généré
#       justifie-t-il le pire drawdown parmi ces réordonnancements
#       (percentile 95) ?
# ============================================================

def executer_monte_carlo(trades, rng, num_simulations=N_SIMULATIONS_MC, segments=None):
    """Monte Carlo d'ordre, sans raccorder artificiellement les fenêtres OOS.

    Sans `segments` : comportement historique, tous les trades de test
    mélangés en un seul bloc avant réordonnancement (voir _pooled).

    Avec `segments` (les listes de trades OOS de chaque fenêtre
    walk-forward) : chaque fenêtre est réordonnée SÉPARÉMENT -- on évite
    de créer une continuité artificielle entre deux fenêtres qui sont en
    réalité séparées par une période de référence exclue. Le ratio de
    sensibilité et le facteur de récupération sont calculés fenêtre par
    fenêtre puis on retient le PIRE cas, plutôt que de comparer un 95e
    percentile poolé (toutes fenêtres mélangées, donc dominé par les
    fenêtres les plus fournies en trades) à une médiane par fenêtre (où
    chaque fenêtre pèse pareil peu importe sa taille) : ces deux agrégats
    ne mesurent pas la même chose, les comparer peut diluer un vrai
    problème localisé sur une seule fenêtre.
    """
    if segments is None:
        return _executer_monte_carlo_pooled(trades, rng, num_simulations)

    fenetres_valides = [np.asarray(seg, dtype=float) for seg in segments if len(seg) >= MIN_TRADES_ABSOLUTE]
    if not fenetres_valides:
        return _executer_monte_carlo_pooled(trades, rng, num_simulations)

    ratios_par_fenetre = []
    recovery_par_fenetre = []
    pire_mdd_95_par_fenetre = []
    for seg in fenetres_valides:
        dd_reel_fenetre = calculer_max_drawdown(seg)
        net_fenetre = float(np.sum(seg))
        simules = np.empty(num_simulations)
        for i in range(num_simulations):
            shuffled = rng.permutation(seg)
            equity = np.cumsum(shuffled)
            max_eq = np.maximum.accumulate(equity)
            dd = max_eq - equity
            simules[i] = np.max(dd) if len(dd) > 0 else 0.0
        p95_fenetre = float(np.percentile(simules, 95))
        pire_mdd_95_par_fenetre.append(p95_fenetre)

        if dd_reel_fenetre > 0:
            ratios_par_fenetre.append(p95_fenetre / dd_reel_fenetre)
        elif net_fenetre > 0:
            ratios_par_fenetre.append(p95_fenetre / net_fenetre)
        # Si ni drawdown ni profit dans cette fenêtre (cas très rare), on
        # l'ignore plutôt que d'injecter un ratio indéfini.

        # Récupération elle aussi comparée fenêtre par fenêtre (pire cas
        # retenu) -- sinon on mélangerait un profit poolé (toutes fenêtres)
        # à un drawdown pire-cas par fenêtre, exactement l'incohérence
        # repérée dans l'approche par médiane de la V2.
        if p95_fenetre > 0:
            recovery_par_fenetre.append(net_fenetre / p95_fenetre)
        elif net_fenetre > 0:
            recovery_par_fenetre.append(float('inf'))
        else:
            recovery_par_fenetre.append(0.0)

    pire_mdd_95 = max(pire_mdd_95_par_fenetre) if pire_mdd_95_par_fenetre else 0.0
    ratio_path = max(ratios_par_fenetre) if ratios_par_fenetre else float('inf')
    recovery = min(recovery_par_fenetre) if recovery_par_fenetre else 0.0

    poids_a = POIDS_MONTE_CARLO // 2
    poids_b = POIDS_MONTE_CARLO - poids_a

    if ratio_path <= RATIO_PATH_BON:
        score_path = poids_a
    elif ratio_path >= RATIO_PATH_MAUVAIS:
        score_path = 0
    else:
        score_path = int(round(poids_a * (1 - (ratio_path - RATIO_PATH_BON) / (RATIO_PATH_MAUVAIS - RATIO_PATH_BON))))

    if recovery >= RECOVERY_FACTOR_CIBLE:
        score_recovery = poids_b
    elif recovery <= 0:
        score_recovery = 0
    else:
        score_recovery = int(round(min(poids_b, max(0, (recovery / RECOVERY_FACTOR_CIBLE) * poids_b))))

    return {
        "pire_mdd_95": round(pire_mdd_95, 2),
        "path_sensitivity_ratio": round(ratio_path, 2) if ratio_path != float('inf') else None,
        "stress_recovery_factor": round(recovery, 2) if recovery != float('inf') else None,
        "score": int(score_path + score_recovery),
        "score_path": int(score_path),
        "score_recovery": int(score_recovery),
        "mc_scope": "oos_segments",
        "mc_segments_testes": len(fenetres_valides),
    }


def _executer_monte_carlo_pooled(trades, rng, num_simulations=N_SIMULATIONS_MC):
    if len(trades) < MIN_TRADES_ABSOLUTE:
        return {
            "pire_mdd_95": 0.0,
            "path_sensitivity_ratio": None,
            "stress_recovery_factor": None,
            "score": 0,
            "score_path": 0,
            "score_recovery": 0,
        }

    max_drawdowns = np.empty(num_simulations)
    for i in range(num_simulations):
        shuffled = rng.permutation(trades)
        equity = np.cumsum(shuffled)
        max_eq = np.maximum.accumulate(equity)
        dd = max_eq - equity
        max_drawdowns[i] = np.max(dd) if len(dd) > 0 else 0

    pire_mdd_95 = float(np.percentile(max_drawdowns, 95))
    oos_net = float(np.sum(trades))
    oos_mdd_reel = calculer_max_drawdown(trades)

    poids_a = POIDS_MONTE_CARLO // 2
    poids_b = POIDS_MONTE_CARLO - poids_a

    # (A) Sensibilité du drawdown réel à l'ordre des trades
    if oos_mdd_reel > 0:
        ratio_path = pire_mdd_95 / oos_mdd_reel
    else:
        ratio_path = (pire_mdd_95 / oos_net) if oos_net > 0 else float('inf')

    if ratio_path <= RATIO_PATH_BON:
        score_path = poids_a
    elif ratio_path >= RATIO_PATH_MAUVAIS:
        score_path = 0
    else:
        score_path = int(round(poids_a * (1 - (ratio_path - RATIO_PATH_BON) / (RATIO_PATH_MAUVAIS - RATIO_PATH_BON))))

    # (B) Facteur de récupération sous le pire drawdown simulé (95e percentile)
    if pire_mdd_95 > 0:
        recovery = oos_net / pire_mdd_95
    else:
        recovery = float('inf') if oos_net > 0 else 0.0

    if recovery >= RECOVERY_FACTOR_CIBLE:
        score_recovery = poids_b
    elif recovery <= 0:
        score_recovery = 0
    else:
        score_recovery = int(round(min(poids_b, max(0, (recovery / RECOVERY_FACTOR_CIBLE) * poids_b))))

    return {
        "pire_mdd_95": round(pire_mdd_95, 2),
        "path_sensitivity_ratio": round(ratio_path, 2) if ratio_path != float('inf') else None,
        "stress_recovery_factor": round(recovery, 2) if recovery != float('inf') else None,
        "score": int(score_path + score_recovery),
        "score_path": int(score_path),
        "score_recovery": int(score_recovery),
    }


# ============================================================
# STRESS PRO — CHANTIER 1 : Test de fragilité (subsampling, sans remise)
#
# NOTE MÉTHODOLOGIQUE : ce test retire aléatoirement STRESS_DROP_RATIO
# des trades SANS remise (chaque trade apparaît 0 ou 1 fois par tirage).
# Ce n'est pas un bootstrap classique (qui tire AVEC remise à taille
# égale) — c'est un test de "que se passe-t-il si certains trades
# n'avaient pas eu lieu". On le nomme donc "fragility test" et pas
# "bootstrap" pour rester rigoureux.
# ============================================================

def stress_test_fragilite(trades_array, rng, n_iterations=N_SIMULATIONS_STRESS, drop_ratio=STRESS_DROP_RATIO):
    if len(trades_array) == 0:
        return {
            "bootstrap_prob_positif": 0.0,
            "bootstrap_mediane_profit": 0.0,
            "bootstrap_pire_profit_5pct": 0.0,
            "bootstrap_pire_mdd": 0.0
        }

    n_trades = len(trades_array)
    sample_size = max(1, int(n_trades * (1 - drop_ratio)))

    final_profits = np.empty(n_iterations)
    max_drawdowns = np.empty(n_iterations)

    for i in range(n_iterations):
        idx = rng.choice(n_trades, size=sample_size, replace=False)
        sampled_trades = trades_array[idx]
        cumulative = np.cumsum(sampled_trades)
        final_profits[i] = cumulative[-1] if len(cumulative) > 0 else 0

        peak = np.maximum.accumulate(cumulative) if len(cumulative) > 0 else np.array([0])
        drawdown = peak - cumulative
        max_drawdowns[i] = np.max(drawdown) if len(drawdown) > 0 else 0

    return {
        "bootstrap_prob_positif": round(float(np.mean(final_profits > 0) * 100), 2),
        "bootstrap_mediane_profit": round(float(np.median(final_profits)), 2),
        "bootstrap_pire_profit_5pct": round(float(np.percentile(final_profits, 5)), 2),
        "bootstrap_pire_mdd": round(float(np.percentile(max_drawdowns, 95)), 2)
    }


# ============================================================
# STRESS PRO — CHANTIER 2 : Coûts & Slippage
#
# NOTE MÉTHODOLOGIQUE : un coût fixe en $ par trade n'a aucun sens indépendamment
# de la taille du compte — 2$ est écrasant sur un compte à 500$ (0.4% par
# trade) et négligeable sur un compte à 100 000$ (0.002%). Par défaut (si
# le volume par trade n'est pas disponible), on utilise le capital de
# référence comme proxy, sous l'hypothèse que la taille de position est
# dimensionnée en % du capital — une approximation raisonnable mais
# UNIFORME (même coût pour chaque trade). Si le volume RÉEL de chaque
# trade est disponible (colonne exportée par la plateforme), le coût
# est proportionnel à la taille relative de chaque trade — plus précis,
# un gros trade coûte proportionnellement plus qu'un petit.
# ============================================================

def stress_test_costs(trades_array, capital_ref, volumes=None, cost_pct_per_trade=COST_PCT_PER_TRADE):
    """Teste la marge face à plusieurs niveaux de coûts supplémentaires.

    Le coût de référence reste une CONVENTION de stress, pas une estimation
    des frais réels du broker. Si le P&L exporté est déjà net de commission,
    ce test ajoute volontairement un coût supplémentaire : il mesure donc une
    marge de sécurité vis-à-vis de conditions plus défavorables, et non les
    frais réellement payés.

    Les scénarios sont exprimés comme multiples du coût conventionnel :
    0.5x, 1x, 1.5x et 2x. Le résultat historique `stressed_profit` reste
    calculé au scénario 1x pour compatibilité avec l'UI existante.
    """
    if len(trades_array) == 0 or not capital_ref or capital_ref <= 0:
        return {
            "stressed_profit": 0.0,
            "cost_penalty_total": 0.0,
            "cost_per_trade_dollars": 0.0,
            "profitable_under_cost_stress": False,
            "cost_ajuste_par_volume": False,
            "cost_base_pct_per_trade": cost_pct_per_trade * 100,
            "cost_break_even_pct_per_trade": None,
            "cost_scenarios": [],
        }

    cost_base = capital_ref * cost_pct_per_trade
    cost_ajuste_par_volume = False
    couts_par_trade_base = None

    if volumes is not None and len(volumes) == len(trades_array):
        volumes_valides = [v for v in volumes if v is not None and v > 0]
        # Ajustement seulement si au moins la moitié des trades ont un volume connu.
        if len(volumes_valides) >= len(trades_array) * 0.5:
            volume_moyen = float(np.mean(volumes_valides))
            if volume_moyen > 0:
                couts_par_trade_base = np.array([
                    cost_base * (v / volume_moyen) if (v is not None and v > 0) else cost_base
                    for v in volumes
                ])
                cost_ajuste_par_volume = True

    if couts_par_trade_base is None:
        couts_par_trade_base = np.full(len(trades_array), cost_base)

    total_penalty_base = float(np.sum(couts_par_trade_base))
    stressed_profit_base = float(np.sum(trades_array - couts_par_trade_base))
    cost_per_trade_moyen = round(total_penalty_base / len(trades_array), 2)

    # Coût moyen supplémentaire maximal avant que le profit OOS ne tombe à 0.
    profit_brut = float(np.sum(trades_array))
    break_even_cost_total = profit_brut if profit_brut > 0 else 0.0
    break_even_cost_pct_per_trade = (
        break_even_cost_total / (len(trades_array) * capital_ref)
        if break_even_cost_total > 0 else 0.0
    )

    multiplicateurs = [0.5, 1.0, 1.5, 2.0]
    scenarios = []
    for mult in multiplicateurs:
        couts = couts_par_trade_base * mult
        penalty = float(np.sum(couts))
        profit = float(np.sum(trades_array - couts))
        scenarios.append({
            "multiplicateur": mult,
            "cout_total": round(penalty, 2),
            "cout_moyen_par_trade": round(penalty / len(trades_array), 2),
            "profit_net_apres_couts": round(profit, 2),
            "rentable": bool(profit > 0),
        })

    return {
        "stressed_profit": round(stressed_profit_base, 2),
        "cost_penalty_total": round(total_penalty_base, 2),
        "cost_per_trade_dollars": cost_per_trade_moyen,
        "profitable_under_cost_stress": bool(stressed_profit_base > 0),
        "cost_ajuste_par_volume": cost_ajuste_par_volume,
        "cost_base_pct_per_trade": round(cost_pct_per_trade * 100, 4),
        "cost_break_even_pct_per_trade": round(break_even_cost_pct_per_trade * 100, 4),
        "cost_scenarios": scenarios,
    }


# ============================================================
# STRESS PRO — CHANTIER 3 : Découpe temporelle (Rolling)
# ============================================================

def _fenetres_rolling_avec_chevauchement(trades, n_windows):
    """Construit de vraies fenêtres rolling sur UNE séquence chronologique.

    Les fenêtres se chevauchent : contrairement à np.array_split, un trade peut
    donc appartenir à plusieurs fenêtres. La taille de fenêtre est choisie
    pour obtenir environ `n_windows` observations et le pas est inférieur à la
    taille de fenêtre. Aucun raccordement n'est fait entre deux segments OOS.
    """
    trades = np.asarray(trades, dtype=float)
    n = len(trades)
    if n == 0 or n_windows < 1:
        return []
    if n < n_windows:
        return []

    # Une vraie fenêtre rolling doit avoir une taille supérieure au pas :
    # deux fenêtres successives doivent donc partager au moins une partie
    # de leurs observations. On vise ici ~50% de chevauchement, puis on
    # répartit les points de départ sur toute la séquence.
    if n_windows == 1:
        return [trades]

    window_size = max(2, int(math.ceil((2 * n) / n_windows)))
    if window_size >= n:
        return []

    max_start = n - window_size
    if max_start < n_windows - 1:
        # Impossible de construire n_windows fenêtres distinctes avec un
        # chevauchement réel sur cet échantillon. Mieux vaut signaler une
        # insuffisance que retourner des fenêtres dupliquées.
        return []

    starts = np.rint(np.linspace(0, max_start, n_windows)).astype(int).tolist()
    starts = list(dict.fromkeys(starts))
    if len(starts) != n_windows:
        return []

    windows = [trades[start:start + window_size] for start in starts]
    # Garde-fou explicite : chaque paire consécutive doit partager au moins
    # une observation.
    for a, b in zip(windows, windows[1:]):
        if not np.intersect1d(a, b).size:
            return []
    return windows


def stress_test_rolling(trades_array, n_windows=N_ROLLING_WINDOWS, segments=None):
    """Stabilité temporelle par vraies fenêtres rolling, sans franchir les gaps OOS.

    Avec `segments`, chaque fenêtre OOS est traitée indépendamment. Cela évite
    de créer une fausse continuité entre deux périodes de test séparées par
    une période IS. Sans `segments`, le comportement reste celui d'une seule
    séquence chronologique.
    """
    if segments is None:
        segments = [np.asarray(trades_array, dtype=float)]
    else:
        segments = [np.asarray(seg, dtype=float) for seg in segments if len(seg) > 0]

    rolling_windows = []
    for segment_index, segment in enumerate(segments):
        for window_index, window in enumerate(_fenetres_rolling_avec_chevauchement(segment, n_windows)):
            rolling_windows.append({
                "segment": segment_index,
                "window": window_index,
                "trades": window,
                "profit": float(np.sum(window)),
            })

    if not rolling_windows:
        return {
            "rolling_profitable_pct": 0.0,
            "rolling_min_window_profit": 0.0,
            "rolling_max_window_profit": 0.0,
            "rolling_stable": False,
            "rolling_windows_testes": 0,
            "rolling_segments_testes": len(segments),
            "rolling_chevauchement": True,
            "rolling_insuffisant": True,
        }

    window_profits = [w["profit"] for w in rolling_windows]
    profitable_windows = sum(1 for p in window_profits if p > 0)
    profitable_pct = (profitable_windows / len(window_profits)) * 100.0

    return {
        "rolling_profitable_pct": round(profitable_pct, 1),
        "rolling_min_window_profit": round(min(window_profits), 2),
        "rolling_max_window_profit": round(max(window_profits), 2),
        "rolling_stable": bool(profitable_pct >= 60.0),
        "rolling_windows_testes": len(rolling_windows),
        "rolling_segments_testes": len(segments),
        "rolling_chevauchement": True,
        "rolling_insuffisant": False,
    }


# ============================================================
# COUVERTURE TEMPORELLE — signal honnête, pas une vraie détection de
# régime de marché.
#
# XTRUNN ne reçoit que le journal de trades, jamais les données de prix
# sous-jacentes (bougies, volatilité réalisée, etc.) — il ne peut donc
# PAS détecter si le backtest a réellement traversé une tendance
# haussière, un marché baissier, une phase de range ou un pic de
# volatilité. Ce qu'on PEUT faire honnêtement avec les dates déjà
# extraites par les parsers : signaler qu'un historique court a
# statistiquement très peu de chances d'avoir traversé des conditions
# variées, sans jamais prétendre confirmer que c'est le cas sur un
# historique long.
# ============================================================

SPAN_JOURS_CRITIQUE = 90
SPAN_JOURS_AVERTISSEMENT = 365


def extraire_periode(details_list):
    """Date de début et de fin d'un ensemble de trades — retourne None si
    les dates sont absentes ou illisibles, plutôt qu'une valeur trompeuse."""
    dates_brutes = [d.get("date") for d in details_list if d and d.get("date")]
    if not dates_brutes:
        return None
    try:
        # Ne jamais appliquer une heuristique day-first aux formats MT5
        # non ambigus YYYY.MM.DD : une liste comme 2023.06.07 /
        # 2023.07.01 serait sinon interprétée avec mois/jour inversés.
        dates_parsees, _ = _parse_dates_robuste(dates_brutes, dayfirst_default=True)
    except Exception:
        return None
    dates_valides = dates_parsees.dropna()
    if len(dates_valides) == 0:
        return None
    return {
        "debut": dates_valides.min().strftime("%Y-%m-%d"),
        "fin": dates_valides.max().strftime("%Y-%m-%d"),
    }


def estimer_couverture_temporelle(details_list):
    dates_brutes = [d.get("date") for d in details_list if d and d.get("date")]
    if len(dates_brutes) < 2:
        return None

    try:
        # Ne jamais appliquer une heuristique day-first aux formats MT5
        # non ambigus YYYY.MM.DD : une liste comme 2023.06.07 /
        # 2023.07.01 serait sinon interprétée avec mois/jour inversés.
        dates_parsees, _ = _parse_dates_robuste(dates_brutes, dayfirst_default=True)
    except Exception:
        return None
    dates_valides = dates_parsees.dropna()
    if len(dates_valides) < 2:
        return None

    span_jours = int((dates_valides.max() - dates_valides.min()).days)
    return {
        "span_jours": span_jours,
        "date_debut": dates_valides.min().strftime("%Y-%m-%d"),
        "date_fin": dates_valides.max().strftime("%Y-%m-%d"),
    }


# ============================================================
# DÉCOUPAGE WALK-FORWARD
# ============================================================

def decouper_fenetres_walkforward(trades_array, details_list, n_windows, is_ratio):
    """Découpe une série déjà normalisée en fenêtres chronologiques contiguës.

    Le découpage est équilibré en *nombre de trades*, pas en durée calendaire.
    Cela garantit une quantité de données comparable par fenêtre sans inventer
    une durée fixe pour les stratégies à fréquence variable. Chaque fenêtre
    contient une portion Référence (IS), puis immédiatement la portion Test
    (OOS). Les fenêtres ne se chevauchent pas et couvrent exactement tout
    l'historique.

    Important : les OOS de fenêtres successives ne sont PAS contigus dans le
    temps, car la période IS de la fenêtre suivante se trouve entre les deux.
    Les métriques de trajectoire doivent donc utiliser les segments séparément.
    """
    total = len(trades_array)
    if total != len(details_list):
        raise ValueError("Impossible de découper les fenêtres : trades et détails désalignés.")
    if n_windows <= 0:
        raise ValueError("Le nombre de fenêtres doit être positif.")

    # np.array_split répartit le reliquat sur les premières fenêtres et évite
    # qu'une dernière fenêtre anormalement longue absorbe tous les trades restants.
    bornes = np.array_split(np.arange(total), n_windows)
    fenetres = []
    for w, indices in enumerate(bornes, start=1):
        if len(indices) == 0:
            raise ValueError(f"Fenêtre {w} vide : historique insuffisant pour ce nombre de fenêtres.")
        debut = int(indices[0])
        fin = int(indices[-1]) + 1
        fin_is = debut + int(round((fin - debut) * is_ratio))
        fin_is = max(debut + 1, min(fin - 1, fin_is))
        fenetres.append({
            "index": w,
            "debut": debut, "fin_is": fin_is, "fin": fin,
            "is_trades": trades_array[debut:fin_is],
            "oos_trades": trades_array[fin_is:fin],
            "is_detail": details_list[debut:fin_is],
            "oos_detail": details_list[fin_is:fin],
        })
    return fenetres


def calculer_ratio_stabilite(is_trades, oos_trades):
    """Compare l'efficacité OOS à l'efficacité IS sans pseudo-efficacité.

    L'efficacité ``profit / drawdown`` n'est interprétable que lorsque le
    profit est positif et que le drawdown est strictement positif. On ne
    remplace donc plus les cas non définis par une constante arbitraire
    (anciennement ``0.1``), qui pouvait produire des ratios trompeurs.

    Cas limites :
    - IS ou OOS non profitable : stabilité = 0 ;
    - IS et OOS profitables sans drawdown : stabilité = 1 (aucune
      dégradation mesurable, mais aucun ratio d'efficacité fini n'existe) ;
    - IS sans drawdown, OOS avec drawdown : stabilité = 0 ;
    - IS avec drawdown, OOS sans drawdown : stabilité = 1 ;
    - sinon : (profit OOS / DD OOS) / (profit IS / DD IS).
    """
    profit_is = float(np.sum(is_trades)) if len(is_trades) > 0 else 0.0
    profit_oos = float(np.sum(oos_trades)) if len(oos_trades) > 0 else 0.0
    dd_is = calculer_max_drawdown(is_trades) if len(is_trades) > 0 else 0.0
    dd_oos = calculer_max_drawdown(oos_trades) if len(oos_trades) > 0 else 0.0

    # Une période non profitable ne démontre pas une stabilité exploitable.
    if profit_is <= 0 or profit_oos <= 0:
        return 0.0

    # Les deux périodes sont profitables sans drawdown : le ratio d'efficacité
    # est mathématiquement indéfini, mais aucune dégradation n'est observable.
    if dd_is <= 0 and dd_oos <= 0:
        return 1.0

    # IS sans DD puis OOS avec DD : l'efficacité s'est dégradée.
    if dd_is <= 0 and dd_oos > 0:
        return 0.0

    # IS avec DD puis OOS sans DD : aucune dégradation mesurable.
    if dd_is > 0 and dd_oos <= 0:
        return 1.0

    eff_is = profit_is / dd_is
    eff_oos = profit_oos / dd_oos
    return float(eff_oos / eff_is) if eff_is > 0 else 0.0


# ============================================================
# PILIER 2 : Stabilité IS -> OOS, VERSION WALK-FORWARD MULTI-FENÊTRES
#
# Au lieu d'un seul ratio de stabilité (un échantillon de taille 1), on en
# calcule un par fenêtre et on note la stratégie sur la MÉDIANE de ces
# ratios — moins sensible à une fenêtre isolée chanceuse ou malchanceuse
# qu'un split unique. Un signal de cohérence supplémentaire pénalise les
# stratégies dont les fenêtres sont très inégales entre elles, même si la
# médiane est correcte.
# ============================================================

def analyser_pilier_is_oos(fenetres, oos_trades_all):
    ratios = []
    fenetres_valides = 0
    for f in fenetres:
        if len(f["is_trades"]) > 0 and len(f["oos_trades"]) > 0:
            ratios.append(calculer_ratio_stabilite(f["is_trades"], f["oos_trades"]))
            fenetres_valides += 1

    warnings = []

    if fenetres_valides == 0:
        return {
            "score": 0, "stability_ratio": 0.0, "oos_profit_factor": 0.0,
            "ratios_par_fenetre": [], "fenetres_faibles_pct": 0.0,
            "warnings": [{"level": "critical", "message": "Aucune fenêtre exploitable pour évaluer la stabilité entre périodes."}]
        }

    stability_ratio = float(np.median(ratios))
    profit_oos_total = float(oos_trades_all.sum()) if len(oos_trades_all) > 0 else 0.0

    score = 0
    if profit_oos_total <= 0:
        score = 0
        warnings.append({"level": "critical", "message": "La stratégie est globalement perdante sur l'ensemble des périodes de test — le signal le plus grave qui soit."})
    elif stability_ratio < 0.5:
        score = 5
        warnings.append({"level": "critical", "message": f"Forte dégradation hors période de référence : l'efficacité OOS est nettement inférieure à celle observée en référence (ratio {stability_ratio:.2f}, sous 0.5). Ce profil est compatible avec une forte dépendance aux conditions de la période de référence."})
    elif stability_ratio < 0.7:
        score = 12
        warnings.append({"level": "warning", "message": f"Baisse de performance notable en dehors des données de réglage (ratio {stability_ratio:.2f}) — à surveiller."})
    elif stability_ratio < 0.9:
        score = 22
    else:
        score = POIDS_STABILITE_ISOOS

    # Cohérence entre fenêtres : même avec une bonne médiane, une stratégie
    # dont la moitié des fenêtres sont des overfits majeurs (<0.5) et l'autre
    # moitié excellentes n'est pas vraiment fiable — la médiane masquerait ça.
    fenetres_faibles = sum(1 for r in ratios if r < 0.5)
    fenetres_faibles_pct = round(fenetres_faibles / len(ratios) * 100, 1)
    if fenetres_faibles_pct >= SEUIL_FENETRES_FAIBLES_PCT and stability_ratio >= SEUIL_STABILITE_POUR_PENALITE:
        score = int(score * MULT_PENALITE_INCOHERENCE)
        warnings.append({
            "level": "warning",
            "message": f"Résultats inégaux selon les périodes testées : {fenetres_faibles_pct:.0f}% d'entre elles montrent un net surapprentissage, malgré une moyenne correcte. La performance ne semble pas également reproductible dans le temps."
        })

    # Plafond de qualité absolue, basé sur le Profit Factor calculé sur
    # l'ensemble des trades OOS de toutes les fenêtres concaténées.
    oos_pf = calculer_profit_factor(oos_trades_all) if len(oos_trades_all) > 0 else 0.0
    if profit_oos_total > 0:
        mult = qualite_absolue_multiplicateur(oos_pf)
        score_avant_plafond = score
        score = int(round(score * mult))
        if mult < 1.0 and score < score_avant_plafond and oos_pf != PF_SENTINELLE_AUCUNE_PERTE:
            warnings.append({
                "level": "warning",
                "message": f"Marge de profit limitée : le Profit Factor sur les périodes de test ({oos_pf:.2f}) reste proche de 1 (le seuil de rentabilité) — le score est plafonné même si la stratégie reste par ailleurs stable."
            })

    return {
        "score": int(score),
        "stability_ratio": round(stability_ratio, 2),
        "oos_profit_factor": round(oos_pf, 2),
        "ratios_par_fenetre": [round(r, 2) for r in ratios],
        "fenetres_faibles_pct": fenetres_faibles_pct,
        "warnings": warnings
    }


# ============================================================
# PILIER 3 : Risque & Drawdown
# ============================================================

def analyser_pilier_risque_drawdown(oos_trades, segments=None):
    if len(oos_trades) == 0:
        return {"score": 0, "warnings": [{"level": "critical", "message": "Aucun trade sur la période de test pour analyser le risque."}], "max_losing_streak": 0, "profit_to_dd_ratio": None}

    profit_oos = float(oos_trades.sum())
    if segments:
        segments_valides = [np.asarray(seg, dtype=float) for seg in segments if len(seg) > 0]
        dd_oos = max((calculer_max_drawdown(seg) for seg in segments_valides), default=0.0)
    else:
        dd_oos = calculer_max_drawdown(oos_trades)

    warnings = []
    score = POIDS_RISQUE_DRAWDOWN
    # Deux vues sont conservées : le ratio agrégé décrit le résultat global,
    # tandis que le score doit aussi tenir compte d'une fenêtre OOS
    # individuellement défavorable. Sinon une grosse bonne fenêtre peut
    # masquer une fenêtre de test très dégradée.
    profit_to_dd_ratio_global = None
    profit_to_dd_ratio_pire_fenetre = None

    if dd_oos > 0:
        profit_to_dd_ratio_global = round(profit_oos / dd_oos, 2)

    if segments:
        ratios_fenetres = []
        for seg in segments:
            seg = np.asarray(seg, dtype=float)
            if len(seg) == 0:
                continue
            seg_profit = float(np.sum(seg))
            seg_dd = calculer_max_drawdown(seg)
            if seg_dd > 0:
                ratios_fenetres.append(seg_profit / seg_dd)
            elif seg_profit <= 0:
                # Une fenêtre sans drawdown mais non profitable reste une
                # fenêtre défavorable pour la robustesse temporelle.
                ratios_fenetres.append(float('-inf'))
        if ratios_fenetres:
            ratio_pire = min(ratios_fenetres)
            profit_to_dd_ratio_pire_fenetre = (
                round(ratio_pire, 2) if np.isfinite(ratio_pire) else None
            )

    ratio_pour_score = profit_to_dd_ratio_pire_fenetre if profit_to_dd_ratio_pire_fenetre is not None else profit_to_dd_ratio_global

    if ratio_pour_score is not None:
        if ratio_pour_score < 0.5:
            score -= 12
            warnings.append({"level": "critical", "message": "Risque élevé : au moins une période de test présente un rapport profit/drawdown inférieur à 0,5."})
        elif ratio_pour_score < 1.0:
            score -= 6
            warnings.append({"level": "warning", "message": "Au moins une période de test présente un drawdown élevé par rapport au profit généré."})

    # Les fenêtres OOS sont séparées dans le temps : une série de pertes ne
    # doit jamais être prolongée artificiellement d'une fenêtre à la suivante.
    segments_streak = segments if segments else [oos_trades]
    max_pertes_consecutives = 0
    for segment in segments_streak:
        pertes_consecutives = 0
        for t in segment:
            if t < 0:
                pertes_consecutives += 1
                max_pertes_consecutives = max(max_pertes_consecutives, pertes_consecutives)
            else:
                pertes_consecutives = 0

    if max_pertes_consecutives >= 8:
        score -= 8
        warnings.append({"level": "critical", "message": f"Série de pertes : {max_pertes_consecutives} pertes consécutives observées pendant les tests."})
    elif max_pertes_consecutives >= 5:
        score -= 4
        warnings.append({"level": "warning", "message": f"Série de pertes notable : {max_pertes_consecutives} pertes d'affilée pendant les tests."})

    return {
        "score": max(0, score),
        "max_losing_streak": int(max_pertes_consecutives),
        "profit_to_dd_ratio": profit_to_dd_ratio_global,
        "profit_to_dd_ratio_pire_fenetre": profit_to_dd_ratio_pire_fenetre,
        "warnings": warnings
    }


# ============================================================
# PILIER 4 : Concentration des profits (effet "loterie")
# ============================================================

def evaluer_fiabilite_evaluation(n_trades_oos, couverture_temporelle=None):
    """Qualifie la fiabilité de l'évaluation sans modifier le score de robustesse.

    Cette qualification décrit la quantité de données disponibles pour
    interpréter le résultat. Elle ne constitue pas une garantie statistique
    et n'est pas une pénalité de score.
    """
    raisons = []
    if n_trades_oos < MIN_TRADES_ABSOLUTE:
        statut = "insuffisante"
        raisons.append(
            f"Seulement {n_trades_oos} trade(s) de test : trop peu d'observations pour interpréter correctement les métriques."
        )
    elif n_trades_oos < MIN_TRADES_RELIABLE:
        statut = "limitée"
        raisons.append(
            f"{n_trades_oos} trades de test : volume d'observations inférieur au seuil recommandé de {MIN_TRADES_RELIABLE}."
        )
    else:
        statut = "fiable"
        raisons.append(
            f"{n_trades_oos} trades de test : volume d'observations au-dessus du seuil recommandé de {MIN_TRADES_RELIABLE}."
        )

    if couverture_temporelle is not None:
        span = couverture_temporelle.get("span_jours")
        if span is not None and span < SPAN_JOURS_CRITIQUE:
            raisons.append(f"Couverture temporelle limitée : {span} jours entre le premier et le dernier trade.")
        elif span is not None and span < SPAN_JOURS_AVERTISSEMENT:
            raisons.append(f"Couverture temporelle inférieure à un an : {span} jours.")

    return {"statut": statut, "raisons": raisons}


def analyser_pilier_concentration(oos_trades):
    if len(oos_trades) == 0:
        return {"score": 0, "warnings": [{"level": "critical", "message": "Aucun trade sur la période de test pour analyser la concentration."}], "concentration_pct": 0, "top_n_trades": 0, "top_gains_sum": 0.0}

    profit_total = float(oos_trades.sum())
    if profit_total <= 0:
        return {"score": 0, "warnings": [{"level": "critical", "message": "Profit net négatif ou nul sur la période de test, impossible d'analyser la concentration."}], "concentration_pct": 100, "top_n_trades": 0, "top_gains_sum": 0.0}

    gains = oos_trades[oos_trades > 0]
    if len(gains) == 0:
        return {"score": 0, "warnings": [{"level": "critical", "message": "Aucun trade gagnant sur la période de test."}], "concentration_pct": 100, "top_n_trades": 0, "top_gains_sum": 0.0}

    gains_tries = np.sort(gains)[::-1]
    top_n = max(1, int(len(gains) * 0.1))
    somme_top_gains = float(np.sum(gains_tries[:top_n]))

    concentration_pct = (somme_top_gains / profit_total) * 100

    score = POIDS_CONCENTRATION
    warnings = []

    if len(oos_trades[oos_trades < 0]) == 0:
        warnings.append({
            "level": "warning",
            "message": "Aucune perte observée sur les périodes de test : les métriques fondées sur les pertes ne peuvent pas être vérifiées sur cet échantillon."
        })

    if concentration_pct > 70:
        score -= 15
        suffix = " Ce pourcentage peut dépasser 100% lorsque les pertes importantes ailleurs dans l'échantillon réduisent fortement le profit net." if concentration_pct > 100 else ""
        warnings.append({"level": "critical", "message": f"Concentration élevée : les 10% de trades les plus rentables génèrent à eux seuls {concentration_pct:.1f}% du profit net — ce résultat dépend fortement d'un petit sous-ensemble de trades.{suffix}"})
    elif concentration_pct > 50:
        score -= 8
        suffix = " Ce pourcentage peut dépasser 100% lorsque les pertes importantes ailleurs dans l'échantillon réduisent fortement le profit net." if concentration_pct > 100 else ""
        warnings.append({"level": "warning", "message": f"Concentration notable des profits : {concentration_pct:.1f}% du profit net vient des 10% de trades les plus rentables.{suffix}"})

    return {
        "score": max(0, score),
        "concentration_pct": round(concentration_pct, 1),
        "top_n_trades": int(top_n),
        "top_gains_sum": round(somme_top_gains, 2),
        "warnings": warnings
    }


# ============================================================
# DÉTECTION DE RISQUE DE RUINE (money management dangereux)
#
# Une stratégie de type martingale, grid, ou moyennage à la baisse
# augmente la taille de position après une perte pour "se refaire". Le
# backtest résultant a typiquement l'air EXCELLENT — winrate élevé, gains
# réguliers, drawdown propre — jusqu'à ce qu'une série de pertes fasse
# sauter le compte. Si cet événement catastrophique n'est pas tombé dans
# la période testée par pur hasard, rien dans une analyse de résultats
# seuls ne le voit venir.
#
# HONNÊTETÉ MÉTHODOLOGIQUE : XTRUNN ne reçoit que le profit de chaque
# trade, jamais la taille de position — il ne peut donc PAS confirmer
# directement qu'un money management dangereux est utilisé. Ce qui suit
# détecte des SIGNATURES STATISTIQUES connues de ce type de risque
# (winrate élevé + ratio gain/perte très faible, pertes qui s'aggravent
# au sein d'une série, perte isolée démesurée) — un signal d'alerte
# sérieux, pas une preuve formelle.
# ============================================================

RUIN_WINRATE_SEUIL = 75.0        # % de trades gagnants au-delà duquel une stratégie devient suspecte si...
RUIN_PAYOFF_SEUIL = 0.35         # ...le ratio gain moyen / perte moyenne tombe sous ce seuil
RUIN_ESCALADE_SEUIL_PCT = 50.0   # % des séries de pertes consécutives montrant une aggravation
RUIN_QUEUE_SEUIL = 6.0           # perte la plus grosse = X fois la perte moyenne


def detecter_pertes_escalade(trades):
    """
    Repère, au sein des séries de pertes consécutives (3 pertes ou plus),
    si les pertes ont tendance à s'aggraver au fil de la série — signature
    typique d'une taille de position qui augmente après chaque perte
    (martingale/grid). Une séquence est jugée "en escalade" si une
    majorité de ses pertes successives augmentent ET que la dernière est
    nettement plus grosse que la première.
    """
    def evaluer_sequence(seq):
        if len(seq) < 3:
            return None
        increases = sum(1 for i in range(1, len(seq)) if seq[i] > seq[i - 1])
        pct_increases = increases / (len(seq) - 1)
        ratio_derniere_premiere = (seq[-1] / seq[0]) if seq[0] > 0 else 0
        return pct_increases >= 0.6 and ratio_derniere_premiere >= 1.4

    sequences_escalade = 0
    total_sequences = 0
    courant = []

    for t in trades:
        if t < 0:
            courant.append(abs(float(t)))
        else:
            resultat = evaluer_sequence(courant)
            if resultat is not None:
                total_sequences += 1
                if resultat:
                    sequences_escalade += 1
            courant = []
    resultat = evaluer_sequence(courant)
    if resultat is not None:
        total_sequences += 1
        if resultat:
            sequences_escalade += 1

    pct_escalade = round(sequences_escalade / total_sequences * 100, 1) if total_sequences > 0 else 0.0
    return pct_escalade, total_sequences


def detecter_volume_escalade(trades, volumes):
    """
    Signal DIRECT (contrairement aux 3 autres, qui sont des inférences
    statistiques) : vérifie si la taille de position augmente RÉELLEMENT
    au sein des séries de pertes consécutives — la vraie signature d'un
    martingale/grid, lue directement dans les données plutôt que devinée
    depuis l'ampleur des pertes. Disponible uniquement si la plateforme
    exportée renseignait une colonne volume/lots.
    """
    def evaluer_sequence(vols):
        if len(vols) < 3:
            return None
        increases = sum(1 for i in range(1, len(vols)) if vols[i] > vols[i - 1])
        pct_increases = increases / (len(vols) - 1)
        ratio_derniere_premiere = (vols[-1] / vols[0]) if vols[0] > 0 else 0
        return pct_increases >= 0.6 and ratio_derniere_premiere >= 1.4

    sequences_escalade = 0
    total_sequences = 0
    courant = []

    for t, v in zip(trades, volumes):
        if t < 0 and v is not None and v > 0:
            courant.append(float(v))
        elif t < 0:
            # Perte avec volume inconnu -> impossible de juger cette
            # séquence, on l'interrompt proprement plutôt que de deviner.
            courant = []
        else:
            resultat = evaluer_sequence(courant)
            if resultat is not None:
                total_sequences += 1
                if resultat:
                    sequences_escalade += 1
            courant = []
    resultat = evaluer_sequence(courant)
    if resultat is not None:
        total_sequences += 1
        if resultat:
            sequences_escalade += 1

    pct_escalade = round(sequences_escalade / total_sequences * 100, 1) if total_sequences > 0 else 0.0
    return pct_escalade, total_sequences


def evaluer_cohesion_volume(volumes):
    """Statistique purement informative (pas de score) : coefficient de
    variation des tailles de position. Ne dit rien de bien ou mal en soi
    (une stratégie peut légitimement varier sa taille selon la
    confiance/volatilité) — juste un fait objectif mis à disposition."""
    volumes_valides = [v for v in volumes if v is not None and v > 0]
    if len(volumes_valides) < 10:
        return None
    arr = np.array(volumes_valides, dtype=float)
    moyenne = float(np.mean(arr))
    if moyenne <= 0:
        return None
    cv = float(np.std(arr) / moyenne)
    return {
        "n_avec_volume": len(volumes_valides),
        "n_total": len(volumes),
        "volume_moyen": round(moyenne, 4),
        "coefficient_variation_pct": round(cv * 100, 1),
    }


def evaluer_ratio_risque_recompense(details):
    """
    Signal purement informatif (jamais noté) : compare la distance au stop
    loss et au take profit RÉELLEMENT CONFIGURÉS sur chaque trade, quand
    ces données sont disponibles dans l'export. Un ratio élevé (SL loin,
    TP proche) n'est PAS en soi la preuve d'un problème — une stratégie
    peut légitimement réduire sa taille de position en conséquence — mais
    XTRUNN ne peut pas vérifier ce calibrage sans connaître la valeur du
    point/pip, donc ça reste une mise en garde structurelle, jamais une
    pénalité de score. Contrairement au Risque de Ruine (basé sur ce qui
    s'est réellement produit), ce signal porte sur ce qui était configuré
    — donc reste pertinent même si le stop n'a, par chance, jamais été
    touché pendant la période testée.
    """
    ratios = []
    for d in details:
        sl, tp, open_price = d.get("sl"), d.get("tp"), d.get("open_price")
        if sl is None or tp is None or open_price is None:
            continue
        risque = abs(open_price - sl)
        recompense = abs(tp - open_price)
        if risque > 0 and recompense > 0:
            ratios.append(risque / recompense)

    if len(ratios) < 10:
        return None

    ratio_median = float(np.median(ratios))
    if ratio_median < 1.5:
        niveau = "normal"
    elif ratio_median < 3.0:
        niveau = "notable"
    else:
        niveau = "extreme"

    return {
        "n_avec_donnees": len(ratios),
        "n_total": len(details),
        "ratio_median": round(ratio_median, 2),
        "niveau": niveau,
    }


def evaluer_risque_de_ruine(trades, volumes=None):
    n = len(trades)
    if n < 10:
        return {
            "mult_risque_ruine": 1.0, "winrate": 0.0, "payoff_ratio": None,
            "pct_sequences_escalade": 0.0, "n_sequences_pertes": 0, "ratio_queue": None,
            "signal_payoff": False, "signal_escalade": False, "signal_queue": False,
            "signal_volume": False, "pct_volume_escalade": None, "n_sequences_volume": 0,
            "cohesion_volume": None,
            "alertes": [],
        }

    gains = trades[trades > 0]
    pertes = trades[trades < 0]
    winrate = (len(gains) / n * 100) if n > 0 else 0.0
    gain_moyen = float(np.mean(gains)) if len(gains) > 0 else 0.0
    perte_moyenne = float(np.mean(pertes)) if len(pertes) > 0 else 0.0
    plus_grosse_perte = float(np.min(trades))

    payoff_ratio = (gain_moyen / abs(perte_moyenne)) if perte_moyenne != 0 else None

    alertes = []

    # Ces signaux sont diagnostiques uniquement. Ils ne modifient jamais le
    # score global : certains recouvrent déjà le Pilier 3 (risque/DD), et les
    # données de trades seules ne permettent pas d'établir un risque de ruine.
    # On conserve leur valeur pour l'interface et le rapport.

    # Signal 1 : winrate élevé + ratio gain/perte très faible.
    signal_payoff = bool(payoff_ratio is not None and winrate >= RUIN_WINRATE_SEUIL and payoff_ratio < RUIN_PAYOFF_SEUIL)
    if signal_payoff:
        alertes.append({
            "level": "critical",
            "message": f"Signal de gestion du risque à examiner : winrate élevé ({winrate:.0f}%) combiné à des gains faibles face aux pertes ({payoff_ratio:.2f}) — profil compatible avec une exposition à des pertes rares mais importantes."
        })

    # Signal 2 : escalade des pertes au sein des séries.
    pct_escalade, n_sequences = detecter_pertes_escalade(trades)
    signal_escalade = bool(n_sequences >= 2 and pct_escalade >= RUIN_ESCALADE_SEUIL_PCT)
    if signal_escalade:
        alertes.append({
            "level": "critical",
            "message": f"Escalade des pertes observée : {pct_escalade:.0f}% des séries de pertes consécutives montrent une perte qui s'aggrave progressivement — signal à vérifier dans la logique de taille de position."
        })

    # Signal 3 : queue de distribution anormalement lourde.
    ratio_queue = (abs(plus_grosse_perte) / abs(perte_moyenne)) if perte_moyenne != 0 else None
    signal_queue = bool(ratio_queue is not None and ratio_queue >= RUIN_QUEUE_SEUIL)
    if signal_queue:
        alertes.append({
            "level": "warning",
            "message": f"Perte extrême isolée : la plus grosse perte du backtest ({plus_grosse_perte:.2f}) est {ratio_queue:.1f}x plus importante que la perte moyenne — la distribution observée comporte donc une perte nettement plus importante que les autres."
        })

    # Signal 4 : escalade RÉELLE du volume au sein des séries de pertes —
    # seul signal direct de ce module (les 3 précédents sont des
    # inférences statistiques). Le signal est plus direct, mais il ne constitue
    # pas à lui seul une preuve de martingale ou de risque de ruine. Disponible seulement si la plateforme a
    # exporté les tailles de position.
    signal_volume = False
    pct_volume_escalade = None
    n_sequences_volume = 0
    if volumes is not None:
        volumes_connus = sum(1 for v in volumes if v is not None and v > 0)
        if volumes_connus >= n * 0.5:  # au moins la moitié des trades ont un volume connu, sinon signal peu fiable
            pct_volume_escalade, n_sequences_volume = detecter_volume_escalade(trades, volumes)
            signal_volume = bool(n_sequences_volume >= 2 and pct_volume_escalade >= RUIN_ESCALADE_SEUIL_PCT)
            if signal_volume:
                alertes.append({
                    "level": "critical",
                    "message": f"Escalade de taille observée : {pct_volume_escalade:.0f}% des séries de pertes consécutives montrent une taille de position qui augmente selon le critère testé — signal directement observé dans les données de volume, à vérifier dans le money management."
                })

    cohesion_volume = evaluer_cohesion_volume(volumes) if volumes is not None else None

    return {
        "mult_risque_ruine": 1.0,  # rétrocompatibilité API ; aucun multiplicateur n'est appliqué
        "winrate": round(winrate, 1),
        "payoff_ratio": round(payoff_ratio, 2) if payoff_ratio is not None else None,
        "pct_sequences_escalade": pct_escalade,
        "n_sequences_pertes": n_sequences,
        "ratio_queue": round(ratio_queue, 2) if ratio_queue is not None else None,
        "signal_payoff": signal_payoff,
        "signal_escalade": signal_escalade,
        "signal_queue": signal_queue,
        "signal_volume": signal_volume,
        "pct_volume_escalade": pct_volume_escalade,
        "n_sequences_volume": n_sequences_volume,
        "cohesion_volume": cohesion_volume,
        "alertes": alertes,
    }


# ============================================================
# FOURCHETTE DE CONFIANCE DU SCORE (bootstrap avec remise sur l'OOS)
#
# Un score ponctuel unique ("62/100") donne une fausse impression de
# précision. En réalité, ce chiffre dépend de l'échantillon de trades OOS
# précis qu'on a observé — un échantillon légèrement différent (même
# stratégie, période légèrement différente) aurait donné un score voisin
# mais pas identique. On estime cette variabilité en ré-échantillonnant les
# trades OOS avec remise (bootstrap classique, taille identique) et en
# recalculant le score complet à chaque réplique. On restitue ensuite la
# médiane et l'intervalle 10e-90e percentile.
# ============================================================

def calculer_fourchette_score(fenetres, rng, n_replicates=N_CI_REPLICATES, n_perms_inner=N_CI_PERMS_INNER):
    """
    Bootstrap PAR BLOCS DE FENÊTRES plutôt que par trade individuel : à
    chaque tirage, on choisit aléatoirement (avec remise) N fenêtres
    parmi les N réelles, en conservant leur contenu IS/OOS intact et leur
    identité de fenêtre.
    Pourquoi ce choix : un rééchantillonnage trade par trade détruit toute
    structure temporelle, donc ne peut par construction jamais reproduire
    la pénalité de cohérence inter-fenêtres du vrai calcul (Pilier 1) ni
    la vraie sensibilité aux trades rares/concentrés (effet loterie) —
    deux limites concrètement observées lors de tests avec de vraies
    données. Rééchantillonner des fenêtres entières (bootstrap par blocs,
    une technique standard pour les données structurées/temporelles)
    préserve cette structure : chaque tirage utilise les vrais ratios
    IS/OOS de vraies fenêtres, donc la pénalité de cohérence s'applique
    naturellement quand elle doit s'appliquer, sans correctif externe.
    """
    n_fenetres = len(fenetres)
    n_oos_total = sum(len(f["oos_trades"]) for f in fenetres) if fenetres else 0
    if n_fenetres == 0 or n_oos_total < MIN_TRADES_ABSOLUTE:
        return {"median": 0.0, "p10": 0.0, "p90": 0.0}

    scores = np.empty(n_replicates)
    for i in range(n_replicates):
        indices_tires = rng.integers(0, n_fenetres, size=n_fenetres)
        fenetres_bootstrap = [fenetres[idx] for idx in indices_tires]
        oos_resample = np.concatenate([f["oos_trades"] for f in fenetres_bootstrap])

        if len(oos_resample) == 0:
            scores[i] = 0
            continue

        # Conserver les fenêtres comme segments pour les métriques
        # path-dependent : un bootstrap ne doit pas recréer artificiellement
        # une continuité entre deux blocs temporels distincts.
        segments_bootstrap = [f["oos_trades"] for f in fenetres_bootstrap]
        mc_r = executer_monte_carlo(
            oos_resample, rng, num_simulations=n_perms_inner, segments=segments_bootstrap
        )
        p2_r = analyser_pilier_is_oos(fenetres_bootstrap, oos_resample)
        p3_r = analyser_pilier_risque_drawdown(oos_resample, segments=segments_bootstrap)
        p4_r = analyser_pilier_concentration(oos_resample)

        total = mc_r["score"] + p2_r["score"] + p3_r["score"] + p4_r["score"]
        scores[i] = max(0, min(100, total))

    return {
        "median": round(float(np.median(scores)), 1),
        "p10": round(float(np.percentile(scores, 10)), 1),
        "p90": round(float(np.percentile(scores, 90)), 1)
    }


# ============================================================
# FOURCHETTE BOOTSTRAP SUR DES MÉTRIQUES SPÉCIFIQUES (PF, Winrate, MDD%)
#
# Sert de référence pour la comparaison forward test : plutôt que des
# seuils de tolérance fixes et arbitraires (ex: "PF forward >= 70% du PF
# backtest"), on rééchantillonne les trades OOS d'origine pour obtenir la
# variabilité NATURELLE de chaque métrique — une stratégie dont l'OOS est
# petit ou volatil a une fourchette large (tolérante), une stratégie très
# stable a une fourchette étroite (stricte). Le forward test est alors
# jugé par rapport à ce que sa propre variabilité d'origine autorise, pas
# par rapport à un pourcentage universel qui ne veut rien dire pour tout
# le monde à la fois.
# ============================================================

def calculer_fourchette_metriques(oos_trades, capital_ref, rng, n_replicates=N_CI_REPLICATES):
    n = len(oos_trades)
    if n < MIN_TRADES_ABSOLUTE:
        return None

    pfs = np.empty(n_replicates)
    winrates = np.empty(n_replicates)
    mdds = np.empty(n_replicates)

    for i in range(n_replicates):
        idx = rng.integers(0, n, size=n)
        resample = oos_trades[idx]
        pfs[i] = calculer_profit_factor(resample)
        gains = resample[resample > 0]
        winrates[i] = (len(gains) / n * 100) if n > 0 else 0.0
        mdds[i] = calculer_max_drawdown_pct(resample, capital_ref)

    return {
        "profit_factor": {"p10": round(float(np.percentile(pfs, 10)), 2), "p90": round(float(np.percentile(pfs, 90)), 2)},
        "winrate_pct": {"p10": round(float(np.percentile(winrates, 10)), 1), "p90": round(float(np.percentile(winrates, 90)), 1)},
        "mdd_pct": {"p10": round(float(np.percentile(mdds, 10)), 1), "p90": round(float(np.percentile(mdds, 90)), 1)},
    }


def calculer_max_drawdown_pct(trades, capital_initial=None):
    """
    Max Drawdown en % du pic d'équity. Sans capital de départ renseigné, le
    pic est calculé sur l'équity cumulée depuis 0 — approximatif au début de
    la période (peu de "coussin"), mais reste la meilleure référence
    disponible sans info de compte. Avec capital_initial, le calcul est
    exact par rapport au solde réel du compte.
    """
    cumul = np.cumsum(trades)
    if capital_initial and capital_initial > 0:
        equity = capital_initial + cumul
    else:
        equity = cumul
    peak = np.maximum.accumulate(equity)
    with np.errstate(divide='ignore', invalid='ignore'):
        dd_pct = np.where(peak > 0, (peak - equity) / peak * 100, 0.0)
    return float(np.max(dd_pct)) if len(dd_pct) > 0 else 0.0


def calculer_statistiques_detaillees(trades, capital_initial=None):
    """
    Statistiques de backtest façon plateforme de trading (MT4/MT5/cTrader) :
    winrate, trade moyen, plus gros gain/perte, séries consécutives, etc.
    Complète les 4 piliers du score avec le niveau de détail qu'un trader
    attend de voir avant de faire confiance à un score unique.
    """
    n = len(trades)
    if n == 0:
        return {
            "n_trades": 0, "n_gagnants": 0, "n_perdants": 0, "winrate_pct": 0.0,
            "trade_moyen": 0.0, "gain_moyen": 0.0, "perte_moyenne": 0.0,
            "plus_gros_gain": 0.0, "plus_grosse_perte": 0.0,
            "total_gains": 0.0, "total_pertes": 0.0,
            "max_gains_consecutifs": 0, "max_pertes_consecutives": 0,
            "mdd_dollars": 0.0, "mdd_pct": 0.0,
            "profit_net_pct": 0.0, "gain_moyen_pct": 0.0, "perte_moyenne_pct": 0.0,
            "trade_moyen_pct": 0.0, "plus_gros_gain_pct": 0.0, "plus_grosse_perte_pct": 0.0,
            "total_gains_pct": 0.0, "total_pertes_pct": 0.0,
        }

    gains = trades[trades > 0]
    pertes = trades[trades < 0]
    n_gagnants = len(gains)
    n_perdants = len(pertes)
    winrate = (n_gagnants / n * 100) if n > 0 else 0.0

    max_gains_conseq = 0
    max_pertes_conseq = 0
    cur_g = 0
    cur_p = 0
    for t in trades:
        if t > 0:
            cur_g += 1
            cur_p = 0
            max_gains_conseq = max(max_gains_conseq, cur_g)
        elif t < 0:
            cur_p += 1
            cur_g = 0
            max_pertes_conseq = max(max_pertes_conseq, cur_p)
        else:
            cur_g = 0
            cur_p = 0

    # Métriques en % : nécessitent un vrai capital de référence pour avoir un
    # sens. Contrairement au Max DD (qui peut s'appuyer sur le pic d'équity
    # cumulée comme repère imparfait mais défini), un "% de profit" sans
    # capital connu n'a AUCUNE valeur fiable — on retourne None plutôt
    # qu'un chiffre qui semblerait précis mais serait trompeur. En pratique,
    # le capital est désormais obligatoire pour lancer une analyse, donc ce
    # None ne devrait plus jamais se produire — gardé par prudence.
    if capital_initial and capital_initial > 0:
        profit_net_pct = round((float(np.sum(trades)) / capital_initial) * 100, 2)
        gain_moyen_pct = round((float(np.mean(gains)) / capital_initial * 100), 2) if len(gains) > 0 else 0.0
        perte_moyenne_pct = round((float(np.mean(pertes)) / capital_initial * 100), 2) if len(pertes) > 0 else 0.0
        trade_moyen_pct = round((float(np.mean(trades)) / capital_initial * 100), 2)
        plus_gros_gain_pct = round((float(np.max(trades)) / capital_initial * 100), 2)
        plus_grosse_perte_pct = round((float(np.min(trades)) / capital_initial * 100), 2)
        total_gains_pct = round((float(np.sum(gains)) / capital_initial * 100), 2) if len(gains) > 0 else 0.0
        total_pertes_pct = round((float(np.sum(pertes)) / capital_initial * 100), 2) if len(pertes) > 0 else 0.0
    else:
        profit_net_pct = gain_moyen_pct = perte_moyenne_pct = None
        trade_moyen_pct = plus_gros_gain_pct = plus_grosse_perte_pct = None
        total_gains_pct = total_pertes_pct = None

    return {
        "n_trades": int(n),
        "n_gagnants": int(n_gagnants),
        "n_perdants": int(n_perdants),
        "winrate_pct": round(winrate, 1),
        "trade_moyen": round(float(np.mean(trades)), 2),
        "gain_moyen": round(float(np.mean(gains)), 2) if len(gains) > 0 else 0.0,
        "perte_moyenne": round(float(np.mean(pertes)), 2) if len(pertes) > 0 else 0.0,
        "plus_gros_gain": round(float(np.max(trades)), 2),
        "plus_grosse_perte": round(float(np.min(trades)), 2),
        "total_gains": round(float(np.sum(gains)), 2) if len(gains) > 0 else 0.0,
        "total_pertes": round(float(np.sum(pertes)), 2) if len(pertes) > 0 else 0.0,
        "max_gains_consecutifs": int(max_gains_conseq),
        "max_pertes_consecutives": int(max_pertes_conseq),
        "profit_net_pct": profit_net_pct,
        "gain_moyen_pct": gain_moyen_pct,
        "perte_moyenne_pct": perte_moyenne_pct,
        "trade_moyen_pct": trade_moyen_pct,
        "plus_gros_gain_pct": plus_gros_gain_pct,
        "plus_grosse_perte_pct": plus_grosse_perte_pct,
        "total_gains_pct": total_gains_pct,
        "total_pertes_pct": total_pertes_pct,
        "mdd_dollars": round(calculer_max_drawdown(trades), 2),
        "mdd_pct": round(calculer_max_drawdown_pct(trades, capital_initial), 2),
    }


# ============================================================
# NORMALISATION / INTÉGRITÉ DES TRADES IMPORTÉS
# ============================================================

def _parse_dates_robuste(values, dayfirst_default=None):
    """Parse les dates sans imposer silencieusement une locale unique.

    Les dates ISO (YYYY-MM-DD...) sont non ambiguës. Les formats à slash
    dont les deux premiers champs sont <= 12 restent potentiellement
    ambigus (MM/DD vs DD/MM) : on les signale au lieu de réordonner les
    trades sur une interprétation incertaine.
    """
    serie = pd.Series(list(values), dtype="object")
    textes = serie.fillna("").astype(str).str.strip()
    ambigus = []
    for i, txt in textes.items():
        m = re.match(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})(?:\s|$)", txt)
        if m and int(m.group(1)) <= 12 and int(m.group(2)) <= 12:
            # ISO yyyy-mm-dd ne passe pas ici ; seuls les formats courts
            # à deux champs numériques sont considérés ambigus.
            ambigus.append(i)

    parsed = pd.Series(pd.NaT, index=serie.index, dtype="datetime64[ns]")
    for i, txt in textes.items():
        if not txt:
            continue
        try:
            # MT5 peut fournir nativement YYYY.MM.DD[ HH:MM[:SS]].
            # Ce format est intrinsèquement non ambigu : parser explicitement
            # l'année/mois/jour avant toute heuristique de locale.
            if re.match(r"^\d{4}\.\d{2}\.\d{2}(?:\s|$)", txt):
                dt = pd.to_datetime(txt, format="%Y.%m.%d %H:%M:%S", errors="coerce")
                if pd.isna(dt):
                    dt = pd.to_datetime(txt, format="%Y.%m.%d %H:%M", errors="coerce")
                if pd.isna(dt):
                    dt = pd.to_datetime(txt, format="%Y.%m.%d", errors="coerce")
            elif re.match(r"^\d{4}-\d{2}-\d{2}", txt):
                dt = pd.to_datetime(txt, errors="coerce", dayfirst=False)
            elif i in ambigus and dayfirst_default is not None:
                dt = pd.to_datetime(txt, errors="coerce", dayfirst=dayfirst_default)
            elif i in ambigus:
                # Locale inconnue : on parse pour pouvoir informer, mais
                # l'ordre ne sera pas modifié sur une date ambiguë.
                dt = pd.to_datetime(txt, errors="coerce", dayfirst=True)
            else:
                dt = pd.to_datetime(txt, errors="coerce", dayfirst=True)

            # Les exports peuvent mélanger dates naïves et timestamps avec
            # offset (ex. +01:00 / Z). Le modèle interne utilise des dates
            # naïves pour le tri ; quand un offset est fourni, on convertit
            # d'abord en UTC afin de ne pas comparer deux heures locales
            # comme si elles appartenaient au même fuseau.
            if pd.isna(dt):
                parsed.loc[i] = pd.NaT
            elif getattr(dt, "tzinfo", None) is not None:
                parsed.loc[i] = dt.tz_convert("UTC").tz_localize(None)
            else:
                parsed.loc[i] = dt
        except Exception:
            parsed.loc[i] = pd.NaT
    return parsed, len(ambigus)


def normaliser_integrite_import(trades_array, details_list, dayfirst_default=None):
    """Valide l'alignement et impose un ordre chronologique sûr si possible.

    Retourne (trades_array, details_list, warnings). Aucun trade n'est
    supprimé à cause d'un P&L nul : 0 est un résultat valide. Les lignes
    sans date ne sont pas supprimées, mais rendent les métriques dépendantes
    de l'ordre fourni par l'export et sont signalées.
    """
    warnings = []
    arr = np.asarray(trades_array, dtype=float)
    details = list(details_list or [])

    if len(arr) != len(details):
        raise ValueError("Le nombre de trades et le nombre de lignes détaillées ne correspondent pas.")
    if len(arr) == 0:
        return arr, details, warnings
    if not np.isfinite(arr).all():
        raise ValueError("Le fichier contient des P&L non numériques après parsing.")

    dates = [d.get("date") if isinstance(d, dict) else None for d in details]
    parsed, n_ambigus = _parse_dates_robuste(dates, dayfirst_default=dayfirst_default)
    n_valides = int(parsed.notna().sum())

    if n_valides == 0:
        warnings.append({
            "level": "warning", "code": "dates_absentes",
            "message": "Aucune date de trade exploitable n'a été trouvée. Les métriques dépendantes de l'ordre des trades (drawdown, Monte Carlo de trajectoire, fenêtres temporelles) utilisent donc l'ordre fourni par l'export."
        })
        return arr, details, warnings

    if n_valides < len(arr):
        warnings.append({
            "level": "warning", "code": "dates_partielles",
            "message": f"{len(arr) - n_valides} trade(s) n'ont pas de date exploitable. L'ordre de ces lignes ne peut pas être vérifié automatiquement."
        })
        return arr, details, warnings

    if n_ambigus and dayfirst_default is None:
        warnings.append({
            "level": "warning", "code": "dates_ambiguës",
            "message": "Certaines dates utilisent un format à slash potentiellement ambigu (MM/JJ vs JJ/MM). Xtrunn n'a pas réordonné ces trades automatiquement car la locale de l'export n'est pas connue."
        })
        return arr, details, warnings

    ordre = np.argsort(parsed.to_numpy(dtype="datetime64[ns]"), kind="stable")
    if not np.array_equal(ordre, np.arange(len(arr))):
        arr = arr[ordre]
        details = [details[int(i)] for i in ordre]
        warnings.append({
            "level": "warning", "code": "ordre_chronologique_corrige",
            "message": "Les trades n'étaient pas dans l'ordre chronologique. Xtrunn les a réordonnés selon leur date avant de calculer les métriques dépendantes de la trajectoire."
        })

    return arr, details, warnings


# ============================================================
# PARSING MT5
# ============================================================

def _classifier_semantique_pnl(col_profit, plateforme=None):
    """Décrit la sémantique probable de la colonne P&L sans la présenter comme une certitude broker.

    La sélection du champ reste basée sur le fichier réellement fourni. Cette fonction
    sert uniquement à rendre explicite ce que Xtrunn sait / ne sait pas du P&L et de ses coûts.
    """
    col = str(col_profit or "").strip().lower()
    p = str(plateforme or "").strip().lower()
    if p == "freqtrade" or "profit_abs" in col:
        return {"source": col_profit, "nature": "net_probable", "couts_deja_inclus": "probable",
                "preuve": "schéma natif Freqtrade / profit_abs", "certitude": "format_plateforme"}
    if any(k in col for k in ("net profit", "net pnl", "net p&l")) or col == "net" or col.startswith("net "):
        return {"source": col_profit, "nature": "net", "couts_deja_inclus": "probable",
                "preuve": "nom_de_colonne", "certitude": "nom_champ"}
    if "gross" in col:
        return {"source": col_profit, "nature": "brut", "couts_deja_inclus": "non",
                "preuve": "nom_de_colonne", "certitude": "nom_champ"}
    return {"source": col_profit, "nature": "profit_non_qualifie", "couts_deja_inclus": "inconnu",
            "preuve": "nom_de_colonne", "certitude": "insuffisante"}


def _choisir_colonne_profit(columns, prefer_net=True, allow_gross=True):
    """Choisit une colonne de P&L par trade, sans confondre une métrique de résumé.

    Les exports de plateformes peuvent contenir dans le même tableau des
    colonnes telles que ``Profit Factor``, ``Cumulative Profit`` ou
    ``Total Net Profit``. Elles contiennent le mot ``profit`` mais ne sont
    pas un P&L individuel par trade. Elles doivent donc être exclues avant
    le classement des vrais champs P&L.
    """
    cols = [str(c).strip().lower() for c in columns]
    exclues = (
        "commission", "commissions", "swap", "fee", "fees", "frais", "tax",
        "%", "percent", "pourcent",
        "profit factor", "profit-factor", "profitfactor",
        "cumulative profit", "cumulative pnl", "cumulative p&l",
        "total profit", "total pnl", "total p&l", "total net profit",
        "average profit", "avg profit", "mean profit",
        "max profit", "maximum profit", "min profit", "minimum profit",
        "profit ratio", "profit/risk", "profit risk",
        "expectancy", "payoff ratio", "win/loss ratio",
    )
    candidates = []
    for col in cols:
        if any(x in col for x in exclues):
            continue
        # Champs explicitement nets : priorité maximale.
        if col in ("net", "net pnl", "net p&l", "net profit"):
            candidates.append((0, col))
        elif prefer_net and any(k in col for k in ("net pnl", "net p&l", "net profit")):
            candidates.append((1, col))
        # Champs génériques de P&L par trade.
        elif col in ("profit", "pnl", "p&l", "profit/loss", "profit loss"):
            candidates.append((2, col))
        elif any(k in col for k in ("profit", "pnl", "p&l")):
            candidates.append((3, col))
        elif prefer_net and "net" in col and not any(x in col for x in ("factor", "ratio")):
            candidates.append((4, col))
        elif allow_gross and col in ("gross", "gross pnl", "gross p&l", "gross profit"):
            candidates.append((5, col))
        elif allow_gross and "gross" in col and "profit" in col:
            candidates.append((6, col))
    if not candidates:
        return None
    return min(candidates, key=lambda x: x[0])[1]


# ============================================================
# PARSING MT5
# ============================================================

def lire_trades_mt5(contents, filename):
    """
    Retourne (trades_array, trades_detail, erreur).
    trades_detail est une liste de dicts {ticket, date, type, profit} alignée
    sur trades_array, dans le même ordre — utilisée pour le journal des
    trades affiché à l'utilisateur (transparence : preuve que le logiciel
    lit bien ses vraies données, pas une simulation).
    """
    filename = filename.lower()
    try:
        if filename.endswith('.csv'):
            try:
                df_raw = pd.read_csv(io.BytesIO(contents), header=None, sep=',', encoding='utf-8')
                if len(df_raw.columns) <= 1:
                    df_raw = pd.read_csv(io.BytesIO(contents), header=None, sep=';', encoding='utf-8')
            except Exception:
                df_raw = pd.read_csv(io.BytesIO(contents), header=None, sep=';', encoding='latin1')
        elif filename.endswith(('.xlsx', '.xls')):
            df_raw = pd.read_excel(io.BytesIO(contents), header=None)
        else:
            return None, None, "Format non supporté (utilisez CSV ou Excel)."
    except Exception:
        return None, None, "Impossible de lire le fichier."

    ligne_entete = None
    for index, row in df_raw.iterrows():
        row_str = " ".join([str(val).strip().lower() for val in row.values])
        if 'profit' in row_str and ('heure' in row_str or 'time' in row_str or 'date' in row_str or 'opération' in row_str or 'ticket' in row_str or 'deal' in row_str or 'order' in row_str):
            ligne_entete = index
            break

    if ligne_entete is None:
        return None, None, "Tableau des transactions MT5 introuvable dans le rapport."

    if filename.endswith('.csv'):
        try:
            df = pd.read_csv(io.BytesIO(contents), header=ligne_entete, sep=',', encoding='utf-8')
            if len(df.columns) <= 1:
                df = pd.read_csv(io.BytesIO(contents), header=ligne_entete, sep=';', encoding='utf-8')
        except Exception:
            df = pd.read_csv(io.BytesIO(contents), header=ligne_entete, sep=';', encoding='latin1')
    else:
        df = pd.read_excel(io.BytesIO(contents), header=ligne_entete)

    df.columns = [str(col).strip().lower() for col in df.columns]

    col_profit = _choisir_colonne_profit(df.columns, prefer_net=True, allow_gross=True)
    col_dir = None
    col_entry = None
    col_date = None
    col_ticket = None
    col_deal = None
    col_position = None
    col_volume = None
    col_sl, col_tp, col_open_price = None, None, None
    col_symbole = None
    for col in df.columns:
        if ('direction' in col or 'type' in col or col == 'side') and col_dir is None:
            col_dir = col
        if any(k in col for k in ['entry', 'entrée', 'entree']) and col_entry is None:
            col_entry = col
        if any(k in col for k in ['closing time', 'close time', 'heure de clôture', 'heure de cloture', 'date de clôture', 'date de cloture', 'exit time', 'closing date']) and col_date is None:
            col_date = col
        elif any(k in col for k in ['heure', 'time', 'date']) and col_date is None:
            col_date = col
        if any(k in col for k in ['ticket', 'order', '#']) and col_ticket is None:
            col_ticket = col
        if 'deal' in col and col_deal is None:
            col_deal = col
        if 'position' in col and 'id' in col and col_position is None:
            col_position = col
        if any(k in col for k in ['volume', 'lots', 'lot']) and col_volume is None:
            col_volume = col
        if any(k in col for k in ['s/l', 's / l', 'stop loss']) and col_sl is None:
            col_sl = col
        if any(k in col for k in ['t/p', 't / p', 'take profit']) and col_tp is None:
            col_tp = col
        if any(k in col for k in ['symbol', 'symbole', 'item', 'instrument']) and col_symbole is None:
            col_symbole = col
    # Même prudence que pour MT4 : le prix d'ouverture partage souvent le
    # même en-tête "Price" que le prix de clôture -> on ne prend la
    # première colonne "price"/"prix" apparaissant AVANT S/L que si ce
    # repère est net, jamais une supposition à l'aveugle.
    cols_list = list(df.columns)
    if col_sl is not None and col_sl in cols_list:
        idx_sl = cols_list.index(col_sl)
        for col in cols_list[:idx_sl]:
            if 'price' in col or 'prix' in col:
                col_open_price = col
                break

    if not col_profit:
        return None, None, "Colonne de P&L introuvable dans le rapport MT5. Recherchez une colonne 'Net', 'Profit' ou 'P&L'."

    df_clean = df.copy()
    df_clean[col_profit] = df_clean[col_profit].apply(extraire_nombre)
    df_clean = df_clean.dropna(subset=[col_profit])

    # MT5 peut exporter plusieurs granularités : ordres, deals/exécutions,
    # ou historique de positions. Le profit exploitable doit venir des
    # lignes de sortie/fermeture, pas des entrées ni des lignes récapitulatives.
    # On privilégie les marqueurs explicites quand ils existent.
    df_trades = df_clean
    granularite = 'inconnue'
    filtre_mode = 'pnl_numerique'

    if col_entry:
        entry_txt = df_clean[col_entry].astype(str).str.strip().str.lower()
        mask_out = entry_txt.str.contains(r'out|close|sortie|clôture|cloture', na=False, regex=True)
        mask_in = entry_txt.str.contains(r'in|entry|entrée|entree', na=False, regex=True)
        if mask_out.any():
            df_trades = df_clean[mask_out]
            granularite = 'deal'
            filtre_mode = 'entry_out'
        elif mask_in.any() and (mask_in.sum() < len(df_clean)):
            df_trades = df_clean[~mask_in]
            granularite = 'deal'
            filtre_mode = 'entry_non_in'
        else:
            granularite = 'deal_ambigu'

    # Les rapports MT5 peuvent contenir des lignes de total/balance dans
    # le même tableau. Dans un export de deals, un Deal numérique + un
    # symbole non vide constituent un marqueur structurel fiable d'une
    # exécution réelle ; les lignes de résumé n'ont généralement pas ces
    # deux attributs.
    if col_deal and col_symbole:
        deal_num = df_trades[col_deal].apply(extraire_nombre)
        symbole_ok = df_trades[col_symbole].notna() & (df_trades[col_symbole].astype(str).str.strip() != '')
        df_trades = df_trades[deal_num.notna() & symbole_ok]

    if col_dir and len(df_trades) == len(df_clean):
        type_txt = df_clean[col_dir].astype(str).str.strip().str.lower()
        mask_out_type = type_txt.str.contains(r'out|close|sortie|clôture|cloture', na=False, regex=True)
        # Sur un historique de deals MT5, Type peut contenir buy/sell sans
        # distinguer l'entrée. Si des lignes OUT existent, elles sont les
        # clôtures économiques. Les lignes balance/deposit/etc. sont exclues.
        mask_non_trade = type_txt.str.contains(r'balance|credit|deposit|withdraw|charge|commission|bonus|transfer|dividend|correction', na=False, regex=True)
        if mask_out_type.any():
            df_trades = df_clean[mask_out_type & ~mask_non_trade]
            granularite = 'deal'
            filtre_mode = 'type_out'
        elif mask_non_trade.any():
            df_trades = df_clean[~mask_non_trade]
            filtre_mode = 'type_non_balance'

    # Une colonne Order/Deal/Position seule ne suffit pas à déterminer la
    # granularité. On ne déduplique donc pas par identifiant : les clôtures
    # partielles peuvent partager un même ordre/position. On retire seulement
    # les doublons strictement identiques (ticket + P&L + date).
    col_id_dedupe = col_deal or col_ticket or col_position
    df_trades, doublons_exacts = _dedoublonner_trades_exacts(
        df_trades, col_profit, col_id_dedupe, col_date
    )

    # Si l'export ressemble à un historique de deals mais ne fournit aucun
    # marqueur d'entrée/sortie exploitable, il est dangereux d'assimiler
    # chaque ligne buy/sell à un trade clôturé : une entrée peut être comptée
    # comme un résultat. On refuse alors l'import plutôt que de fabriquer un
    # historique faux. Un rapport de positions/trades explicite n'est pas
    # concerné par ce garde-fou.
    if col_deal and granularite in {'inconnue', 'deal_ambigu'} and col_entry is None and col_dir:
        type_txt_final = df_trades[col_dir].astype(str).str.strip().str.lower()
        if type_txt_final.str.contains(r'^(?:buy|sell|achat|vente)$', na=False, regex=True).all():
            return None, None, "Export MT5 de type 'deals' détecté mais les marqueurs d'entrée/sortie sont absents ou ambigus. Exportez l'historique avec la colonne 'Entry' (In/Out) afin d'éviter de compter une entrée comme un trade clôturé."

    if len(df_trades) < MIN_TRADES_ABSOLUTE:
        return None, None, f"Pas assez de trades valides trouvés (minimum {MIN_TRADES_ABSOLUTE} requis)."

    trades_filtres = df_trades[col_profit].values

    trades_detail = []
    for _, row in df_trades.iterrows():
        volume = extraire_nombre(row[col_volume]) if col_volume else np.nan
        sl = extraire_nombre(row[col_sl]) if col_sl else np.nan
        tp = extraire_nombre(row[col_tp]) if col_tp else np.nan
        open_price = extraire_nombre(row[col_open_price]) if col_open_price else np.nan
        trades_detail.append({
            "ticket": str(row[col_ticket]) if col_ticket else None,
            "date": str(row[col_date]) if col_date else None,
            "type": str(row[col_dir]) if col_dir else None,
            "profit": round(float(row[col_profit]), 2),
            "volume": round(float(volume), 4) if not pd.isna(volume) else None,
            "sl": round(float(sl), 5) if not pd.isna(sl) and sl != 0 else None,
            "tp": round(float(tp), 5) if not pd.isna(tp) and tp != 0 else None,
            "open_price": round(float(open_price), 5) if not pd.isna(open_price) and open_price != 0 else None,
            "symbole": str(row[col_symbole]).strip() if col_symbole and pd.notna(row[col_symbole]) else None,
        })
    pnl_semantique = _classifier_semantique_pnl(col_profit if 'col_profit' in locals() else 'profit_abs', "mt5")
    for _d in trades_detail:
        _d["pnl_source"] = pnl_semantique["source"]
        _d["pnl_nature"] = pnl_semantique["nature"]
        _d["pnl_couts_deja_inclus"] = pnl_semantique["couts_deja_inclus"]
        _d["mt5_granularite"] = granularite
        _d["mt5_filtre_execution"] = filtre_mode
        _d["mt5_doublons_exacts_supprimes"] = doublons_exacts

    return trades_filtres, trades_detail, None



# ============================================================
# UTILITAIRE PARTAGÉ — extraction robuste d'un nombre depuis une
# cellule qui peut contenir du formatage (devise, séparateurs de
# milliers, %) selon la plateforme d'export.
# ============================================================

def _dedoublonner_trades_exacts(df, col_profit, col_ticket=None, col_date=None):
    """Supprime uniquement les doublons manifestement identiques.

    On ne fusionne JAMAIS deux lignes partageant seulement le même ticket :
    un même identifiant peut légitimement apparaître plusieurs fois lors
    d'une exécution/fermeture partielle. Le dédoublonnage est volontairement
    conservateur et ne s'applique que lorsque les champs disponibles sont
    identiques ligne pour ligne (ticket + profit + date si disponible).
    """
    if df.empty or col_ticket is None or col_ticket not in df.columns:
        return df, 0
    subset = [col_ticket, col_profit]
    if col_date is not None and col_date in df.columns:
        subset.append(col_date)
    dup_mask = df.duplicated(subset=subset, keep='first')
    removed = int(dup_mask.sum())
    if removed:
        return df.loc[~dup_mask].copy(), removed
    return df, 0


def extraire_nombre(valeur):
    """Parse un nombre exporté avec conventions FR/EU/US.

    Exemples : ``1 234,56`` -> 1234.56, ``1,234.56`` -> 1234.56,
    ``1.234,56`` -> 1234.56. Les symboles monétaires/% sont ignorés.
    """
    if pd.isna(valeur):
        return np.nan
    s = str(valeur).strip().replace('\xa0', '').replace(' ', '').replace("'", '')
    # Garder signe, chiffres et séparateurs décimaux/milliers.
    match = re.search(r'[-+]?\d[\d.,]*', s)
    if not match:
        return np.nan
    token = match.group(0)
    if ',' in token and '.' in token:
        # Le dernier séparateur est normalement le séparateur décimal.
        decimal_sep = ',' if token.rfind(',') > token.rfind('.') else '.'
        thousands_sep = '.' if decimal_sep == ',' else ','
        token = token.replace(thousands_sep, '').replace(decimal_sep, '.')
    elif ',' in token:
        # Virgule seule : décimale si 1-2 chiffres suivent, sinon milliers.
        parts = token.split(',')
        if len(parts) == 2 and (len(parts[1]) <= 2 or (len(parts[1]) == 3 and len(parts[0].lstrip('+-')) > 3)):
            token = parts[0] + '.' + parts[1]
        else:
            token = ''.join(parts)
    elif '.' in token:
        parts = token.split('.')
        if len(parts) > 2:
            token = ''.join(parts[:-1]) + '.' + parts[-1]
    try:
        return float(token)
    except ValueError:
        return np.nan


# ============================================================
# PARSING MT4
#
# Export par défaut = rapport HTML (.htm) via clic droit sur l'onglet
# "Report" du Testeur de Stratégies -> "Save as Report" (MT4 n'exporte
# PAS de CSV natif pour ça, contrairement à MT5). Le tableau contient
# une ligne par ouverture ET une ligne par clôture ; seule la ligne de
# clôture a une valeur de Profit renseignée — dropna() sur cette colonne
# suffit à isoler les trades clôturés, sans dépendre de mots-clés de
# type ("close", "s/l", "t/p"...) qui varient selon la langue de MT4.
# ============================================================

def lire_trades_mt4(contents, filename):
    filename_lower = filename.lower()
    tables = None
    try:
        if filename_lower.endswith(('.htm', '.html')):
            tables = pd.read_html(io.BytesIO(contents), header=None)
        elif filename_lower.endswith('.csv'):
            try:
                df_raw = pd.read_csv(io.BytesIO(contents), header=None, sep=',', encoding='utf-8')
                if len(df_raw.columns) <= 1:
                    df_raw = pd.read_csv(io.BytesIO(contents), header=None, sep=';', encoding='utf-8')
            except Exception:
                df_raw = pd.read_csv(io.BytesIO(contents), header=None, sep=';', encoding='latin1')
            tables = [df_raw]
        elif filename_lower.endswith(('.xlsx', '.xls')):
            tables = [pd.read_excel(io.BytesIO(contents), header=None)]
        else:
            return None, None, "Format non supporté pour MT4. Utilisez le rapport HTML exporté via \"Save as Report\", ou un CSV/Excel."
    except Exception:
        return None, None, "Impossible de lire le fichier. Vérifiez qu'il s'agit bien d'un rapport MT4 exporté via \"Save as Report\"."

    df = None
    for candidate in tables:
        cols_lower = [("" if pd.isna(c) else str(c)).strip().lower() for c in candidate.columns]
        if any('profit' in c for c in cols_lower) and any(('open time' in c or 'ticket' in c or 'order' in c) for c in cols_lower):
            df = candidate.copy()
            df.columns = cols_lower
            break

        # Repli : le rapport ne distinguait pas l'en-tête (pas de <th>) ->
        # on scanne les lignes de données pour la retrouver, comme pour MT5.
        for index, row in candidate.iterrows():
            row_str = " ".join([("" if pd.isna(v) else str(v)).strip().lower() for v in row.values])
            if 'profit' in row_str and ('open time' in row_str or 'ticket' in row_str or 'order' in row_str):
                new_header = [("" if pd.isna(v) else str(v)).strip().lower() for v in row.values]
                df = candidate.iloc[index + 1:].copy()
                df.columns = new_header
                break
        if df is not None:
            break

    if df is None:
        return None, None, "Tableau des transactions introuvable dans le rapport MT4. Vérifiez qu'il s'agit du rapport complet (\"Save as Report\"), pas d'un résumé."

    col_profit = _choisir_colonne_profit(df.columns, prefer_net=True, allow_gross=True)
    col_type, col_date, col_ticket, col_volume = None, None, None, None
    col_sl, col_tp, col_open_price = None, None, None
    col_symbole = None
    for col in df.columns:
        if 'type' in col and col_type is None:
            col_type = col
        if any(k in col for k in ['ticket', 'order', '#']) and col_ticket is None:
            col_ticket = col
        if col == 'size' and col_volume is None:
            col_volume = col
        if ('s/l' in col or col == 'sl') and col_sl is None:
            col_sl = col
        if ('t/p' in col or col == 'tp') and col_tp is None:
            col_tp = col
        if any(k in col for k in ['symbol', 'symbole', 'item', 'instrument']) and col_symbole is None:
            col_symbole = col
    # Le prix d'ouverture partage souvent le même en-tête "Price" que le prix
    # de clôture (deux colonnes distinctes, même nom) — on ne prend la
    # première colonne "price" que si elle apparaît AVANT S/L dans l'ordre
    # des colonnes (le prix d'ouverture précède toujours S/L/T/P dans la
    # mise en page standard du rapport). Si ce repère n'est pas clair, on
    # renonce plutôt que de risquer de confondre ouverture et clôture.
    cols_list = list(df.columns)
    if col_sl is not None and col_sl in cols_list:
        idx_sl = cols_list.index(col_sl)
        for i, col in enumerate(cols_list[:idx_sl]):
            if 'price' in col or 'prix' in col:
                col_open_price = col
                break
    # Priorité de date : le rapport MT4 nomme la colonne d'heure de CLÔTURE
    # juste "Time" (distincte de "Open Time", vide sur les lignes retenues
    # puisqu'on filtre sur les lignes ayant un Profit renseigné).
    for col in df.columns:
        if col == 'close time':
            col_date = col
            break
    if col_date is None:
        for col in df.columns:
            if col == 'time':
                col_date = col
                break
    if col_date is None:
        for col in df.columns:
            if 'open time' in col:
                col_date = col
                break
    if col_date is None:
        for col in df.columns:
            if 'date' in col:
                col_date = col
                break

    if not col_profit:
        return None, None, "Colonne de P&L introuvable dans le rapport MT4. Recherchez une colonne 'Net', 'Profit' ou 'P&L'."

    df_clean = df.copy()
    df_clean[col_profit] = df_clean[col_profit].apply(extraire_nombre)
    df_clean = df_clean.dropna(subset=[col_profit])
    df_clean, _doublons_exacts = _dedoublonner_trades_exacts(df_clean, col_profit, col_ticket, col_date)

    if len(df_clean) < MIN_TRADES_ABSOLUTE:
        return None, None, f"Pas assez de trades valides trouvés (minimum {MIN_TRADES_ABSOLUTE} requis)."

    trades_filtres = df_clean[col_profit].values

    trades_detail = []
    for _, row in df_clean.iterrows():
        volume = extraire_nombre(row[col_volume]) if col_volume else np.nan
        sl = extraire_nombre(row[col_sl]) if col_sl else np.nan
        tp = extraire_nombre(row[col_tp]) if col_tp else np.nan
        open_price = extraire_nombre(row[col_open_price]) if col_open_price else np.nan
        trades_detail.append({
            "ticket": str(row[col_ticket]) if col_ticket else None,
            "date": str(row[col_date]) if col_date else None,
            "type": str(row[col_type]) if col_type else None,
            "profit": round(float(row[col_profit]), 2),
            "volume": round(float(volume), 4) if not pd.isna(volume) else None,
            "sl": round(float(sl), 5) if not pd.isna(sl) and sl != 0 else None,
            "tp": round(float(tp), 5) if not pd.isna(tp) and tp != 0 else None,
            "open_price": round(float(open_price), 5) if not pd.isna(open_price) and open_price != 0 else None,
            "symbole": str(row[col_symbole]).strip() if col_symbole and pd.notna(row[col_symbole]) else None,
        })

        pnl_semantique = _classifier_semantique_pnl(col_profit if 'col_profit' in locals() else 'profit_abs', "mt4")
    for _d in trades_detail:
        _d["pnl_source"] = pnl_semantique["source"]
        _d["pnl_nature"] = pnl_semantique["nature"]
        _d["pnl_couts_deja_inclus"] = pnl_semantique["couts_deja_inclus"]

    return trades_filtres, trades_detail, None


# ============================================================
# PARSING CTRADER
#
# Export via l'onglet "History" (backtest cBot) -> clic droit ->
# "Export to Excel". Les colonnes sont CHOISIES par l'utilisateur dans
# cTrader (case à cocher par champ), donc les noms varient — recherche
# par mots-clés avec ordre de priorité plutôt que noms de colonnes fixes.
# Contrairement à MT4/TradingView, une ligne = un trade déjà clôturé
# (pas de ligne d'ouverture séparée), donc pas de filtre entrée/sortie.
# ============================================================

def lire_trades_ctrader(contents, filename):
    filename_lower = filename.lower()
    try:
        if filename_lower.endswith('.csv'):
            try:
                df = pd.read_csv(io.BytesIO(contents), sep=',', encoding='utf-8')
                if len(df.columns) <= 1:
                    df = pd.read_csv(io.BytesIO(contents), sep=';', encoding='utf-8')
            except Exception:
                df = pd.read_csv(io.BytesIO(contents), sep=';', encoding='latin1')
        elif filename_lower.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            return None, None, "Format non supporté pour cTrader (utilisez CSV ou Excel exporté depuis l'onglet History)."
    except Exception:
        return None, None, "Impossible de lire le fichier."

    df.columns = [str(c).strip().lower() for c in df.columns]

    col_profit = _choisir_colonne_profit(df.columns, prefer_net=True, allow_gross=True)

    if not col_profit:
        return None, None, "Colonne de profit introuvable dans l'export cTrader. Assurez-vous d'inclure une colonne 'Net' ou 'Profit' lors de l'export."

    col_type, col_date, col_ticket, col_volume = None, None, None, None
    col_symbole = None
    for col in df.columns:
        if ('direction' in col or 'type' in col) and col_type is None:
            col_type = col
        if any(k in col for k in ['position id', 'deal id', 'order id', 'ticket', 'id']) and col_ticket is None:
            col_ticket = col
        if any(k in col for k in ['volume', 'quantity', 'lots']) and col_volume is None:
            col_volume = col
        if any(k in col for k in ['symbol', 'symbole', 'instrument']) and col_symbole is None:
            col_symbole = col
    # Priorité de date : clôture d'abord (plus pertinente : quand le trade
    # a effectivement impacté l'équity), ouverture en repli. cTrader
    # localise l'intitulé des colonnes selon la langue de l'interface —
    # confirmé sur un vrai export en français ("Heure de clôture"), d'où
    # les variantes françaises en plus des anglaises.
    for keyword in ['closing time', 'close time', 'heure de clôture', 'heure de cloture',
                     'opening time', 'open time', "heure d'entrée", "heure d'entree",
                     'time', 'heure', 'date']:
        for col in df.columns:
            if keyword in col:
                col_date = col
                break
        if col_date:
            break

    df_clean = df.copy()
    df_clean[col_profit] = df_clean[col_profit].apply(extraire_nombre)
    df_clean = df_clean.dropna(subset=[col_profit])
    df_clean, _doublons_exacts = _dedoublonner_trades_exacts(df_clean, col_profit, col_ticket, col_date)

    if len(df_clean) < MIN_TRADES_ABSOLUTE:
        return None, None, f"Pas assez de trades valides trouvés (minimum {MIN_TRADES_ABSOLUTE} requis)."

    trades_filtres = df_clean[col_profit].values

    trades_detail = []
    for _, row in df_clean.iterrows():
        volume = extraire_nombre(row[col_volume]) if col_volume else np.nan
        trades_detail.append({
            "ticket": str(row[col_ticket]) if col_ticket else None,
            "date": str(row[col_date]) if col_date else None,
            "type": str(row[col_type]) if col_type else None,
            "profit": round(float(row[col_profit]), 2),
            "volume": round(float(volume), 4) if not pd.isna(volume) else None,
            "symbole": str(row[col_symbole]).strip() if col_symbole and pd.notna(row[col_symbole]) else None,
        })

        pnl_semantique = _classifier_semantique_pnl(col_profit if 'col_profit' in locals() else 'profit_abs', "ctrader")
    for _d in trades_detail:
        _d["pnl_source"] = pnl_semantique["source"]
        _d["pnl_nature"] = pnl_semantique["nature"]
        _d["pnl_couts_deja_inclus"] = pnl_semantique["couts_deja_inclus"]

    return trades_filtres, trades_detail, None


# ============================================================
# PARSING TRADINGVIEW
#
# Export CSV depuis l'onglet "List of Trades" du Strategy Tester.
# PARTICULARITÉ IMPORTANTE : chaque trade produit DEUX lignes (une
# "Entry ..." et une "Exit ...") ; le Profit n'est renseigné que sur la
# ligne de sortie. Ne pas filtrer dessus ferait DOUBLE COMPTER les
# trades (une fois à 0/vide, une fois avec le vrai profit) et fausserait
# tout le score -> on filtre explicitement sur les lignes "exit".
# ============================================================

def lire_trades_tradingview(contents, filename):
    filename_lower = filename.lower()
    if not filename_lower.endswith('.csv'):
        return None, None, "TradingView exporte uniquement en CSV depuis l'onglet \"List of Trades\"."

    try:
        try:
            df = pd.read_csv(io.BytesIO(contents), sep=',', encoding='utf-8')
            if len(df.columns) <= 1:
                df = pd.read_csv(io.BytesIO(contents), sep=';', encoding='utf-8')
        except Exception:
            df = pd.read_csv(io.BytesIO(contents), sep=';', encoding='latin1')
    except Exception:
        return None, None, "Impossible de lire le fichier."

    df.columns = [str(c).strip().lower() for c in df.columns]

    # Même garde-fou que les autres parsers : ne jamais prendre une
    # métrique de résumé (Profit Factor, Cumulative Profit, etc.) pour
    # le P&L individuel d'un trade.
    col_profit = _choisir_colonne_profit(df.columns, prefer_net=True, allow_gross=True)

    col_type = None
    for col in df.columns:
        if col == 'type':
            col_type = col
            break

    if not col_profit or not col_type:
        return None, None, "Colonnes 'Type' et/ou 'Profit' introuvables. Vérifiez qu'il s'agit bien de l'export \"List of Trades\" du Strategy Tester."

    df[col_type] = df[col_type].astype(str).str.lower()
    df_exits = df[df[col_type].str.contains('exit', na=False)].copy()

    if len(df_exits) == 0:
        return None, None, "Aucune ligne \"Exit\" trouvée — ce fichier ne semble pas être un export \"List of Trades\" standard de TradingView."

    # Les métadonnées d'identification doivent être déterminées AVANT le
    # dédoublonnage : un export TradingView peut contenir deux lignes Exit
    # identiques en apparence, mais appartenant à des trades distincts si
    # aucun identifiant n'est disponible. Dans ce cas, on ne dédoublonne pas.
    col_date = None
    for col in df.columns:
        if 'date' in col or 'time' in col:
            col_date = col
            break
    col_ticket = None
    for col in df.columns:
        if 'trade #' in col or col == 'trade' or '#' in col:
            col_ticket = col
            break

    df_exits[col_profit] = df_exits[col_profit].apply(extraire_nombre)
    df_exits = df_exits.dropna(subset=[col_profit])
    df_exits, _doublons_exacts = _dedoublonner_trades_exacts(df_exits, col_profit, col_ticket, col_date)

    if len(df_exits) < MIN_TRADES_ABSOLUTE:
        return None, None, f"Pas assez de trades valides trouvés (minimum {MIN_TRADES_ABSOLUTE} requis)."
    col_volume = None
    for col in df.columns:
        if any(k in col for k in ['contracts', 'quantity', 'position size', 'shares']):
            col_volume = col
            break

    trades_filtres = df_exits[col_profit].values

    trades_detail = []
    for _, row in df_exits.iterrows():
        volume = extraire_nombre(row[col_volume]) if col_volume else np.nan
        trades_detail.append({
            "ticket": str(row[col_ticket]) if col_ticket else None,
            "date": str(row[col_date]) if col_date else None,
            "type": str(row[col_type]) if col_type else None,
            "profit": round(float(row[col_profit]), 2),
            "volume": round(float(volume), 4) if not pd.isna(volume) else None,
        })

        pnl_semantique = _classifier_semantique_pnl(col_profit if 'col_profit' in locals() else 'profit_abs', "tradingview")
    for _d in trades_detail:
        _d["pnl_source"] = pnl_semantique["source"]
        _d["pnl_nature"] = pnl_semantique["nature"]
        _d["pnl_couts_deja_inclus"] = pnl_semantique["couts_deja_inclus"]

    return trades_filtres, trades_detail, None


# ============================================================
# PARSING NINJATRADER
#
# Export via l'onglet "Trades" du Strategy Analyzer -> clic droit ->
# "Export...". Comme pour cTrader, les colonnes varient selon la
# configuration -> reconnaissance par mots-clés plutôt que noms figés.
# Non vérifié sur un vrai export (contrairement à cTrader) : repose sur
# les noms de colonnes habituels de NinjaTrader, à confirmer si des
# échecs de lecture remontent en usage réel.
# ============================================================

def lire_trades_ninjatrader(contents, filename):
    filename_lower = filename.lower()
    try:
        if filename_lower.endswith('.csv'):
            try:
                df = pd.read_csv(io.BytesIO(contents), sep=',', encoding='utf-8')
                if len(df.columns) <= 1:
                    df = pd.read_csv(io.BytesIO(contents), sep=';', encoding='utf-8')
            except Exception:
                df = pd.read_csv(io.BytesIO(contents), sep=';', encoding='latin1')
        elif filename_lower.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            return None, None, "Format non supporté pour NinjaTrader (utilisez CSV ou Excel exporté depuis l'onglet Trades)."
    except Exception:
        return None, None, "Impossible de lire le fichier."

    df.columns = [str(c).strip().lower() for c in df.columns]

    col_profit = _choisir_colonne_profit(df.columns, prefer_net=True, allow_gross=True)
    if not col_profit:
        return None, None, "Colonne de profit introuvable dans l'export NinjaTrader. Assurez-vous d'inclure une colonne 'Profit' lors de l'export."

    col_type, col_date, col_ticket, col_volume, col_symbole = None, None, None, None, None
    for col in df.columns:
        if ('market pos' in col or 'direction' in col or col == 'type') and col_type is None:
            col_type = col
        if any(k in col for k in ['qty', 'quantity', 'contracts']) and col_volume is None:
            col_volume = col
        if any(k in col for k in ['instrument', 'symbol']) and col_symbole is None:
            col_symbole = col
    for keyword in ['exit time', 'exit date', 'entry time', 'entry date', 'time', 'date']:
        for col in df.columns:
            if keyword in col:
                col_date = col
                break
        if col_date:
            break

    df_clean = df.copy()
    df_clean[col_profit] = df_clean[col_profit].apply(extraire_nombre)
    df_clean = df_clean.dropna(subset=[col_profit])
    df_clean, _doublons_exacts = _dedoublonner_trades_exacts(df_clean, col_profit, col_ticket, col_date)

    if len(df_clean) < MIN_TRADES_ABSOLUTE:
        return None, None, f"Pas assez de trades valides trouvés (minimum {MIN_TRADES_ABSOLUTE} requis)."

    trades_filtres = df_clean[col_profit].values

    trades_detail = []
    for i, (_, row) in enumerate(df_clean.iterrows()):
        volume = extraire_nombre(row[col_volume]) if col_volume else np.nan
        trades_detail.append({
            "ticket": str(i + 1),
            "date": str(row[col_date]) if col_date else None,
            "type": str(row[col_type]) if col_type else None,
            "profit": round(float(row[col_profit]), 2),
            "volume": round(float(volume), 4) if not pd.isna(volume) else None,
            "symbole": str(row[col_symbole]).strip() if col_symbole and pd.notna(row[col_symbole]) else None,
        })

        pnl_semantique = _classifier_semantique_pnl(col_profit if 'col_profit' in locals() else 'profit_abs', "ninjatrader")
    for _d in trades_detail:
        _d["pnl_source"] = pnl_semantique["source"]
        _d["pnl_nature"] = pnl_semantique["nature"]
        _d["pnl_couts_deja_inclus"] = pnl_semantique["couts_deja_inclus"]

    return trades_filtres, trades_detail, None


# ============================================================
# PARSING FREQTRADE
#
# Export natif : freqtrade backtesting --export trades produit un JSON
# (ou un .zip qui le contient) avec une clé "strategy" -> [nom] ->
# "trades", chaque trade ayant pair / close_date / profit_abs. Non
# vérifié sur un vrai export -- repose sur le schéma documenté de
# Freqtrade, à confirmer si des échecs de lecture remontent en usage réel.
# ============================================================

def lire_trades_freqtrade(contents, filename):
    filename_lower = filename.lower()
    donnees = None
    try:
        if filename_lower.endswith('.zip'):
            import zipfile
            with zipfile.ZipFile(io.BytesIO(contents)) as z:
                noms_json = [n for n in z.namelist() if n.endswith('.json') and 'config' not in n.lower()]
                if not noms_json:
                    return None, None, "Aucun fichier JSON de résultats trouvé dans l'archive Freqtrade."
                with z.open(noms_json[0]) as f:
                    donnees = json.load(f)
        elif filename_lower.endswith('.json'):
            donnees = json.loads(contents.decode('utf-8'))
        else:
            return None, None, "Format non supporté pour Freqtrade (utilisez le .json ou .zip généré par --export trades)."
    except Exception:
        return None, None, "Impossible de lire le fichier. Vérifiez qu'il s'agit bien d'un export de résultats Freqtrade."

    liste_trades = None
    if isinstance(donnees, dict) and 'strategy' in donnees:
        for nom_strategie, contenu_strat in donnees['strategy'].items():
            if isinstance(contenu_strat, dict) and 'trades' in contenu_strat:
                liste_trades = contenu_strat['trades']
                break
    elif isinstance(donnees, list):
        liste_trades = donnees
    elif isinstance(donnees, dict) and 'trades' in donnees:
        liste_trades = donnees['trades']

    if not liste_trades:
        return None, None, "Aucun trade trouvé dans ce fichier. Vérifiez qu'il s'agit bien d'un export --export trades (pas juste le résumé)."

    profits, trades_detail = [], []
    for t in liste_trades:
        profit = t.get('profit_abs')
        if profit is None:
            continue
        profits.append(float(profit))
        trades_detail.append({
            "ticket": str(t.get('trade_id', len(trades_detail) + 1)),
            "date": str(t.get('close_date') or t.get('open_date') or ''),
            "type": 'sell' if t.get('is_short') else 'buy',
            "profit": round(float(profit), 2),
            "volume": round(float(t['amount']), 4) if t.get('amount') is not None else None,
            "symbole": str(t.get('pair')) if t.get('pair') else None,
        })

    if len(profits) < MIN_TRADES_ABSOLUTE:
        return None, None, f"Pas assez de trades valides trouvés (minimum {MIN_TRADES_ABSOLUTE} requis)."

    pnl_semantique = _classifier_semantique_pnl("profit_abs", "freqtrade")
    for _d in trades_detail:
        _d["pnl_source"] = pnl_semantique["source"]
        _d["pnl_nature"] = pnl_semantique["nature"]
        _d["pnl_couts_deja_inclus"] = pnl_semantique["couts_deja_inclus"]

    return np.array(profits), trades_detail, None


# ============================================================
# PARSING GÉNÉRIQUE / FORMAT PERSONNALISÉ
#
# Pour toute plateforme sans guide dédié : reconnaissance par mots-clés
# la plus large possible (date, profit, type/direction), sur un CSV ou
# Excel classique une ligne = un trade. C'est délibérément la même
# logique que les parseurs dédiés, juste sans les repères spécifiques
# à une plateforme précise.
# ============================================================

def _filtrer_lignes_probablement_trades(df, col_profit, col_date=None, col_ticket=None, col_type=None, col_symbole=None):
    """Écarte les lignes de résumé manifestes dans un tableau générique.

    Une ligne avec un P&L numérique n'est pas automatiquement un trade : certains
    exports mélangent historique et statistiques de synthèse. On ne filtre que si
    un marqueur structurel de trade est disponible ; sinon on conserve les lignes
    et on laisse l'analyse d'intégrité signaler l'incertitude.
    """
    if not any(c is not None for c in (col_date, col_ticket, col_type, col_symbole)):
        return df, 0
    masks = []
    if col_date is not None:
        dates, _ = _parse_dates_robuste(df[col_date].tolist(), dayfirst_default=None)
        masks.append(dates.notna().to_numpy())
    if col_ticket is not None:
        txt = df[col_ticket].fillna("").astype(str).str.strip()
        masks.append(txt.ne("").to_numpy())
    if col_type is not None:
        txt = df[col_type].fillna("").astype(str).str.strip().str.lower()
        trade_words = r"(buy|sell|long|short|entry|exit|achat|vente|position|trade)"
        masks.append(txt.str.contains(trade_words, regex=True, na=False).to_numpy())
    if col_symbole is not None:
        txt = df[col_symbole].fillna("").astype(str).str.strip()
        masks.append(txt.ne("").to_numpy())
    # Une seule preuve forte suffit ; l'intersection serait trop agressive sur
    # les exports qui n'ont pas tous les champs renseignés.
    mask = np.logical_or.reduce(masks) if masks else np.ones(len(df), dtype=bool)
    removed = int((~mask).sum())
    return df.loc[mask].copy(), removed


def lire_trades_generique(contents, filename):
    filename_lower = filename.lower()
    try:
        if filename_lower.endswith('.csv'):
            try:
                df = pd.read_csv(io.BytesIO(contents), sep=',', encoding='utf-8')
                if len(df.columns) <= 1:
                    df = pd.read_csv(io.BytesIO(contents), sep=';', encoding='utf-8')
            except Exception:
                df = pd.read_csv(io.BytesIO(contents), sep=';', encoding='latin1')
        elif filename_lower.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            return None, None, "Format non supporté (utilisez CSV ou Excel, une ligne par trade)."
    except Exception:
        return None, None, "Impossible de lire le fichier."

    df.columns = [str(c).strip().lower() for c in df.columns]

    col_profit = _choisir_colonne_profit(df.columns, prefer_net=True, allow_gross=False)
    if col_profit is None:
        for col in df.columns:
            if 'gain' in col and not any(x in col for x in ['%', 'percent', 'pourcent']):
                col_profit = col
                break
    if not col_profit:
        return None, None, "Colonne de profit introuvable. Le fichier doit contenir une colonne dont le nom inclut 'profit', 'pnl' ou 'net'."

    col_type, col_date, col_volume, col_symbole = None, None, None, None
    for col in df.columns:
        if any(k in col for k in ['type', 'direction', 'side']) and col_type is None:
            col_type = col
        if any(k in col for k in ['volume', 'quantity', 'lots', 'size']) and col_volume is None:
            col_volume = col
        if any(k in col for k in ['symbol', 'symbole', 'instrument', 'pair', 'paire']) and col_symbole is None:
            col_symbole = col
    for keyword in ['close', 'clôture', 'cloture', 'exit', 'date', 'time', 'heure']:
        for col in df.columns:
            if keyword in col:
                col_date = col
                break
        if col_date:
            break

    df_clean = df.copy()
    df_clean[col_profit] = df_clean[col_profit].apply(extraire_nombre)
    df_clean = df_clean.dropna(subset=[col_profit])
    df_clean, _lignes_resume_exclues = _filtrer_lignes_probablement_trades(
        df_clean, col_profit, col_date, None, col_type, col_symbole
    )

    if len(df_clean) < MIN_TRADES_ABSOLUTE:
        return None, None, f"Pas assez de trades valides trouvés (minimum {MIN_TRADES_ABSOLUTE} requis). Vérifiez que chaque ligne représente bien un trade clôturé."

    trades_filtres = df_clean[col_profit].values

    trades_detail = []
    for i, (_, row) in enumerate(df_clean.iterrows()):
        volume = extraire_nombre(row[col_volume]) if col_volume else np.nan
        trades_detail.append({
            "ticket": str(i + 1),
            "date": str(row[col_date]) if col_date else None,
            "type": str(row[col_type]) if col_type else None,
            "profit": round(float(row[col_profit]), 2),
            "volume": round(float(volume), 4) if not pd.isna(volume) else None,
            "symbole": str(row[col_symbole]).strip() if col_symbole and pd.notna(row[col_symbole]) else None,
        })

        pnl_semantique = _classifier_semantique_pnl(col_profit if 'col_profit' in locals() else 'profit_abs', "autre")
    for _d in trades_detail:
        _d["pnl_source"] = pnl_semantique["source"]
        _d["pnl_nature"] = pnl_semantique["nature"]
        _d["pnl_couts_deja_inclus"] = pnl_semantique["couts_deja_inclus"]

    return trades_filtres, trades_detail, None
PARSERS_PLATEFORME = {
    "mt5": lire_trades_mt5,
    "mt4": lire_trades_mt4,
    "ctrader": lire_trades_ctrader,
    "tradingview": lire_trades_tradingview,
    "ninjatrader": lire_trades_ninjatrader,
    "freqtrade": lire_trades_freqtrade,
    "autre": lire_trades_generique,
}

def obtenir_parser_plateforme(plateforme):
    """Retourne le parser explicite de la plateforme, sans fallback silencieux."""
    cle = str(plateforme or "").strip().lower()
    parser = PARSERS_PLATEFORME.get(cle)
    if parser is None:
        raise HTTPException(status_code=400, detail=f"Plateforme non prise en charge : {plateforme}")
    return parser

@app.get("/protocole-standard-dates")
async def obtenir_dates_protocole_standard():
    """Les dates exactes à backtester pour le protocole Standard,
    recalculées à chaque appel par rapport à aujourd'hui -- jamais
    stockées, jamais figées, pour que l'instruction affichée reste
    toujours valide peu importe quand elle est consultée."""
    aujourdhui = datetime.now(timezone.utc).date()
    date_debut = aujourdhui - timedelta(days=PROTOCOLE_STANDARD_DUREE_JOURS)
    return {
        "date_debut": date_debut.isoformat(),
        "date_fin": aujourdhui.isoformat(),
        "duree_jours": PROTOCOLE_STANDARD_DUREE_JOURS,
    }


# ============================================================
# ENDPOINT PRINCIPAL
# ============================================================

def construire_resultat_analyse(
    contents: bytes,
    filename: str,
    bot_name: str,
    plateforme: str = "mt5",
    capital_initial: float = None,
    protocole: str = "personnalise",
    n_windows: int = N_WINDOWS_DEFAULT,
    is_ratio: float = IS_RATIO_DEFAULT,
    n_trials_testes: int = None,
):
    """Calcule une analyse complète à partir des octets d'un fichier de
    trades, sans aucun effet de bord (ni écriture en base, ni id attribué).

    Fonction pure partagée entre l'endpoint /analyser (usage local, avec
    persistance SQLite) et l'endpoint /demo/analyser (démo web publique,
    sans aucune persistance) -- pour que les deux chemins exécutent
    exactement le même moteur de calcul, jamais deux implémentations qui
    pourraient diverger.
    """
    # Le protocole Standard fixe ses propres réglages, quoi que l'appelant
    # ait pu envoyer par ailleurs -- la comparabilité entre stratégies
    # n'a de sens que si tout le monde est mesuré de la même façon.
    if protocole == "standard":
        n_windows = N_WINDOWS_DEFAULT
        is_ratio = IS_RATIO_DEFAULT

    if not bot_name or not bot_name.strip():
        return {"erreur": "Le nom de la stratégie est requis pour lancer une analyse."}

    # Le capital de départ est obligatoire : sans lui, aucun des
    # pourcentages affichés dans l'interface (l'essentiel de ce qu'un
    # trader regarde en pratique) ne peut être calculé de façon fiable.
    if capital_initial is None or capital_initial <= 0:
        return {"erreur": "Le capital de départ est requis pour lancer une analyse (nécessaire au calcul de tous les pourcentages)."}

    if n_windows < N_WINDOWS_MIN or n_windows > N_WINDOWS_MAX:
        return {"erreur": f"Le nombre de fenêtres walk-forward doit être compris entre {N_WINDOWS_MIN} et {N_WINDOWS_MAX}."}
    if is_ratio < IS_RATIO_MIN or is_ratio > IS_RATIO_MAX:
        return {"erreur": f"Le ratio Référence/Test par fenêtre doit être compris entre {int(IS_RATIO_MIN*100)}% et {int(IS_RATIO_MAX*100)}%."}

    parser = obtenir_parser_plateforme(plateforme)

    trades_array, details_list, err = parser(contents, filename)
    if err:
        return {"erreur": err}

    try:
        # Politique de date connue par plateforme. Les exports francophones
        # MT4/MT5/cTrader utilisent usuellement JJ/MM ; TradingView,
        # NinjaTrader et Freqtrade sont traités sans hypothèse de locale
        # lorsqu'ils fournissent de l'ISO ou des dates non ambiguës. Le
        # format générique reste volontairement conservateur.
        date_policy = {
            "mt4": True, "mt5": True, "ctrader": True,
            "tradingview": None, "ninjatrader": None, "freqtrade": None, "autre": None
        }
        trades_array, details_list, import_warnings = normaliser_integrite_import(
            trades_array, details_list, dayfirst_default=date_policy.get(str(plateforme or "").strip().lower())
        )
    except ValueError as exc:
        return {"erreur": str(exc)}

    # Le seuil minimum par fenêtre s'ajuste au ratio Référence/Test choisi :
    # à ratio élevé (ex: 90% Référence), une fenêtre de 12 trades ne
    # laisserait qu'1 seul trade de Test — la validation passerait, mais le
    # ratio de stabilité calculé sur cette fenêtre serait du bruit
    # statistique pur. On garantit ici un minimum de MIN_TRADES_ABSOLUTE
    # trades de Test par fenêtre en moyenne, quel que soit le ratio choisi.
    taille_min_fenetre = max(MIN_TRADES_PER_WINDOW, math.ceil(MIN_TRADES_ABSOLUTE / (1 - is_ratio)))
    if len(trades_array) < n_windows * taille_min_fenetre:
        return {
            "erreur": f"Historique trop court pour {n_windows} fenêtres walk-forward à un ratio Référence de {int(is_ratio*100)}% "
                      f"({len(trades_array)} trades, {taille_min_fenetre * n_windows} requis au minimum pour garantir assez de trades de Test par fenêtre). "
                      f"Réduisez le nombre de fenêtres, baissez le ratio Référence/Test, ou fournissez un historique plus long."
        }

    rng = get_rng()  # une seule instance de RNG seedée, réutilisée pour toutes les simulations -> résultat 100% reproductible

    # Couverture temporelle de l'historique complet (avant découpage en
    # fenêtres) — signal honnête sur la durée testée, pas une détection
    # de régime de marché (voir la note méthodologique sur la fonction).
    couverture_temporelle = estimer_couverture_temporelle(details_list)

    # Le protocole Standard exige une vraie couverture temporelle -- le
    # nombre de trades, lui, n'est jamais bloquant (une stratégie à
    # faible fréquence n'est pas moins légitime, juste plus lente à
    # prouver statistiquement). Tolérant par rapport aux dates exactes
    # affichées à l'utilisateur (voir /protocole-standard-dates) : un
    # décalage de quelques jours ne doit jamais faire échouer une analyse
    # pour une simple question de timing.
    # La durée/ancienneté du protocole Standard est une RECOMMANDATION de
    # qualité, pas un blocage : un historique court ou ancien reste analysable.
    # Les écarts sont remontés plus bas dans la fiabilité/les alertes afin de
    # ne pas confondre disponibilité des données et robustesse observée.

    # 1. Découpage walk-forward + courbe chronologique continue sur
    # l'historique COMPLET (une seule fois), découpée ensuite exactement
    # comme les trades pour que les segments IS/OOS de chaque fenêtre se
    # raccordent sans discontinuité une fois affichés bout à bout.
    fenetres = decouper_fenetres_walkforward(trades_array, details_list, n_windows, is_ratio)
    courbe_globale = np.cumsum(trades_array)
    for f in fenetres:
        f["is_curve"] = courbe_globale[f["debut"]:f["fin_is"]].tolist()
        f["oos_curve"] = courbe_globale[f["fin_is"]:f["fin"]].tolist()

    # Statistiques globales — le backtest complet tel quel, sans aucune
    # distinction IS/OOS. Sert de repère simple et sans ambiguïté : "voici
    # ce qui s'est passé au total", avant même de rentrer dans le détail
    # walk-forward.
    stats_globales = calculer_statistiques_detaillees(trades_array, capital_initial)
    profit_global = float(trades_array.sum()) if len(trades_array) > 0 else 0.0
    mdd_global = calculer_max_drawdown(trades_array)
    pf_global = calculer_profit_factor(trades_array)

    # Détail propre à CHAQUE fenêtre : contrairement aux stats agrégées
    # (qui regroupent bout à bout des segments non contigus dans le temps),
    # le IS et l'OOS d'une seule fenêtre sont, eux, deux blocs réellement
    # continus et consécutifs — donc sans l'artefact de concaténation.
    capital_courant = capital_initial
    for f in fenetres:
        f_is_profit = float(f["is_trades"].sum()) if len(f["is_trades"]) > 0 else 0.0
        f_oos_capital_ref = capital_courant + f_is_profit
        f["is_stats"] = calculer_statistiques_detaillees(f["is_trades"], capital_courant)
        f["oos_stats"] = calculer_statistiques_detaillees(f["oos_trades"], f_oos_capital_ref)
        f["is_mdd"] = round(calculer_max_drawdown(f["is_trades"]), 2)
        f["oos_mdd"] = round(calculer_max_drawdown(f["oos_trades"]), 2)
        f["is_pf"] = round(calculer_profit_factor(f["is_trades"]), 2)
        f["oos_pf"] = round(calculer_profit_factor(f["oos_trades"]), 2)
        # Périodes calendaires réelles de la fenêtre — ce qui manquait pour
        # comprendre concrètement "à quoi correspond la fenêtre 3 ?".
        f["periode_is"] = extraire_periode(f["is_detail"])
        f["periode_oos"] = extraire_periode(f["oos_detail"])
        f["periode_totale"] = extraire_periode(f["is_detail"] + f["oos_detail"])
        # Le capital repart du solde réel cumulé jusqu'ici, fenêtre après
        # fenêtre, pour une continuité réaliste tout au long de l'historique.
        capital_courant = f_oos_capital_ref + (float(f["oos_trades"].sum()) if len(f["oos_trades"]) > 0 else 0.0)

    is_trades = np.concatenate([f["is_trades"] for f in fenetres]) if fenetres else np.array([])
    oos_trades = np.concatenate([f["oos_trades"] for f in fenetres]) if fenetres else np.array([])
    # Chaque trade garde une trace de sa fenêtre d'origine — utile dans le
    # Journal des trades pour comprendre les sauts de date quand on filtre
    # sur IS ou OOS seul (les segments de fenêtres différentes ne se
    # suivent pas dans le temps une fois regroupés).
    is_detail = [dict(d, fenetre=f["index"]) for f in fenetres for d in f["is_detail"]]
    oos_detail = [dict(d, fenetre=f["index"]) for f in fenetres for d in f["oos_detail"]]
    # Volumes alignés positionnellement avec oos_trades — None partout si
    # la plateforme exportée ne renseignait pas cette colonne (dégradation
    # gracieuse, aucune fonctionnalité existante n'en dépend).
    oos_volumes = [d.get("volume") for d in oos_detail]
    # Calculé sur l'historique COMPLET (pas juste OOS) : c'est une
    # propriété structurelle de la configuration de la stratégie, pas un
    # résultat qui dépend de la période testée.
    ratio_risque_recompense = evaluer_ratio_risque_recompense(details_list)

    # 2. Métriques brutes (agrégées sur l'ensemble des fenêtres)
    is_profit = float(is_trades.sum()) if len(is_trades) > 0 else 0.0
    is_mdd = calculer_max_drawdown(is_trades)
    is_gains = is_trades[is_trades > 0].sum() if len(is_trades) > 0 else 0.0
    is_pertes = abs(is_trades[is_trades < 0].sum()) if len(is_trades) > 0 else 0.0
    is_pf = float(is_gains / is_pertes if is_pertes != 0 else (0.1 if is_gains > 0 else 0))

    oos_profit = float(oos_trades.sum()) if len(oos_trades) > 0 else 0.0
    oos_mdd = calculer_max_drawdown(oos_trades)
    oos_gains = oos_trades[oos_trades > 0].sum() if len(oos_trades) > 0 else 0.0
    oos_pertes = abs(oos_trades[oos_trades < 0].sum()) if len(oos_trades) > 0 else 0.0
    oos_pf = float(oos_gains / oos_pertes if oos_pertes != 0 else (0.1 if oos_gains > 0 else 0))

    # 2bis. Statistiques détaillées façon plateforme de trading, sur les
    # trades IS/OOS agrégés de toutes les fenêtres.
    stats_is = calculer_statistiques_detaillees(is_trades, capital_initial)
    oos_capital_ref = capital_initial + is_profit
    stats_oos = calculer_statistiques_detaillees(oos_trades, oos_capital_ref)

    # 3. Exécution des Piliers & Stress Pro — tous calculés sur l'OOS
    # agrégé (toutes fenêtres concaténées dans l'ordre chronologique), sauf
    # le Pilier 2 qui est spécifiquement multi-fenêtres (voir sa fonction).
    mc_res = executer_monte_carlo(oos_trades, rng, N_SIMULATIONS_MC, segments=[f["oos_trades"] for f in fenetres])
    score_mc = mc_res["score"]
    fragilite_res = stress_test_fragilite(oos_trades, rng, N_SIMULATIONS_STRESS, STRESS_DROP_RATIO)
    costs_res = stress_test_costs(oos_trades, oos_capital_ref, oos_volumes)
    rolling_res = stress_test_rolling(oos_trades, N_ROLLING_WINDOWS, segments=[f["oos_trades"] for f in fenetres])

    def pct(valeur):
        return round(valeur / oos_capital_ref * 100, 2)

    mc_res["pire_mdd_95_pct"] = pct(mc_res["pire_mdd_95"])
    fragilite_res["bootstrap_mediane_profit_pct"] = pct(fragilite_res["bootstrap_mediane_profit"])
    fragilite_res["bootstrap_pire_profit_5pct_pct"] = pct(fragilite_res["bootstrap_pire_profit_5pct"])
    fragilite_res["bootstrap_pire_mdd_pct"] = pct(fragilite_res["bootstrap_pire_mdd"])
    costs_res["stressed_profit_pct"] = pct(costs_res["stressed_profit"])
    costs_res["cost_penalty_total_pct"] = pct(costs_res["cost_penalty_total"])
    rolling_res["rolling_min_window_profit_pct"] = pct(rolling_res["rolling_min_window_profit"])
    rolling_res["rolling_max_window_profit_pct"] = pct(rolling_res["rolling_max_window_profit"])

    pilier2_res = analyser_pilier_is_oos(fenetres, oos_trades)
    pilier3_res = analyser_pilier_risque_drawdown(oos_trades, segments=[f["oos_trades"] for f in fenetres])
    pilier4_res = analyser_pilier_concentration(oos_trades)
    risque_ruine_res = evaluer_risque_de_ruine(oos_trades, oos_volumes)

    # 4. Score Global (Total 100 points)
    score_global_brut = max(0, min(100, pilier2_res["score"] + score_mc + pilier3_res["score"] + pilier4_res["score"]))

    # 4bis. Signatures de gestion du risque : DIAGNOSTIC, pas multiplicateur
    # global. Les signaux de payoff, d'escalade des pertes, de queue extrême
    # et de volume peuvent être utiles, mais ce sont des signatures
    # heuristiques qui se recouvrent partiellement avec le pilier Risque/DD.
    # Les transformer en multiplicateur global arbitraire ferait compter
    # certains risques deux fois et pourrait réduire fortement un score sur
    # la base d'une inférence qui n'est pas une preuve de martingale/ruine.
    # Le score /100 reste donc strictement la somme des 4 piliers définis.
    mult_risque_ruine = 1.0
    score_global = score_global_brut

    # 4ter. Fiabilité de l'évaluation : information séparée du score. Un
    # échantillon limité réduit la portée interprétative du résultat, mais
    # ne modifie pas artificiellement le score de robustesse.
    echantillon_fiable = len(oos_trades) >= MIN_TRADES_RELIABLE
    fiabilite_evaluation = evaluer_fiabilite_evaluation(len(oos_trades), couverture_temporelle)

    # 4quater. Fourchette de variabilité empirique du score par bootstrap par blocs de
    # fenêtres (voir calculer_fourchette_score) — la pénalité de cohérence
    # inter-fenêtres s'applique désormais naturellement à chaque tirage,
    # plus besoin de correctif externe.
    fourchette = calculer_fourchette_score(fenetres, rng, N_CI_REPLICATES, N_CI_PERMS_INNER)
    # La fourchette de variabilité porte sur le même score /100 que le score
    # global ; aucun multiplicateur de diagnostic n'est appliqué.

    # 4quinquies. Seuils de surveillance pour le suivi en réel — même
    # mécanisme de fourchette bootstrap que la comparaison forward test,
    # mais calculé dès l'analyse initiale pour donner à l'utilisateur des
    # repères concrets ("si vous descendez sous X en réel, méfiance") sans
    # attendre d'avoir déjà collecté des données de suivi.
    seuils_surveillance = (
        calculer_fourchette_metriques(oos_trades, oos_capital_ref, rng, N_CI_REPLICATES)
        if len(oos_trades) >= MIN_TRADES_ABSOLUTE else None
    )

    # Rassemblement des alertes de santé
    toutes_les_alertes = (
        [dict(w, pilier="diagnostics") for w in import_warnings]
        + [dict(w, pilier="isoos") for w in pilier2_res["warnings"]]
        + [dict(w, pilier="pilier3") for w in pilier3_res["warnings"]]
        + [dict(w, pilier="pilier4") for w in pilier4_res["warnings"]]
        + [dict(w, pilier="diagnostics") for w in risque_ruine_res["alertes"]]
    )

    # Absence totale ou quasi-totale de pertes : signal direct, vérifié
    # dans les données elles-mêmes (pas une inférence), qui mérite un
    # message honnête plutôt que de laisser deviner la cause via le
    # plafond de qualité indirect (Profit Factor sentinelle) plus haut.
    n_perdants_oos = int((oos_trades < 0).sum())
    if len(oos_trades) >= MIN_TRADES_ABSOLUTE:
        if n_perdants_oos == 0:
            toutes_les_alertes.insert(0, {
                "level": "critical", "pilier": "diagnostics",
                "message": f"Aucune perte enregistrée sur les {len(oos_trades)} trades testés — extrêmement suspect. Vérifiez qu'un stop loss est bien configuré : un backtest sans aucune perte cache souvent un risque non protégé, où une seule perte imprévue pourrait effacer tout le gain accumulé, voire plus."
            })
        elif n_perdants_oos / len(oos_trades) < 0.03:
            toutes_les_alertes.append({
                "level": "warning", "pilier": "diagnostics",
                "message": f"Très peu de pertes enregistrées ({n_perdants_oos} sur {len(oos_trades)} trades testés, {n_perdants_oos / len(oos_trades) * 100:.1f}%) — vérifiez que le stop loss est correctement configuré et testé, ce profil peut cacher un risque mal protégé plutôt qu'une vraie qualité de signal."
            })

    if ratio_risque_recompense is not None:
        niveau = ratio_risque_recompense["niveau"]
        ratio = ratio_risque_recompense["ratio_median"]
        if niveau == "extreme":
            toutes_les_alertes.append({
                "level": "warning", "pilier": "diagnostics",
                "message": f"À titre informatif (n'affecte pas le score) : votre stop loss est configuré en moyenne {ratio:.1f}x plus loin que votre take profit. Ce n'est pas nécessairement un problème si la taille de position est réduite en conséquence — mais si elle reste fixe, une seule perte peut effacer de nombreux gains. Vérifiez votre money management avant de conclure."
            })
        elif niveau == "notable":
            toutes_les_alertes.append({
                "level": "warning", "pilier": "diagnostics",
                "message": f"À titre informatif (n'affecte pas le score) : votre stop loss est configuré en moyenne {ratio:.1f}x plus loin que votre take profit — un profil à surveiller si la taille de position ne s'ajuste pas en conséquence."
            })

    note_biais_selection = message_biais_selection(n_trials_testes)
    if n_trials_testes is None:
        toutes_les_alertes.append({
            "level": "warning", "pilier": "diagnostics",
            "message": "Nombre de variantes testées non renseigné : à titre indicatif seulement (n'affecte pas le score), renseignez-le dans le Diagnostic pour une mise en garde contextualisée sur le biais de sélection multiple."
        })
    elif n_trials_testes >= N_TRIALS_CRITICAL_THRESHOLD:
        toutes_les_alertes.append({
            "level": "warning", "pilier": "diagnostics",
            "message": f"À titre informatif (n'affecte pas le score) : {n_trials_testes} variantes testées avant celle-ci. Même un résultat jamais vu pendant les tests reste biaisé par la sélection de la meilleure variante parmi tant d'autres — voir 'Biais de Sélection' dans le Diagnostic pour comprendre pourquoi."
        })
    elif n_trials_testes >= N_TRIALS_WARNING_THRESHOLD:
        toutes_les_alertes.append({
            "level": "warning", "pilier": "diagnostics",
            "message": f"À titre informatif (n'affecte pas le score) : {n_trials_testes} variantes testées avant celle-ci — voir 'Biais de Sélection' dans le Diagnostic."
        })

    if fiabilite_evaluation["statut"] == "insuffisante":
        toutes_les_alertes.insert(0, {
            "level": "critical",
            "pilier": "diagnostics",
            "message": fiabilite_evaluation["raisons"][0] + " La robustesse calculée reste affichée, mais sa portée interprétative est très limitée."
        })
    elif fiabilite_evaluation["statut"] == "limitée":
        toutes_les_alertes.insert(0, {
            "level": "warning",
            "pilier": "diagnostics",
            "message": fiabilite_evaluation["raisons"][0] + " Interprétez le score avec davantage de prudence."
        })
    if couverture_temporelle is not None:
        span = couverture_temporelle["span_jours"]
        if span < SPAN_JOURS_CRITIQUE:
            toutes_les_alertes.append({
                "level": "critical", "pilier": "diagnostics",
                "message": f"Historique très court : seulement {span} jours au total. Une période aussi courte offre une couverture limitée des conditions de marché observées — la portée interprétative du score est donc réduite."
            })
        elif span < SPAN_JOURS_AVERTISSEMENT:
            toutes_les_alertes.append({
                "level": "warning", "pilier": "diagnostics",
                "message": f"Historique de moins d'un an ({span} jours) : votre stratégie n'a peut-être pas encore été testée sur des conditions de marché très différentes de celles observées jusqu'ici."
            })
    if mc_res["path_sensitivity_ratio"] is not None and mc_res["path_sensitivity_ratio"] >= 2.5:
        toutes_les_alertes.append({
            "level": "warning", "pilier": "montecarlo",
            "message": f"Le drawdown observé dépend sensiblement de l'ordre des trades : parmi les réordonnancements testés, un scénario pessimiste plausible atteint jusqu'à {mc_res['path_sensitivity_ratio']:.1f}x le drawdown observé."
        })
    if mc_res["stress_recovery_factor"] is not None and mc_res["stress_recovery_factor"] < 1.0:
        toutes_les_alertes.append({
            "level": "critical", "pilier": "montecarlo",
            "message": "Dans un scénario de drawdown pessimiste mais plausible, le profit ne suffirait pas à couvrir le risque pris."
        })
    if fragilite_res["bootstrap_prob_positif"] < 60:
        toutes_les_alertes.append({
            "level": "warning", "pilier": "stresspro",
            "message": f"Fragilité : en retirant au hasard {int(STRESS_DROP_RATIO*100)}% des trades (des milliers de fois), seulement {fragilite_res['bootstrap_prob_positif']}% des cas restent rentables — le résultat dépend peut-être de quelques trades précis plutôt que d'un avantage réel et répétable."
        })
    if not costs_res["profitable_under_cost_stress"]:
        toutes_les_alertes.append({
            "level": "critical", "pilier": "stresspro",
            "message": "En simulant des frais de courtage/spread plus élevés, la stratégie devient déficitaire — sa marge est trop fine pour absorber des conditions réelles un peu moins favorables."
        })
    if rolling_res.get("rolling_insuffisant", False):
        toutes_les_alertes.append({
            "level": "info", "pilier": "stresspro",
            "message": "Historique insuffisant pour tester la stabilité temporelle avec les fenêtres rolling configurées — aucune conclusion d'instabilité ne peut être tirée de ce test."
        })
    elif not rolling_res["rolling_stable"]:
        toutes_les_alertes.append({
            "level": "warning", "pilier": "stresspro",
            "message": f"Instabilité dans le temps : seulement {rolling_res['rolling_profitable_pct']}% des périodes de l'historique sont individuellement rentables — la performance globale repose peut-être sur une seule bonne période plutôt que d'être régulière."
        })

    if len(toutes_les_alertes) == 0:
        toutes_les_alertes.append({
            "level": "ok",
            "message": "Aucune alerte détectée. La stratégie présente un profil statistique sain sur l'ensemble des tests effectués."
        })

    resultat = {
        "capital_initial": round(capital_initial, 2),
        "nom_fichier": filename,
        "plateforme": plateforme,
        "protocole": protocole,
        "couverture_temporelle": couverture_temporelle,
        "oos_capital_ref": round(oos_capital_ref, 2),
        "seuils_surveillance": seuils_surveillance,
        "ratio_risque_recompense": ratio_risque_recompense,
        "global": {
            "profit_net": round(profit_global, 2),
            "mdd": round(mdd_global, 2),
            "profit_factor": round(pf_global, 2),
            "n_trades": int(len(trades_array)),
            "stats": stats_globales,
            "curve": [round(v, 2) for v in courbe_globale.tolist()],
            "trades": details_list,
        },
        "walkforward": {
            "n_windows": n_windows,
            "is_ratio": is_ratio,
            "fenetres": [
                {
                    "index": f["index"],
                    "is_n": len(f["is_trades"]),
                    "oos_n": len(f["oos_trades"]),
                    "is_profit": round(float(f["is_trades"].sum()), 2) if len(f["is_trades"]) > 0 else 0.0,
                    "oos_profit": round(float(f["oos_trades"].sum()), 2) if len(f["oos_trades"]) > 0 else 0.0,
                    "is_mdd": f["is_mdd"],
                    "oos_mdd": f["oos_mdd"],
                    "is_pf": f["is_pf"],
                    "oos_pf": f["oos_pf"],
                    "is_stats": f["is_stats"],
                    "oos_stats": f["oos_stats"],
                    "is_curve": [round(v, 2) for v in f["is_curve"]],
                    "oos_curve": [round(v, 2) for v in f["oos_curve"]],
                    "periode_is": f["periode_is"],
                    "periode_oos": f["periode_oos"],
                    "periode_totale": f["periode_totale"],
                    "stability_ratio": (
                        round(calculer_ratio_stabilite(f["is_trades"], f["oos_trades"]), 2)
                        if len(f["is_trades"]) > 0 and len(f["oos_trades"]) > 0 else None
                    ),
                }
                for f in fenetres
            ],
            "stability_ratio_median": pilier2_res["stability_ratio"],
            "fenetres_faibles_pct": pilier2_res["fenetres_faibles_pct"],
        },
        "is": {
            "profit_net": round(is_profit, 2),
            "mdd": round(is_mdd, 2),
            "profit_factor": round(is_pf, 2),
            "curve": np.cumsum(is_trades).tolist() if len(is_trades) > 0 else [],
            "n_trades": int(len(is_trades)),
            "stats": stats_is,
            "trades": is_detail
        },
        "oos": {
            "profit_net": round(oos_profit, 2),
            "mdd": round(oos_mdd, 2),
            "profit_factor": round(oos_pf, 2),
            "curve": np.cumsum(oos_trades).tolist() if len(oos_trades) > 0 else [],
            "n_trades": int(len(oos_trades)),
            "stats": stats_oos,
            "trades": oos_detail
        },
        "monte_carlo": {
            "pire_mdd_95": mc_res["pire_mdd_95"],
            "pire_mdd_95_pct": mc_res["pire_mdd_95_pct"],
            "path_sensitivity_ratio": mc_res["path_sensitivity_ratio"],
            "stress_recovery_factor": mc_res["stress_recovery_factor"],
            "score_path": mc_res["score_path"],
            "score_recovery": mc_res["score_recovery"]
        },
        "stress_pro_bootstrap": fragilite_res,
        "stress_pro_costs": costs_res,
        "stress_pro_rolling": rolling_res,
        "robustesse": {
            "score_global": score_global,
            "score_global_brut": score_global_brut,
            "n_trials_testes": n_trials_testes,
            "biais_selection_note": note_biais_selection,
            "mult_risque_ruine": mult_risque_ruine,
            "ruine_winrate": risque_ruine_res["winrate"],
            "ruine_payoff_ratio": risque_ruine_res["payoff_ratio"],
            "ruine_pct_escalade": risque_ruine_res["pct_sequences_escalade"],
            "ruine_n_sequences": risque_ruine_res["n_sequences_pertes"],
            "ruine_ratio_queue": risque_ruine_res["ratio_queue"],
            "ruine_signal_volume": risque_ruine_res["signal_volume"],
            "ruine_pct_volume_escalade": risque_ruine_res["pct_volume_escalade"],
            "ruine_cohesion_volume": risque_ruine_res["cohesion_volume"],
            "score_fourchette": fourchette,
            "score_monte_carlo": score_mc,
            "score_stabilite_is_oos": pilier2_res["score"],
            "score_risque_drawdown": pilier3_res["score"],
            "score_concentration": pilier4_res["score"],
            "max_losing_streak": pilier3_res["max_losing_streak"],
            "profit_to_dd_ratio": pilier3_res["profit_to_dd_ratio"],
            "profit_to_dd_ratio_pire_fenetre": pilier3_res.get("profit_to_dd_ratio_pire_fenetre"),
            "concentration_pct": pilier4_res["concentration_pct"],
            "top_n_trades": pilier4_res["top_n_trades"],
            "top_gains_sum": pilier4_res["top_gains_sum"],
            "stability_ratio": pilier2_res["stability_ratio"],
            "ratios_par_fenetre": pilier2_res["ratios_par_fenetre"],
            "fenetres_faibles_pct": pilier2_res["fenetres_faibles_pct"],
            "oos_profit_factor": pilier2_res["oos_profit_factor"],
            "echantillon_fiable": echantillon_fiable,
            "n_trades_oos": int(len(oos_trades)),
            "warnings": toutes_les_alertes
        }
    }

    return resultat


@app.post("/analyser")
async def analyser(
    trades_file: UploadFile = File(...),
    bot_name: str = Form(...),
    plateforme: str = Form("mt5"),
    capital_initial: float = Form(None),
    protocole: str = Form("personnalise"),  # "standard" ou "personnalise"
    n_windows: int = Form(N_WINDOWS_DEFAULT),
    is_ratio: float = Form(IS_RATIO_DEFAULT),
    n_trials_testes: int = Form(None),
    strategie_id: int = Form(None),
    code_source: str = Form(None),
):
    contents = await trades_file.read()
    resultat = construire_resultat_analyse(
        contents, trades_file.filename, bot_name, plateforme, capital_initial,
        protocole, n_windows, is_ratio, n_trials_testes,
    )
    if "erreur" in resultat:
        return resultat

    # Auto-sauvegarde : chaque analyse lancée devient directement une
    # version de la stratégie nommée, sans étape "Enregistrer" séparée —
    # revenir sur une stratégie après avoir quitté XTRUNN, ou relancer une
    # analyse après avoir modifié son algo, ne demande plus jamais de
    # ressaisir quoi que ce soit. Une version individuelle reste
    # supprimable a posteriori depuis la liste des versions d'une
    # stratégie, si un essai s'avère sans intérêt à garder.
    analyse_id, strategie_id_resultante = _enregistrer_analyse_en_base(bot_name, resultat, strategie_id, code_source, contents)
    resultat["id"] = analyse_id
    resultat["strategie_id"] = strategie_id_resultante
    return resultat


# ============================================================
# PONT VERS LE FORWARD TESTING
#
# Un backtest, même walk-forward et bien stress-testé, reste une
# prédiction — pas une garantie. La seule vraie validation, c'est de
# comparer ce qui était annoncé à ce qui s'est réellement passé une fois
# la stratégie déployée en démo (forward test). Cette fonction compare
# les statistiques d'un nouveau journal de trades (le forward test) aux
# statistiques OOS de l'analyse backtest d'origine, sur 3 axes : Profit
# Factor, Winrate, Max Drawdown %. Un désaccord ne prouve pas que le
# backtest était faux — les conditions de marché changent — mais un
# forward test qui diverge nettement de ce qui était annoncé mérite
# d'être pris au sérieux avant d'aller plus loin.
# ============================================================

# ============================================================
# MOTEUR DE RÈGLES DU MODE EPREUVE
#
# Simule les règles typiques d'une épreuve de prop firm : objectif de
# profit, drawdown maximal, perte quotidienne maximale, règle de
# régularité (aucune journée ne doit représenter une part disproportionnée
# du profit total), nombre minimum de jours de trading. Chaque règle est
# optionnelle (None = non appliquée), pour permettre des épreuves aussi
# simples ou complets que voulu — y compris un "Epreuve XTRUNN" officiel
# avec un jeu de règles standardisé.
# ============================================================

def calculer_score_epreuve(epreuve, violations, profit_pct, drawdown_pct, consistency_pct):
    """
    Score 0-100, distinct du verdict réussi/échoué : mesure la qualité de
    l'exécution, pas seulement si l'objectif a été coché. Seule une
    VRAIE VIOLATION DE RÈGLE (drawdown, perte quotidienne, régularité)
    ramène le score à 0 — pas de demi-mesure sur un risque dépassé,
    cohérent avec le fonctionnement d'une épreuve à règles de risque strictes.

    Ne pas atteindre l'objectif de profit dans les temps, SANS avoir
    enfreint la moindre règle, reste un échec du VERDICT (l'épreuve
    n'est pas validée), mais pas nécessairement un score de 0 — quelqu'un
    qui finit à 90% de l'objectif avec un excellent contrôle du risque
    n'a pas la même exécution que quelqu'un qui a explosé son drawdown.

    Retourne le détail par composante (pas juste le total), pour pouvoir
    l'expliquer clairement dans l'onglet Score plutôt que d'afficher un
    chiffre opaque.
    """
    if violations:
        return {"total": 0, "profit": 0.0, "risque": 0.0, "regularite": 0.0}

    # Objectif de profit (jusqu'à 50 pts) : 40 pts pour l'atteindre pile,
    # jusqu'à 50 si dépassé largement (rendements décroissants au-delà).
    score_profit = 25.0  # base neutre si aucun objectif de profit défini
    if epreuve.get("objectif_profit_pct") and epreuve["objectif_profit_pct"] > 0:
        ratio = profit_pct / epreuve["objectif_profit_pct"]
        score_profit = min(50.0, max(0.0, ratio * 40.0))
    elif profit_pct > 0:
        score_profit = min(50.0, 25.0 + profit_pct)

    # Marge de drawdown non utilisée (jusqu'à 30 pts) : moins on a entamé
    # le budget de risque autorisé, mieux c'est.
    score_risque = 30.0
    if epreuve.get("drawdown_max_pct") and epreuve["drawdown_max_pct"] > 0:
        ratio_dd = drawdown_pct / epreuve["drawdown_max_pct"]
        score_risque = max(0.0, 30.0 * (1 - ratio_dd))

    # Régularité (jusqu'à 20 pts) : moins le profit dépend d'une seule
    # journée exceptionnelle, mieux c'est (même logique que le Pilier
    # Concentration du backtest, appliquée ici jour par jour).
    score_regularite = max(0.0, 20.0 * (1 - min(1.0, consistency_pct / 100)))

    total = round(max(0, min(100, score_profit + score_risque + score_regularite)))
    return {
        "total": total,
        "profit": round(score_profit, 1),
        "risque": round(score_risque, 1),
        "regularite": round(score_regularite, 1),
    }


def evaluer_epreuve(epreuve, trades, periode_ecoulee=False):
    """
    epreuve : dict avec capital_initial et les règles optionnelles
                (objectif_profit_pct, drawdown_max_pct,
                perte_quotidienne_max_pct, consistency_max_pct,
                jours_min_trading).
    trades : liste de dicts {date: "YYYY-MM-DD", profit: float},
             pas nécessairement triée.
    periode_ecoulee : la durée totale de l'épreuve est-elle passée ? Une
                       réussite ne se fige QUE si oui -- atteindre
                       l'objectif tôt ne termine plus l'épreuve par
                       anticipation (comme un vrai challenge de prop
                       firm, il faut tenir toute la période), et surtout
                       ça permet de ne jamais révéler le score avant la
                       fin réelle (voir plus bas).
    """
    capital = epreuve["capital_initial"]
    trades_tries = sorted(trades, key=lambda t: t["date"])

    if not trades_tries:
        return {
            "statut": "en_cours", "profit_pct": 0.0, "profit_net": 0.0,
            "drawdown_pct": 0.0, "drawdown_dollars": 0.0, "pire_jour_pct": 0.0,
            "consistency_pct": 0.0, "jours_trading": 0, "n_trades": 0,
            "violations": [], "objectif_atteint": False, "jours_suffisants": False,
            "courbe": [capital], "score": None, "score_detail": None,
        }

    courbe = [capital]
    cumul = capital
    for t in trades_tries:
        cumul += t["profit"]
        courbe.append(cumul)

    pic = capital
    max_dd_pct = 0.0
    max_dd_dollars = 0.0
    for v in courbe:
        pic = max(pic, v)
        dd_dollars = pic - v
        dd_pct = (dd_dollars / pic * 100) if pic > 0 else 0.0
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct
            max_dd_dollars = dd_dollars

    profit_net = courbe[-1] - capital
    profit_pct = (profit_net / capital * 100) if capital > 0 else 0.0

    par_jour = {}
    for t in trades_tries:
        par_jour.setdefault(t["date"], 0.0)
        par_jour[t["date"]] += t["profit"]

    pire_jour_dollars = min(par_jour.values()) if par_jour else 0.0
    pire_jour_pct = (pire_jour_dollars / capital * 100) if capital > 0 else 0.0
    meilleur_jour_dollars = max(par_jour.values()) if par_jour else 0.0
    consistency_pct = (meilleur_jour_dollars / profit_net * 100) if profit_net > 0 else 0.0
    jours_trading = len(par_jour)

    violations = []
    if epreuve.get("drawdown_max_pct") and max_dd_pct > epreuve["drawdown_max_pct"]:
        violations.append(f"Drawdown maximal dépassé : {max_dd_pct:.1f}% (limite : {epreuve['drawdown_max_pct']:.0f}%)")
    if (epreuve.get("perte_quotidienne_max_pct") and pire_jour_pct < 0
            and abs(pire_jour_pct) > epreuve["perte_quotidienne_max_pct"]):
        violations.append(f"Perte quotidienne maximale dépassée : {abs(pire_jour_pct):.1f}% en une seule journée (limite : {epreuve['perte_quotidienne_max_pct']:.0f}%)")
    if (epreuve.get("consistency_max_pct") and profit_net > 0 and jours_trading >= EPREUVE_JOURS_MIN_REGULARITE
            and consistency_pct > epreuve["consistency_max_pct"]):
        violations.append(f"Règle de régularité non respectée : {consistency_pct:.0f}% du profit vient d'une seule journée (limite : {epreuve['consistency_max_pct']:.0f}%)")

    objectif_atteint = not epreuve.get("objectif_profit_pct") or profit_pct >= epreuve["objectif_profit_pct"]
    jours_suffisants = not epreuve.get("jours_min_trading") or jours_trading >= epreuve["jours_min_trading"]

    # raison_echec distingue deux situations que le statut "echoue" seul
    # confondait : une vraie violation de règle de risque (grave, score à
    # 0) contre un simple objectif non atteint dans les temps, sans avoir
    # enfreint la moindre règle (le score, lui, reste mérité).
    raison_echec = None
    if violations:
        statut = "echoue"
        raison_echec = "violation"
    elif periode_ecoulee:
        if objectif_atteint and jours_suffisants:
            statut = "reussi"
        else:
            statut = "echoue"
            raison_echec = "objectif_non_atteint"
    else:
        statut = "en_cours"

    score = calculer_score_epreuve(epreuve, violations, profit_pct, max_dd_pct, consistency_pct)
    # Le score (et son détail) ne sont jamais transmis tant que l'épreuve
    # est en cours -- une vraie certification "à l'aveugle" jusqu'au
    # bout, pas seulement une interface qui s'abstient de l'afficher :
    # même en inspectant la réponse brute, rien à voir avant la fin.
    if statut == "en_cours":
        score = None

    return {
        "statut": statut,
        "raison_echec": raison_echec,
        "profit_pct": round(profit_pct, 2),
        "profit_net": round(profit_net, 2),
        "drawdown_pct": round(max_dd_pct, 2),
        "drawdown_dollars": round(max_dd_dollars, 2),
        "pire_jour_pct": round(pire_jour_pct, 2),
        "consistency_pct": round(consistency_pct, 1),
        "jours_trading": jours_trading,
        "n_trades": len(trades_tries),
        "violations": violations,
        "objectif_atteint": objectif_atteint,
        "jours_suffisants": jours_suffisants,
        "courbe": [round(v, 2) for v in courbe],
        "score": score["total"] if score else None,
        "score_detail": score,
    }


# ============================================================
# ENDPOINTS HISTORIQUE
# ============================================================

class SauvegardeRequest(BaseModel):
    bot_name: str
    resultat: dict
    strategie_id: int = None  # si fourni, rattache à cette stratégie existante plutôt que d'en créer une nouvelle


def _enregistrer_analyse_en_base(bot_name, resultat, strategie_id_force=None, code_source=None, fichier_original=None, periode_deja_reglee=None):
    """Logique partagée entre l'auto-sauvegarde à l'analyse et l'ancien
    endpoint de sauvegarde explicite : crée la stratégie si elle n'existe
    pas encore (ou la réutilise si le même nom existe déjà), puis
    enregistre cette version de l'analyse. Retourne (id_analyse, id_strategie).

    code_source et fichier_original ne viennent que de l'endpoint
    /analyser (une vraie nouvelle analyse, avec le fichier encore en
    mémoire) -- jamais du ré-enregistrement via /historique, qui ne
    dispose plus du fichier d'origine ni d'un nouveau code à coller."""
    bot_name = bot_name.strip()
    if not bot_name:
        raise HTTPException(status_code=400, detail="Le nom de la stratégie est requis pour l'enregistrement.")

    conn = get_db()
    if strategie_id_force:
        strat = conn.execute("SELECT * FROM strategies WHERE id = ?", (strategie_id_force,)).fetchone()
        if strat is None:
            conn.close()
            raise HTTPException(status_code=404, detail="Stratégie introuvable.")
        strategie_id = strategie_id_force
    else:
        # Retaper le même nom qu'une stratégie déjà existante = continuer
        # cette même stratégie (nouvelle version), pas en créer un doublon.
        existante = conn.execute("SELECT id FROM strategies WHERE nom = ?", (bot_name,)).fetchone()
        if existante:
            strategie_id = existante["id"]
        else:
            cur = conn.execute(
                "INSERT INTO strategies (nom, created_at) VALUES (?, ?)",
                (bot_name, datetime.now(timezone.utc).isoformat())
            )
            strategie_id = cur.lastrowid

    rob = resultat.get("robustesse", {})
    oos = resultat.get("oos", {})
    fourchette = rob.get("score_fourchette", {})

    cur = conn.execute(
        """
        INSERT INTO analyses
            (bot_name, strategie_id, created_at, score_global, score_median, score_p10, score_p90,
             oos_profit_net, oos_profit_factor, n_trades_oos, echantillon_fiable, fiabilite_evaluation, raisons_fiabilite, resultat_json, nom_fichier,
             capital_initial, plateforme, code_source, fichier_original, periode_deja_reglee)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            bot_name,
            strategie_id,
            datetime.now(timezone.utc).isoformat(),
            rob.get("score_global"),
            fourchette.get("median"),
            fourchette.get("p10"),
            fourchette.get("p90"),
            oos.get("profit_net"),
            oos.get("profit_factor"),
            oos.get("n_trades"),
            1 if rob.get("echantillon_fiable") else 0,
            rob.get("fiabilite_evaluation"),
            json.dumps(rob.get("raisons_fiabilite", []), ensure_ascii=False),
            json.dumps(resultat),
            resultat.get("nom_fichier"),
            resultat.get("capital_initial"),
            resultat.get("plateforme"),
            code_source.strip() if code_source else None,
            fichier_original,
            (1 if periode_deja_reglee else 0) if periode_deja_reglee is not None else None,
        )
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id, strategie_id


@app.post("/historique")
async def sauvegarder_analyse(req: SauvegardeRequest):
    new_id, strategie_id = _enregistrer_analyse_en_base(req.bot_name, req.resultat, req.strategie_id)
    return {"id": new_id, "strategie_id": strategie_id, "message": "Analyse enregistrée."}


@app.get("/historique/noms")
async def lister_noms_connus():
    conn = get_db()
    rows = conn.execute("SELECT DISTINCT nom FROM strategies ORDER BY nom ASC").fetchall()
    conn.close()
    return {"noms": [r["nom"] for r in rows]}


@app.get("/historique/{analyse_id}")
async def charger_analyse(analyse_id: int):
    conn = get_db()
    row = conn.execute("SELECT * FROM analyses WHERE id = ?", (analyse_id,)).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail="Analyse introuvable.")
    result = dict(row)
    result["resultat"] = json.loads(result.pop("resultat_json"))
    # Le fichier brut est un BLOB binaire, jamais sérialisable en JSON --
    # on ne renvoie qu'un indicateur de présence, le vrai contenu se
    # récupère via /historique/{id}/fichier-original en téléchargement direct.
    result["a_fichier_original"] = result.pop("fichier_original") is not None
    return result


@app.get("/historique/{analyse_id}/fichier-original")
async def telecharger_fichier_original(analyse_id: int):
    """Retourne le fichier brut tel qu'importé à l'époque, même si
    l'utilisateur ne l'a plus sur son propre ordinateur depuis longtemps."""
    conn = get_db()
    row = conn.execute("SELECT fichier_original, nom_fichier FROM analyses WHERE id = ?", (analyse_id,)).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail="Analyse introuvable.")
    if row["fichier_original"] is None:
        raise HTTPException(status_code=404, detail="Aucun fichier conservé pour cette analyse.")
    nom = row["nom_fichier"] or f"trades_{analyse_id}.csv"
    return Response(
        content=row["fichier_original"],
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{nom}"'}
    )


class CodeSourcePayload(BaseModel):
    code_source: str = None


@app.put("/historique/{analyse_id}/code-source")
async def mettre_a_jour_code_source(analyse_id: int, payload: CodeSourcePayload):
    """Colle ou corrige le code source après coup -- utile pour ceux qui
    l'ont oublié à la création, ou qui veulent le rectifier."""
    conn = get_db()
    cur = conn.execute(
        "UPDATE analyses SET code_source = ? WHERE id = ?",
        (payload.code_source.strip() if payload.code_source else None, analyse_id)
    )
    conn.commit()
    updated = cur.rowcount
    conn.close()
    if updated == 0:
        raise HTTPException(status_code=404, detail="Analyse introuvable.")
    return {"message": "Code source mis à jour."}


@app.delete("/historique/{analyse_id}")
async def supprimer_analyse(analyse_id: int):
    conn = get_db()
    # Une épreuve liée à cette version précise n'a plus de sens une fois
    # l'analyse supprimée -- sans ce nettoyage, il devient orphelin,
    # invisible et indestructible par les moyens normaux (vécu concrètement :
    # affiché comme "Stratégie" générique à l'accueil, impossible à ouvrir).
    ids_epreuves = [r["id"] for r in conn.execute("SELECT id FROM epreuves WHERE analyse_id = ?", (analyse_id,)).fetchall()]
    for cid in ids_epreuves:
        conn.execute("DELETE FROM epreuve_trades WHERE epreuve_id = ?", (cid,))
    conn.execute("DELETE FROM epreuves WHERE analyse_id = ?", (analyse_id,))
    cur = conn.execute("DELETE FROM analyses WHERE id = ?", (analyse_id,))
    conn.commit()
    deleted = cur.rowcount
    conn.close()
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Analyse introuvable.")
    return {"message": "Analyse supprimée."}


# ============================================================
# ENDPOINTS STRATÉGIES
#
# Une stratégie regroupe toutes ses analyses (versions successives d'un
# même algo, au fil des réglages) et toutes ses épreuves — le vrai point
# d'entrée à l'accueil, plutôt que de naviguer analyse par analyse.
# ============================================================

@app.get("/export-donnees")
async def exporter_donnees():
    """Renvoie le fichier de base de données tel quel -- une sauvegarde
    complète et fidèle (stratégies, analyses, épreuves, fichiers
    originaux inclus), sans reconstruire un format intermédiaire qui
    risquerait d'oublier quelque chose."""
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=404, detail="Aucune donnée à exporter pour le moment.")
    nom_fichier = f"xtrunn_sauvegarde_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.db"
    return FileResponse(DB_PATH, media_type="application/octet-stream", filename=nom_fichier)


@app.post("/importer-donnees")
async def importer_donnees(fichier: UploadFile = File(...)):
    """Remplace la base de données actuelle par le fichier importé --
    après une vérification minimale que c'est bien une sauvegarde
    XTRUNN, pas n'importe quel fichier .db au hasard."""
    contenu = await fichier.read()
    chemin_temp = DB_PATH + ".import_temp"
    with open(chemin_temp, "wb") as f:
        f.write(contenu)

    try:
        conn_test = sqlite3.connect(chemin_temp)
        tables = {r[0] for r in conn_test.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        conn_test.close()
        if not {"strategies", "analyses"}.issubset(tables):
            os.remove(chemin_temp)
            return {"erreur": "Ce fichier ne ressemble pas à une sauvegarde XTRUNN valide."}
    except sqlite3.DatabaseError:
        if os.path.exists(chemin_temp):
            os.remove(chemin_temp)
        return {"erreur": "Fichier illisible -- ce n'est pas une base de données SQLite valide."}

    os.replace(chemin_temp, DB_PATH)
    init_db()  # applique les migrations éventuelles à la sauvegarde importée, si elle vient d'une version plus ancienne
    return {"ok": True}


@app.post("/reinitialiser-donnees")
async def reinitialiser_donnees():
    """Efface toutes les stratégies, analyses et épreuves -- geste
    irréversible, la confirmation est de la responsabilité de
    l'interface avant d'appeler cet endpoint."""
    conn = get_db()
    for table in ["epreuve_trades", "epreuves", "analyses", "strategies"]:
        conn.execute(f"DELETE FROM {table}")
    conn.commit()
    conn.close()
    return {"ok": True}


@app.get("/strategies")
async def lister_strategies():
    conn = get_db()
    strategies = conn.execute("SELECT * FROM strategies ORDER BY nom ASC").fetchall()
    resultats = []
    for s in strategies:
        s = dict(s)
        derniere = conn.execute(
            "SELECT id, created_at, score_global, score_p10, nom_fichier, capital_initial, plateforme, (code_source IS NOT NULL) AS a_code_source, (fichier_original IS NOT NULL) AS a_fichier_original FROM analyses WHERE strategie_id = ? ORDER BY created_at DESC LIMIT 1",
            (s["id"],)
        ).fetchone()
        n_analyses = conn.execute(
            "SELECT COUNT(*) as n FROM analyses WHERE strategie_id = ?", (s["id"],)
        ).fetchone()["n"]
        # Jointure directe sur epreuves.strategie_id -- l'ancienne version
        # passait par analyses.strategie_id, ratant toute épreuve liée
        # sans version d'analyse précise (ex: Épreuve Officielle sans
        # lien à une analyse particulière).
        n_epreuves_actives = conn.execute(
            "SELECT COUNT(*) as n FROM epreuves WHERE strategie_id = ? AND statut = 'en_cours'", (s["id"],)
        ).fetchone()["n"]
        # Certification : au moins une Épreuve XTRUNN Officielle réussie
        # pour cette stratégie -- le seul type d'épreuve dont le
        # règlement est standardisé, donc le seul qui a du sens comme
        # certification comparable d'une stratégie à l'autre.
        certification = conn.execute(
            """SELECT id, created_at FROM epreuves
               WHERE strategie_id = ? AND type_epreuve = 'xtrunn_officiel' AND statut = 'reussi'
               ORDER BY created_at DESC LIMIT 1""",
            (s["id"],)
        ).fetchone()
        # Validation : le vrai accomplissement -- analyse robuste ET
        # exécution réelle restée cohérente avec elle. Distinct de la
        # certification (entraînement à règles fixes, non liée au backtest).
        validation = conn.execute(
            """SELECT id, created_at FROM epreuves
               WHERE strategie_id = ? AND type_epreuve = 'validation' AND statut = 'reussi'
               ORDER BY created_at DESC LIMIT 1""",
            (s["id"],)
        ).fetchone()
        s["derniere_analyse"] = dict(derniere) if derniere else None
        s["n_analyses"] = n_analyses
        s["n_epreuves_actives"] = n_epreuves_actives
        s["validation_reussie"] = dict(validation) if validation else None
        s["certification_officielle"] = dict(certification) if certification else None
        resultats.append(s)
    conn.close()
    # Stratégies les plus récemment actives en premier (par date de leur
    # dernière analyse, à défaut par date de création).
    resultats.sort(key=lambda s: s["derniere_analyse"]["created_at"] if s["derniere_analyse"] else s["created_at"], reverse=True)
    return {"strategies": resultats}


@app.get("/strategies/{strategie_id}")
async def obtenir_strategie(strategie_id: int):
    conn = get_db()
    strat = conn.execute("SELECT * FROM strategies WHERE id = ?", (strategie_id,)).fetchone()
    if strat is None:
        conn.close()
        raise HTTPException(status_code=404, detail="Stratégie introuvable.")
    analyses = conn.execute(
        """SELECT id, created_at, score_global, score_median, score_p10, score_p90,
                  oos_profit_net, oos_profit_factor, n_trades_oos, echantillon_fiable, nom_fichier,
                  capital_initial, plateforme,
                  (code_source IS NOT NULL) AS a_code_source,
                  (fichier_original IS NOT NULL) AS a_fichier_original
           FROM analyses WHERE strategie_id = ? ORDER BY created_at DESC""",
        (strategie_id,)
    ).fetchall()
    conn.close()
    analyses_dict = [dict(a) for a in analyses]
    for a in analyses_dict:
        a["eligible_validation"] = a["score_p10"] is not None and a["score_p10"] >= SEUIL_SCORE_VALIDATION
    return {"strategie": dict(strat), "analyses": analyses_dict}


@app.delete("/strategies/{strategie_id}")
async def supprimer_strategie(strategie_id: int):
    conn = get_db()
    analyse_ids = [r["id"] for r in conn.execute("SELECT id FROM analyses WHERE strategie_id = ?", (strategie_id,)).fetchall()]
    ids_epreuves_a_purger = [
        r["id"] for r in conn.execute(
            "SELECT id FROM epreuves WHERE strategie_id = ? OR analyse_id IN ({})".format(
                ",".join("?" * len(analyse_ids)) if analyse_ids else "NULL"
            ),
            [strategie_id] + analyse_ids
        ).fetchall()
    ]
    for cid in ids_epreuves_a_purger:
        conn.execute("DELETE FROM epreuve_trades WHERE epreuve_id = ?", (cid,))
    conn.execute(
        "DELETE FROM epreuves WHERE id IN ({})".format(",".join("?" * len(ids_epreuves_a_purger)) if ids_epreuves_a_purger else "NULL"),
        ids_epreuves_a_purger
    )
    conn.execute("DELETE FROM analyses WHERE strategie_id = ?", (strategie_id,))
    cur = conn.execute("DELETE FROM strategies WHERE id = ?", (strategie_id,))
    conn.commit()
    deleted = cur.rowcount
    conn.close()
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Stratégie introuvable.")
    return {"message": "Stratégie et toutes ses analyses/épreuves supprimées."}


# ============================================================
# DIAGNOSTIC ÉPREUVE — calculé uniquement une fois l'épreuve terminée
# (jamais pendant qu'elle est en cours, cohérent avec le score caché
# jusqu'à la fin). Trois volets distincts, qui portent tous les trois
# sur l'EXÉCUTION RÉELLE plutôt que sur un backtest simulé -- ce qui
# rend ce diagnostic complémentaire à celui de l'Analyse, pas redondant
# avec lui :
#   1. Cohérence avec le backtest d'origine (si l'épreuve est liée à
#      une stratégie/analyse) — les résultats réels sont-ils restés
#      dans la fourchette statistiquement attendue par ce backtest ?
#   2. Robustesse à l'ordre réel des trades (Monte Carlo) — le verdict
#      aurait-il tenu avec une séquence différente des mêmes trades ?
#   3. Récapitulatif factuel de l'exécution — la trace honnête de ce
#      qui s'est passé, non noté.
# ============================================================

def evaluer_axe_seuil(valeur_reelle, p10, p90, superieur_est_pire=False):
    """
    Compare une métrique réelle de l'épreuve à la fourchette bootstrap
    [P10, P90] du backtest d'origine -- la variabilité naturelle que
    cette métrique aurait pu prendre avec un tirage légèrement différent
    des mêmes trades OOS. Marge de tolérance de 50% de la largeur de la
    fourchette avant de basculer en alerte, pour ne pas réagir au
    moindre dépassement d'un seuil qui reste une estimation statistique.
    """
    if valeur_reelle is None or p10 is None or p90 is None:
        return "indetermine"
    largeur = p90 - p10
    if largeur <= 0:
        largeur = abs(p90) * 0.1 + 0.01
    marge = largeur * 0.5
    if not superieur_est_pire:
        if valeur_reelle >= p10:
            return "coherent"
        elif valeur_reelle >= p10 - marge:
            return "a_surveiller"
        else:
            return "alerte"
    else:
        if valeur_reelle <= p90:
            return "coherent"
        elif valeur_reelle <= p90 + marge:
            return "a_surveiller"
        else:
            return "alerte"


def calculer_coherence_backtest(epreuve_row, trades_array, stats):
    """Compare les métriques réelles de l'épreuve aux "Seuils de
    Surveillance" déjà calculés et stockés dans l'analyse liée -- pas
    besoin de refaire le bootstrap, il existe déjà."""
    if not epreuve_row.get("analyse_id"):
        return None
    conn = get_db()
    analyse = conn.execute("SELECT resultat_json FROM analyses WHERE id = ?", (epreuve_row["analyse_id"],)).fetchone()
    conn.close()
    if analyse is None:
        return None
    resultat_backtest = json.loads(analyse["resultat_json"])
    seuils = resultat_backtest.get("seuils_surveillance")
    if not seuils:
        return None

    pf_reel = calculer_profit_factor(trades_array)
    winrate_reel = stats["winrate_pct"]
    mdd_reel = stats["mdd_pct"]

    axe_pf = evaluer_axe_seuil(pf_reel, seuils["profit_factor"]["p10"], seuils["profit_factor"]["p90"], superieur_est_pire=False)
    axe_winrate = evaluer_axe_seuil(winrate_reel, seuils["winrate_pct"]["p10"], seuils["winrate_pct"]["p90"], superieur_est_pire=False)
    axe_mdd = evaluer_axe_seuil(mdd_reel, seuils["mdd_pct"]["p10"], seuils["mdd_pct"]["p90"], superieur_est_pire=True)

    statuts = [axe_pf, axe_winrate, axe_mdd]
    if "alerte" in statuts:
        verdict = "alerte"
    elif "a_surveiller" in statuts:
        verdict = "a_surveiller"
    elif all(s == "coherent" for s in statuts):
        verdict = "coherent"
    else:
        verdict = "indetermine"

    return {
        "verdict": verdict,
        "nom_fichier_reference": resultat_backtest.get("nom_fichier"),
        "axes": {
            "profit_factor": {"reel": round(pf_reel, 2), "statut": axe_pf, "fourchette": seuils["profit_factor"]},
            "winrate_pct": {"reel": winrate_reel, "statut": axe_winrate, "fourchette": seuils["winrate_pct"]},
            "mdd_pct": {"reel": mdd_reel, "statut": axe_mdd, "fourchette": seuils["mdd_pct"]},
        },
    }


def evaluer_epreuve_validation(epreuve, trades, periode_ecoulee=False):
    """
    Verdict et score d'une ÉPREUVE DE VALIDATION -- pas de règles fixes ni
    d'objectif de profit imposé par XTRUNN. La question posée est
    entièrement différente de celle d'une épreuve classique : "l'exécution
    réelle reste-t-elle cohérente avec ce que le backtest de la stratégie
    prédisait ?", pas "as-tu atteint tel objectif".

    Deux mécanismes, dans cet ordre de priorité :
      1. En temps réel, une seule règle dure : ne jamais dépasser le
         drawdown P90 prédit par le bootstrap du backtest -- dérivé
         automatiquement de l'analyse liée, jamais fixé par
         l'utilisateur. Une violation ici est immédiate, comme pour une
         épreuve classique.
      2. En fin de période, le verdict repose directement sur
         calculer_coherence_backtest -- le même mécanisme déjà construit
         pour le Diagnostic d'une épreuve classique, sauf qu'ici ce n'est
         plus une information annexe, c'est LE verdict. "à surveiller"
         reste toléré (zone de marge, pas un vrai signal d'alarme) ;
         seule une "alerte" sur un axe fait échouer la validation.
    """
    trades_tries = sorted(trades, key=lambda t: t["date"])
    profits = np.array([t["profit"] for t in trades_tries], dtype=float)
    capital = epreuve.get("capital_initial", 0) or 0

    courbe_delta = np.cumsum(profits) if len(profits) > 0 else np.array([])
    courbe = capital + courbe_delta
    courbe_complete = np.concatenate(([capital], courbe))
    pic = np.maximum.accumulate(courbe_complete)
    dd_dollars = pic - courbe_complete
    max_dd_dollars = float(dd_dollars.max()) if len(dd_dollars) else 0.0
    max_dd_pct = (max_dd_dollars / capital * 100) if capital > 0 else 0.0

    violations = []
    limite_dd = epreuve.get("drawdown_max_pct")
    if limite_dd and max_dd_pct > limite_dd:
        violations.append(
            f"Drawdown au-delà du pire cas prédit par le backtest : {max_dd_pct:.1f}% "
            f"(limite dérivée du backtest : {limite_dd:.1f}%)"
        )

    jours_trading = len(set(t["date"] for t in trades_tries))
    profit_net = float(profits.sum()) if len(profits) > 0 else 0.0
    profit_pct = (profit_net / capital * 100) if capital > 0 else 0.0

    coherence = None
    if len(profits) > 0:
        stats = calculer_statistiques_detaillees(profits, capital)
        coherence = calculer_coherence_backtest(epreuve, profits, stats)

    raison_echec = None
    if violations:
        statut = "echoue"
        raison_echec = "violation"
    elif periode_ecoulee:
        if coherence is None:
            statut = "echoue"
            raison_echec = "donnees_insuffisantes"
        elif coherence["verdict"] == "alerte":
            statut = "echoue"
            raison_echec = "incoherence"
        else:
            # "coherent" et "a_surveiller" valident tous les deux --
            # "à surveiller" est une zone de marge, pas un signal d'alarme.
            statut = "reussi"
    else:
        statut = "en_cours"

    score = None
    score_detail = None
    if statut != "en_cours":
        if violations or (coherence and coherence["verdict"] == "alerte"):
            score = 0
            score_detail = {"total": 0, "axes": coherence["axes"] if coherence else None}
        elif coherence:
            points_axe = {"coherent": 100 / 3, "a_surveiller": 50 / 3, "indetermine": 50 / 3}
            total = sum(points_axe.get(a["statut"], 0) for a in coherence["axes"].values())
            score = round(total)
            score_detail = {"total": score, "axes": coherence["axes"]}

    if statut == "en_cours":
        score = None
        score_detail = None

    return {
        "statut": statut,
        "raison_echec": raison_echec,
        "profit_pct": round(profit_pct, 2),
        "profit_net": round(profit_net, 2),
        "drawdown_pct": round(max_dd_pct, 2),
        "drawdown_dollars": round(max_dd_dollars, 2),
        "pire_jour_pct": 0.0,
        "consistency_pct": 0.0,
        "jours_trading": jours_trading,
        "n_trades": len(trades_tries),
        "violations": violations,
        "objectif_atteint": None,
        "jours_suffisants": True,
        "courbe": [round(v, 2) for v in courbe_complete],
        "score": score,
        "score_detail": score_detail,
        "coherence_validation": coherence,
    }


def calculer_robustesse_ordre(epreuve_row, trades, rng, n_replicates=N_CI_REPLICATES):
    """Monte Carlo sur l'ORDRE RÉEL des trades de l'épreuve : les mêmes
    résultats, mais dans un ordre différent, auraient-ils quand même
    validé l'épreuve ? Un pourcentage bas révèle un verdict qui a
    peut-être dépendu fortement de l'ordre d'enchaînement plutôt que d'une
    marge de sécurité stable."""
    n = len(trades)
    if n < 5:
        return None

    profits = np.array([t["profit"] for t in trades], dtype=float)
    dates_triees = sorted(t["date"] for t in trades)
    n_reussites = 0

    for _ in range(n_replicates):
        ordre = rng.permutation(n)
        profits_reordonnes = profits[ordre]
        trades_simules = [{"date": dates_triees[i], "profit": float(profits_reordonnes[i])} for i in range(n)]
        evaluation = evaluer_epreuve(epreuve_row, trades_simules, periode_ecoulee=True)
        if evaluation["statut"] == "reussi":
            n_reussites += 1

    return {
        "pct_reussite": round(n_reussites / n_replicates * 100, 1),
        "n_simulations": n_replicates,
    }


def calculer_diagnostic_epreuve(epreuve_row, trades):
    if not trades:
        return None
    trades_array = np.array([t["profit"] for t in trades], dtype=float)
    stats = calculer_statistiques_detaillees(trades_array, epreuve_row.get("capital_initial"))
    rng = np.random.default_rng(RANDOM_SEED)
    return {
        "stats_factuelles": stats,
        "coherence_backtest": calculer_coherence_backtest(epreuve_row, trades_array, stats),
        "robustesse_ordre": calculer_robustesse_ordre(epreuve_row, trades, rng),
    }


def _construire_etat_epreuve(epreuve_row, trades_rows):
    """Combine la config d'une épreuve et ses trades pour produire son
    état complet évalué (statut, score, courbe, violations...)."""
    epreuve = dict(epreuve_row)
    trades = [{"date": t["date"], "profit": t["profit"]} for t in trades_rows]

    maintenant = datetime.now(timezone.utc)
    date_debut = datetime.fromisoformat(epreuve["date_debut"])
    date_fin = datetime.fromisoformat(epreuve["date_fin"])
    periode_ecoulee = maintenant >= date_fin

    if epreuve["type_epreuve"] == "validation":
        evaluation = evaluer_epreuve_validation(epreuve, trades, periode_ecoulee)
    else:
        evaluation = evaluer_epreuve(epreuve, trades, periode_ecoulee)
    statut_stocke = epreuve["statut"]
    epreuve.update(evaluation)
    # Le statut stocké fait autorité s'il a déjà été figé (réussi, échoué,
    # abandonné) -- une réévaluation dynamique ne doit jamais revenir en
    # arrière sur un verdict déjà acté ou un abandon volontaire.
    if statut_stocke != "en_cours":
        epreuve["statut"] = statut_stocke

    jours_ecoules = max(0, (maintenant - date_debut).days)
    epreuve["jours_ecoules"] = min(jours_ecoules, epreuve["duree_jours"])
    epreuve["jours_restants"] = max(0, epreuve["duree_jours"] - jours_ecoules)
    epreuve["progression_pct"] = round(min(100, jours_ecoules / epreuve["duree_jours"] * 100), 1) if epreuve["duree_jours"] > 0 else 100.0
    epreuve["periode_ecoulee"] = periode_ecoulee

    # Nom de la stratégie liée, pour l'affichage -- l'id brut seul ne
    # suffit pas à construire une interface lisible.
    epreuve["strategie_nom"] = None
    if epreuve.get("strategie_id"):
        conn2 = get_db()
        strat = conn2.execute("SELECT nom FROM strategies WHERE id = ?", (epreuve["strategie_id"],)).fetchone()
        conn2.close()
        if strat:
            epreuve["strategie_nom"] = strat["nom"]

    # Score de l'analyse d'origine (Épreuve de Validation uniquement) --
    # pour l'écran de verdict, sans requête séparée.
    epreuve["analyse_score_p10"] = None
    if epreuve.get("analyse_id"):
        conn3 = get_db()
        analyse_row = conn3.execute("SELECT score_p10 FROM analyses WHERE id = ?", (epreuve["analyse_id"],)).fetchone()
        conn3.close()
        if analyse_row:
            epreuve["analyse_score_p10"] = analyse_row["score_p10"]

    return epreuve


class EpreuveCreation(BaseModel):
    nom: str = None
    capital_initial: float = None
    duree_jours: int = 30
    objectif_profit_pct: float = None
    drawdown_max_pct: float = None
    perte_quotidienne_max_pct: float = None
    consistency_max_pct: float = None
    jours_min_trading: int = None
    strategie_id: int = None
    analyse_id: int = None
    type_epreuve: str = "personnalise"


# Règlement standardisé du "Epreuve XTRUNN" -- un jeu de règles fixe,
# façon vraie épreuve de prop firm, plutôt que des seuils personnalisés
# au choix. Imposé ici côté serveur (jamais transmis tel quel par le
# client) pour garantir que deux épreuves XTRUNN officielles sont
# toujours strictement comparables, aujourd'hui comme dans un an.
EPREUVE_XTRUNN_REGLES = {
    "duree_jours": 30,
    "objectif_profit_pct": 10.0,
    "drawdown_max_pct": 10.0,
    "perte_quotidienne_max_pct": 5.0,
    "consistency_max_pct": 40.0,
    "jours_min_trading": 5,
}


@app.post("/epreuves")
async def creer_epreuve(payload: EpreuveCreation):

    # Un Epreuve XTRUNN officiel impose son propre règlement standard,
    # peu importe ce que le client a transmis pour ces champs -- seuls le
    # nom, le capital et le lien optionnel à une stratégie restent au
    # choix de l'utilisateur.
    if payload.type_epreuve == "xtrunn_officiel":
        regles = EPREUVE_XTRUNN_REGLES
        duree_jours = regles["duree_jours"]
        objectif_profit_pct = regles["objectif_profit_pct"]
        drawdown_max_pct = regles["drawdown_max_pct"]
        perte_quotidienne_max_pct = regles["perte_quotidienne_max_pct"]
        consistency_max_pct = regles["consistency_max_pct"]
        jours_min_trading = regles["jours_min_trading"]
    elif payload.type_epreuve == "validation":
        # Aucune règle fixée par XTRUNN ni par l'utilisateur -- tout est
        # dérivé du backtest de l'analyse liée, qui devient obligatoire.
        # Le seul seuil dur (drawdown) vient du pire cas prédit par le
        # bootstrap ; tout le reste du verdict repose sur la cohérence
        # avec le backtest, évaluée à la fin (voir evaluer_epreuve_validation).
        if payload.analyse_id is None:
            raise HTTPException(status_code=400, detail="Une Épreuve de Validation doit être liée à une analyse.")
        conn_verif = get_db()
        analyse_verif = conn_verif.execute(
            "SELECT strategie_id, score_p10, resultat_json FROM analyses WHERE id = ?", (payload.analyse_id,)
        ).fetchone()
        conn_verif.close()
        if analyse_verif is None:
            raise HTTPException(status_code=404, detail="Analyse introuvable.")
        if analyse_verif["score_p10"] is None or analyse_verif["score_p10"] < SEUIL_SCORE_VALIDATION:
            raise HTTPException(
                status_code=400,
                detail=f"Cette analyse n'atteint pas le seuil requis pour la Validation (score P10 ≥ {SEUIL_SCORE_VALIDATION})."
            )
        seuils = json.loads(analyse_verif["resultat_json"]).get("seuils_surveillance")
        if not seuils:
            raise HTTPException(status_code=400, detail="Cette analyse ne contient pas de Seuils de Surveillance exploitables.")
        duree_jours = EPREUVE_XTRUNN_REGLES["duree_jours"]
        objectif_profit_pct = None
        drawdown_max_pct = seuils["mdd_pct"]["p90"]
        perte_quotidienne_max_pct = None
        consistency_max_pct = None
        jours_min_trading = None
        if payload.strategie_id is None:
            payload.strategie_id = analyse_verif["strategie_id"]
        # Le logiciel dérive lui-même tout ce qui peut l'être -- le
        # capital du backtest d'origine, un nom généré à partir de la
        # stratégie -- pour que lancer une Validation depuis l'Analyse ne
        # demande jamais de ressaisir des informations déjà connues.
        if not payload.capital_initial:
            resultat_analyse = json.loads(analyse_verif["resultat_json"])
            payload.capital_initial = resultat_analyse.get("capital_initial")
        if not payload.nom or not payload.nom.strip():
            conn_nom = get_db()
            strat_nom_row = conn_nom.execute("SELECT nom FROM strategies WHERE id = ?", (payload.strategie_id,)).fetchone()
            conn_nom.close()
            nom_strategie = strat_nom_row["nom"] if strat_nom_row else "Stratégie"
            payload.nom = f"Validation — {nom_strategie}"
    else:
        if payload.duree_jours < 1 or payload.duree_jours > 365:
            raise HTTPException(status_code=400, detail="La durée doit être comprise entre 1 et 365 jours.")
        duree_jours = payload.duree_jours
        objectif_profit_pct = payload.objectif_profit_pct
        drawdown_max_pct = payload.drawdown_max_pct
        perte_quotidienne_max_pct = payload.perte_quotidienne_max_pct
        consistency_max_pct = payload.consistency_max_pct
        jours_min_trading = payload.jours_min_trading

    # Vérifié ici, après le branchement -- pour la Validation, ces deux
    # champs viennent d'être dérivés automatiquement s'ils manquaient ;
    # pour les autres types, l'utilisateur doit toujours les fournir.
    if not payload.nom or not payload.nom.strip():
        raise HTTPException(status_code=400, detail="Le nom de l'épreuve est requis.")
    if payload.capital_initial is None or payload.capital_initial <= 0:
        raise HTTPException(status_code=400, detail="Le capital de départ est requis.")

    date_debut = datetime.now(timezone.utc)
    date_fin = date_debut + timedelta(days=duree_jours)

    # Un lien vers une analyse n'a de sens que s'il correspond bien à la
    # stratégie choisie -- sans ce contrôle, une incohérence silencieuse
    # (analyse d'un autre algo) fausserait toute comparaison future avec
    # le backtest d'origine.
    conn = get_db()
    if payload.analyse_id is not None:
        analyse = conn.execute("SELECT strategie_id FROM analyses WHERE id = ?", (payload.analyse_id,)).fetchone()
        if analyse is None:
            conn.close()
            raise HTTPException(status_code=404, detail="Analyse introuvable.")
        if payload.strategie_id is not None and analyse["strategie_id"] != payload.strategie_id:
            conn.close()
            raise HTTPException(status_code=400, detail="Cette analyse n'appartient pas à la stratégie sélectionnée.")

    cur = conn.execute(
        """
        INSERT INTO epreuves
            (nom, strategie_id, analyse_id, capital_initial, objectif_profit_pct, drawdown_max_pct,
             perte_quotidienne_max_pct, consistency_max_pct, jours_min_trading, duree_jours,
             type_epreuve, date_debut, date_fin, statut, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'en_cours', ?)
        """,
        (
            payload.nom.strip(), payload.strategie_id, payload.analyse_id, payload.capital_initial,
            objectif_profit_pct, drawdown_max_pct, perte_quotidienne_max_pct,
            consistency_max_pct, jours_min_trading, duree_jours,
            payload.type_epreuve, date_debut.isoformat(), date_fin.isoformat(), date_debut.isoformat(),
        )
    )
    conn.commit()
    new_id = cur.lastrowid
    row = conn.execute("SELECT * FROM epreuves WHERE id = ?", (new_id,)).fetchone()
    conn.close()
    return _construire_etat_epreuve(row, [])


@app.get("/epreuves")
async def lister_toutes_les_epreuves():
    """Toutes les épreuves, peu importe leur statut."""
    conn = get_db()
    rows = conn.execute("SELECT * FROM epreuves ORDER BY created_at DESC").fetchall()
    resultats = []
    for row in rows:
        trades = conn.execute("SELECT * FROM epreuve_trades WHERE epreuve_id = ? ORDER BY date ASC", (row["id"],)).fetchall()
        resultats.append(_construire_etat_epreuve(row, trades))
    conn.close()
    return {"epreuves": resultats}


@app.get("/epreuves/actives")
async def lister_epreuves_actives():
    """Toutes les épreuves en cours -- pour l'affichage direct à l'écran
    d'accueil."""
    conn = get_db()
    rows = conn.execute("SELECT * FROM epreuves WHERE statut = 'en_cours' ORDER BY date_fin ASC").fetchall()
    resultats = []
    for row in rows:
        trades = conn.execute("SELECT * FROM epreuve_trades WHERE epreuve_id = ? ORDER BY date ASC", (row["id"],)).fetchall()
        etat = _construire_etat_epreuve(row, trades)
        # Une épreuve dont l'issue est désormais connue (réussi/échoué)
        # mais encore marqué "en_cours" en base -> on fige le verdict ici,
        # pour que l'accueil le reflète sans attendre une action manuelle.
        if etat["statut"] in ("reussi", "echoue"):
            conn.execute("UPDATE epreuves SET statut = ? WHERE id = ?", (etat["statut"], row["id"]))
            conn.commit()
        resultats.append(etat)
    conn.close()
    return {"epreuves": resultats}


@app.get("/epreuves/regles-officielles")
async def obtenir_regles_officielles():
    return EPREUVE_XTRUNN_REGLES


@app.get("/epreuves/{epreuve_id}")
async def obtenir_epreuve(epreuve_id: int):
    conn = get_db()
    row = conn.execute("SELECT * FROM epreuves WHERE id = ?", (epreuve_id,)).fetchone()
    if row is None:
        conn.close()
        raise HTTPException(status_code=404, detail="Epreuve introuvable.")
    trades = conn.execute("SELECT * FROM epreuve_trades WHERE epreuve_id = ? ORDER BY date ASC, id ASC", (epreuve_id,)).fetchall()
    etat = _construire_etat_epreuve(row, trades)
    etat["trades"] = [dict(t) for t in trades]

    if etat["statut"] in ("reussi", "echoue") and row["statut"] == "en_cours":
        conn.execute("UPDATE epreuves SET statut = ? WHERE id = ?", (etat["statut"], epreuve_id))
        conn.commit()

    # La mise en scène de révélation ne se déclenche qu'une seule fois,
    # la toute première fois que le verdict est connu -- "revele" se fige
    # à 1 immédiatement après, pour que revisiter l'épreuve plus tard ne
    # rejoue jamais l'animation.
    etat["nouvelle_revelation"] = False
    if etat["statut"] in ("reussi", "echoue") and row["revele"] == 0:
        etat["nouvelle_revelation"] = True
        conn.execute("UPDATE epreuves SET revele = 1 WHERE id = ?", (epreuve_id,))
        conn.commit()
    conn.close()

    # Le diagnostic complet (cohérence backtest, robustesse à l'ordre,
    # récapitulatif factuel) n'a de sens qu'une fois le verdict connu --
    # coûteux à calculer (simulations Monte Carlo) et sans intérêt tant
    # que l'épreuve tourne encore, en plus de risquer de laisser deviner
    # le score caché par la bande.
    if etat["statut"] in ("reussi", "echoue"):
        etat["diagnostic"] = calculer_diagnostic_epreuve(dict(row), etat["trades"])
    else:
        etat["diagnostic"] = None

    return etat


class TradeManuel(BaseModel):
    date: str  # "YYYY-MM-DD"
    profit: float
    symbole: str = None
    notes: str = None


@app.post("/epreuves/{epreuve_id}/trades")
async def ajouter_trade_manuel(epreuve_id: int, trade: TradeManuel):
    conn = get_db()
    row = conn.execute("SELECT * FROM epreuves WHERE id = ?", (epreuve_id,)).fetchone()
    if row is None:
        conn.close()
        raise HTTPException(status_code=404, detail="Epreuve introuvable.")
    conn.execute(
        "INSERT INTO epreuve_trades (epreuve_id, date, profit, symbole, notes, source, created_at) VALUES (?, ?, ?, ?, ?, 'manuel', ?)",
        (epreuve_id, trade.date, trade.profit,
         trade.symbole.strip() if trade.symbole else None,
         trade.notes.strip() if trade.notes else None,
         datetime.now(timezone.utc).isoformat())
    )
    conn.commit()
    conn.close()
    return {"message": "Trade ajouté."}


@app.post("/verification-retroactive")
async def verifier_retroactivement(
    trades_file: UploadFile = File(...),
    plateforme: str = Form("mt5"),
    capital_initial: float = Form(...),
    objectif_profit_pct: float = Form(None),
    drawdown_max_pct: float = Form(None),
    perte_quotidienne_max_pct: float = Form(None),
    consistency_max_pct: float = Form(None),
    jours_min_trading: int = Form(None),
):
    """
    Vérification instantanée d'un historique déjà complet contre un jeu de
    règles -- "cet historique aurait-il validé une épreuve ?" -- sans le
    suivi jour par jour d'un vrai Epreuve en cours. Ne crée rien en
    base : un outil de contrôle ponctuel, pas un objet à retrouver plus
    tard. Réutilise le même moteur d'évaluation (evaluer_epreuve) que
    les Epreuves en direct, pour un verdict strictement cohérent entre
    les deux usages.
    """
    if capital_initial is None or capital_initial <= 0:
        raise HTTPException(status_code=400, detail="Le capital de départ est requis.")

    contents = await trades_file.read()
    parser = obtenir_parser_plateforme(plateforme)
    trades_array, detail_list, err = parser(contents, trades_file.filename)
    if err:
        return {"erreur": err}

    trades = []
    for d in detail_list:
        date_brute = d.get("date")
        if not date_brute:
            continue
        try:
            parsed_date, _ = _parse_dates_robuste([date_brute], dayfirst_default={"mt4": True, "mt5": True, "ctrader": True}.get(str(plateforme or "").strip().lower()))
            if parsed_date.iloc[0] is pd.NaT or pd.isna(parsed_date.iloc[0]):
                continue
            date_parsee = parsed_date.iloc[0].strftime("%Y-%m-%d")
        except Exception:
            continue
        trades.append({"date": date_parsee, "profit": float(d.get("profit", 0.0))})

    if not trades:
        return {"erreur": "Aucun trade exploitable n'a été trouvé dans ce fichier."}

    epreuve = {
        "capital_initial": capital_initial,
        "objectif_profit_pct": objectif_profit_pct,
        "drawdown_max_pct": drawdown_max_pct,
        "perte_quotidienne_max_pct": perte_quotidienne_max_pct,
        "consistency_max_pct": consistency_max_pct,
        "jours_min_trading": jours_min_trading,
    }
    resultat = evaluer_epreuve(epreuve, trades)
    resultat["capital_initial"] = capital_initial
    return resultat


@app.post("/epreuves/{epreuve_id}/trades/import")
async def importer_trades_epreuve(
    epreuve_id: int,
    trades_file: UploadFile = File(...),
    plateforme: str = Form("mt5"),
):
    conn = get_db()
    row = conn.execute("SELECT * FROM epreuves WHERE id = ?", (epreuve_id,)).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail="Epreuve introuvable.")

    contents = await trades_file.read()
    parser = obtenir_parser_plateforme(plateforme)
    trades_array, detail_list, err = parser(contents, trades_file.filename)
    if err:
        return {"erreur": err}

    conn = get_db()
    n_ajoutes = 0
    for d in detail_list:
        date_brute = d.get("date")
        if not date_brute:
            continue
        try:
            parsed_date, _ = _parse_dates_robuste([date_brute], dayfirst_default={"mt4": True, "mt5": True, "ctrader": True}.get(str(plateforme or "").strip().lower()))
            if parsed_date.iloc[0] is pd.NaT or pd.isna(parsed_date.iloc[0]):
                continue
            date_parsee = parsed_date.iloc[0].strftime("%Y-%m-%d")
        except Exception:
            continue
        conn.execute(
            "INSERT INTO epreuve_trades (epreuve_id, date, profit, ticket, source, created_at) VALUES (?, ?, ?, ?, 'import_fichier', ?)",
            (epreuve_id, date_parsee, float(d.get("profit", 0.0)), d.get("ticket"), datetime.now(timezone.utc).isoformat())
        )
        n_ajoutes += 1
    conn.commit()
    conn.close()
    return {"message": f"{n_ajoutes} trade(s) importé(s).", "n_ajoutes": n_ajoutes}


@app.delete("/epreuves/{epreuve_id}/trades/{trade_id}")
async def supprimer_trade_epreuve(epreuve_id: int, trade_id: int):
    conn = get_db()
    cur = conn.execute("DELETE FROM epreuve_trades WHERE id = ? AND epreuve_id = ?", (trade_id, epreuve_id))
    conn.commit()
    deleted = cur.rowcount
    conn.close()
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Trade introuvable.")
    return {"message": "Trade supprimé."}


@app.post("/epreuves/{epreuve_id}/abandonner")
async def abandonner_epreuve(epreuve_id: int):
    conn = get_db()
    cur = conn.execute("UPDATE epreuves SET statut = 'abandonne' WHERE id = ?", (epreuve_id,))
    conn.commit()
    updated = cur.rowcount
    conn.close()
    if updated == 0:
        raise HTTPException(status_code=404, detail="Epreuve introuvable.")
    return {"message": "Épreuve abandonnée."}


class AvancerTempsPayload(BaseModel):
    jours: int = 1


@app.post("/epreuves/{epreuve_id}/avancer-temps")
async def avancer_temps_epreuve(epreuve_id: int, payload: AvancerTempsPayload):
    """
    Outil de test : recule artificiellement date_debut ET date_fin du même
    nombre de jours, ce qui a pour effet d'avancer l'épreuve dans sa
    propre chronologie sans toucher à sa durée totale -- pour vérifier le
    déroulement complet (calendrier, révélation du rapport...) sans
    attendre de vrais jours. Jamais exposé comme une vraie fonctionnalité
    utilisateur, seulement un raccourci de développement/test.
    """
    if payload.jours < 1:
        raise HTTPException(status_code=400, detail="Le nombre de jours doit être positif.")
    conn = get_db()
    row = conn.execute("SELECT * FROM epreuves WHERE id = ?", (epreuve_id,)).fetchone()
    if row is None:
        conn.close()
        raise HTTPException(status_code=404, detail="Epreuve introuvable.")
    nouvelle_date_debut = datetime.fromisoformat(row["date_debut"]) - timedelta(days=payload.jours)
    nouvelle_date_fin = datetime.fromisoformat(row["date_fin"]) - timedelta(days=payload.jours)
    conn.execute(
        "UPDATE epreuves SET date_debut = ?, date_fin = ? WHERE id = ?",
        (nouvelle_date_debut.isoformat(), nouvelle_date_fin.isoformat(), epreuve_id)
    )
    conn.commit()
    conn.close()
    return {"message": f"Épreuve avancée de {payload.jours} jour(s)."}


@app.delete("/epreuves/{epreuve_id}")
async def supprimer_epreuve(epreuve_id: int):
    conn = get_db()
    conn.execute("DELETE FROM epreuve_trades WHERE epreuve_id = ?", (epreuve_id,))
    cur = conn.execute("DELETE FROM epreuves WHERE id = ?", (epreuve_id,))
    conn.commit()
    deleted = cur.rowcount
    conn.close()
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Epreuve introuvable.")
    return {"message": "Épreuve supprimée."}




if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
