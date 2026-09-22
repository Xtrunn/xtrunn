# XTRUNN — Guide complet pour sortir la v1

Ce guide part du principe que tu n'as jamais fait ça. Chaque étape explique le "pourquoi" avant le "comment". Suis-les dans l'ordre — certaines dépendent des précédentes.

**Vue d'ensemble des 6 étapes :**
1. Créer un compte GitHub et y déposer le code
2. Mettre le serveur en ligne (Render)
3. Acheter le domaine (Cloudflare)
4. Connecter le domaine au serveur
5. Empaqueter l'app de bureau en vrai exécutable (Windows + Mac)
6. Mettre les exécutables en téléchargement et finaliser les liens du site

---

## Étape 1 — GitHub : déposer le code

**Pourquoi.** Render (l'étape 2) a besoin de lire ton code depuis quelque part pour le faire tourner. GitHub est l'endroit standard où déposer du code — gratuit, et c'est aussi lui qui te servira à distribuer l'exécutable plus tard.

**Comment.**
1. Va sur [github.com](https://github.com), crée un compte gratuit.
2. Clique sur le bouton vert "New" (ou "+ " en haut à droite → "New repository").
3. Nom du dépôt : `xtrunn`. Laisse-le en "Public" (nécessaire pour que le niveau gratuit de Render fonctionne). Ne coche aucune case d'initialisation (pas de README, pas de licence — on ajoutera ça après).
4. Une fois créé, GitHub te montre une page avec des commandes. Le plus simple si tu ne connais pas Git : clique sur "uploading an existing file" (lien visible sur cette même page) et glisse-dépose directement ces fichiers depuis ton ordinateur :
   - `demo_api.py`
   - `api.py`
   - `essayer.html`
   - `requirements.txt` (fourni avec ce guide)
5. Écris un message en bas ("premier envoi") et clique sur "Commit changes".

Ton code est maintenant en ligne, à une adresse du type `github.com/tonpseudo/xtrunn`.

---

## Étape 2 — Render : faire tourner le serveur

**Pourquoi.** C'est l'endroit où `demo_api.py` va tourner en permanence, pour que n'importe qui puisse utiliser la démo en ligne.

**Comment.**
1. Va sur [render.com](https://render.com), crée un compte (tu peux te connecter directement avec ton compte GitHub, plus simple).
2. Dans le tableau de bord, clique sur "New +" → "Web Service".
3. Render te propose de choisir un dépôt GitHub — sélectionne `xtrunn`.
4. Renseigne :
   - **Name** : `xtrunn` (ou ce que tu veux, ça devient une partie de l'adresse temporaire)
   - **Runtime** : Python 3
   - **Build Command** : `pip install -r requirements.txt`
   - **Start Command** : `uvicorn demo_api:app --host 0.0.0.0 --port $PORT`
   - **Instance Type** : Free
5. Clique sur "Create Web Service". Le tout premier déploiement prend quelques minutes.
6. Une fois prêt, Render te donne une adresse temporaire du type `xtrunn.onrender.com` — ouvre-la, tu dois voir la page de démo XTRUNN fonctionner réellement.

**À savoir** : sur le niveau gratuit, le service s'endort après une période sans visite, et le premier visiteur suivant attend 30-60 secondes que ça se réveille. Normal, pas une panne.

---

## Étape 3 — Cloudflare : acheter le domaine

**Comment.**
1. Va sur [dash.cloudflare.com](https://dash.cloudflare.com), crée un compte.
2. Dans le menu, cherche "Domain Registration" → "Register a Domain".
3. Cherche `xtrunn.com` (ou l'extension que tu préfères), confirme l'achat (~10,46 € pour .com).

---

## Étape 4 — Connecter le domaine à Render

**Pourquoi.** Pour l'instant, `xtrunn.com` ne pointe nulle part, et `xtrunn.onrender.com` fonctionne mais ce n'est pas ton nom. Cette étape relie les deux.

**Comment.**
1. Retourne sur Render, sur la page de ton service → onglet "Settings" → section "Custom Domains".
2. Clique "Add Custom Domain", entre `xtrunn.com` (et `www.xtrunn.com` si tu veux les deux). Render t'indique une valeur à copier (un enregistrement CNAME).
3. Va sur Cloudflare → sélectionne ton domaine → onglet "DNS".
4. Ajoute un enregistrement :
   - Type : `CNAME`
   - Name : `@` (pour xtrunn.com directement) ou `www`
   - Target : la valeur donnée par Render
   - Proxy status : mets-le sur "DNS only" (nuage gris, pas orange) — le proxy orange de Cloudflare peut interférer avec la vérification de domaine de Render au début.
5. Attends quelques minutes à quelques heures (propagation DNS). Render confirme automatiquement une fois que ça fonctionne.

À partir de là, `xtrunn.com` affiche réellement ta démo.

---

## Étape 5 — Empaqueter l'app de bureau en exécutable

**C'est quoi, exactement.** Aujourd'hui, pour lancer XTRUNN sur ton ordinateur, il faut avoir Python installé et taper une commande dans un terminal — ce que quasiment personne ne veut faire. "Empaqueter en exécutable" veut dire regrouper Python + toutes les bibliothèques nécessaires + ton code dans un seul fichier que n'importe qui peut double-cliquer, exactement comme n'importe quel logiciel qu'on installe normalement.

**L'outil** : PyInstaller, le standard pour faire ça avec du Python. Il regarde ton programme et fabrique un dossier (ou un seul fichier) contenant tout le nécessaire.

**Contrainte importante** : je ne peux pas faire cette étape à ta place. Je tourne sur un serveur Linux, donc je ne peux fabriquer ni un `.exe` Windows ni un `.app` Mac — il faut le faire directement sur une vraie machine Windows (pour le .exe) et une vraie machine Mac (pour le .app). Si tu n'as accès qu'à un seul type de machine, tu peux commencer par celui-là et faire l'autre plus tard, ou demander à quelqu'un qui a l'autre système.

### Sur Windows

1. Installe Python depuis [python.org](https://python.org) si ce n'est pas déjà fait (coche bien "Add Python to PATH" pendant l'installation).
2. Crée un dossier, mets-y : `api.py`, `index.html`, `run.py` (fourni avec ce guide).
3. Ouvre une invite de commande (Menu Démarrer → tape "cmd") dans ce dossier, puis :
   ```
   pip install fastapi uvicorn numpy pandas openpyxl python-multipart pydantic pywebview pyinstaller
   ```
4. Lance l'empaquetage :
   ```
   pyinstaller --onefile --windowed --add-data "index.html;." --name XTRUNN run.py
   ```
5. Attends (ça peut prendre plusieurs minutes). Le résultat apparaît dans un nouveau dossier `dist/` — `XTRUNN.exe`.
6. **Teste-le** : double-clique dessus sur cette même machine (ou idéalement une autre machine Windows "propre", sans Python installé, pour vérifier que ça marche vraiment sans dépendance externe).

### Sur Mac

1. Installe Python depuis [python.org](https://python.org) si nécessaire.
2. Même dossier, mêmes 3 fichiers.
3. Ouvre l'app "Terminal", places-toi dans ce dossier (`cd chemin/vers/le/dossier`), puis :
   ```
   pip3 install fastapi uvicorn numpy pandas openpyxl python-multipart pydantic pywebview pyinstaller
   ```
4. Lance l'empaquetage :
   ```
   pyinstaller --onefile --windowed --add-data "index.html:." --name XTRUNN run.py
   ```
   (remarque : `:` au lieu de `;` avant "." — seule différence avec Windows)
5. Résultat dans `dist/XTRUNN.app`.
6. **Teste-le** en double-cliquant.

### Si quelque chose ne fonctionne pas

L'empaquetage Python est connu pour être capricieux (bibliothèque non détectée automatiquement, chemin de fichier incorrect une fois empaqueté, etc.). Si tu obtiens une erreur en testant l'exécutable, copie-colle-la moi — je peux très probablement diagnostiquer le problème et te donner la commande corrigée, même sans pouvoir exécuter PyInstaller moi-même.

---

## Étape 6 — Distribuer l'exécutable et finaliser le site

**Comment.**
1. Retourne sur ton dépôt GitHub (`github.com/tonpseudo/xtrunn`).
2. Clique sur "Releases" (dans le menu de droite) → "Create a new release".
3. Tag : `v1.0`. Titre : "XTRUNN v1.0".
4. Glisse-dépose `XTRUNN.exe` et `XTRUNN.app` (ou un `.zip` du `.app`, plus simple pour Mac) dans la zone de dépôt de fichiers.
5. Clique "Publish release". GitHub te donne des liens de téléchargement permanents et stables pour chaque fichier.
6. Reprends ces deux liens, et remplace les `href="#"` des boutons de téléchargement sur la page "Téléchargement" du site (celle que j'ai construite) par ces vraies adresses. Dis-le-moi et je le fais avec toi à ce moment-là.

---

## Récapitulatif de l'ordre à suivre

Les étapes 1 → 4 peuvent se faire tout de suite et ne dépendent de rien d'autre. L'étape 5 peut se faire en parallèle, dès que tu as accès à une machine Windows et/ou Mac. L'étape 6 arrive en dernier, une fois que tu as au moins un exécutable prêt.
