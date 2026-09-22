# XTRUNN — Récapitulatif complet avant la sortie

## 1. Sur les "versions" — pourquoi il n'y en a en réalité qu'une

Ce n'est pas immédiatement visible, mais il n'existe **pas deux codebases séparées** pour "V1" et "version complète" — et c'est volontaire, pas un oubli.

**Pourquoi je ne recommande pas de séparer en deux fichiers distincts.** Si on avait un `index_v1.html` et un `index_complet.html` qui divergent, chaque futur correctif (un bug trouvé, une amélioration) devrait être appliqué deux fois, dans deux fichiers qui s'écartent un peu plus à chaque fois — exactement le genre de désordre que tu essaies d'éviter en me demandant ce récap.

**Ce qu'on a fait à la place** : une seule codebase, où le mode Validation Officielle existe bien dans le code, mais reste désactivé pour la sortie — invisible, inatteignable, mais pas supprimé. La V1, c'est cette même codebase telle qu'elle est aujourd'hui. Le jour où le mode Validation aura reçu le même travail d'audit et de refonte que le mode Analyse, on le réactive dans cette même base plutôt que de fusionner deux fichiers qui auraient divergé entre-temps.

**Concrètement, tes fichiers actuels SONT la V1** :
- `index.html` — l'app de bureau
- `essayer.html` + `demo_api.py` — la démo web
- `api.py` — le moteur, partagé par les deux

Rien à séparer davantage. Ce sont ces fichiers-là, tels qu'ils sont maintenant, que tu empaquettes et déploies.

---

## 2. Récapitulatif des fichiers — à quoi sert chacun

| Fichier | Usage |
|---|---|
| `api.py` | Le moteur complet (parseurs, calcul du score, tous les piliers). Utilisé par l'app de bureau ET la démo web. |
| `index.html` | Interface de l'app de bureau — historique, stratégies sauvegardées, comparaison de versions. |
| `run.py` | Lance l'app de bureau (ouvre la fenêtre native). |
| `build.bat` | Empaquette l'app de bureau en `.exe` Windows. |
| `xtrunn.ico` | Icône de l'app. |
| `demo_api.py` | Backend de la démo web — même moteur, sans aucune sauvegarde, avec limite anti-abus. |
| `essayer.html` | Interface de la démo web — version simplifiée, pas d'historique. |
| `requirements.txt` | Liste des dépendances pour déployer la démo web sur Render. |
| `GUIDE_MISE_EN_LIGNE.md` | Le guide étape par étape pour la mise en ligne (GitHub, Render, Cloudflare, empaquetage). |

`demo_index.html` traîne encore dans ton dossier — c'est l'ancien nom avant que je le renomme en `essayer.html`. Tu peux le supprimer, il ne sert plus à rien.

---

## 3. Ce que XTRUNN fait aujourd'hui — récapitulatif complet

### Import et compatibilité
Accepte les exports de **MetaTrader 4, MetaTrader 5, cTrader, TradingView, NinjaTrader, Freqtrade**, plus un format personnalisé (CSV/Excel) qui tente de reconnaître automatiquement les colonnes pour toute autre plateforme. Reconnaît les nombres au format FR/EU/US et les dates ambiguës.

### Le score sur 100 — quatre piliers
- **Stabilité Référence → Test (30 points)** — compare la période utilisée pour régler la stratégie à celle qui suit immédiatement, fenêtre par fenêtre (walk-forward chronologique, sans chevauchement).
- **Monte Carlo (30 points)** — réordonne 1000 fois les trades de chaque fenêtre de test séparément, pour mesurer si le résultat dépend d'un enchaînement de trades favorable. Donne un ratio de sensibilité et un facteur de récupération.
- **Risque & Drawdown (20 points)** — le pire ratio profit/drawdown parmi les fenêtres de test (pas juste une moyenne globale qui pourrait masquer une mauvaise fenêtre).
- **Concentration (20 points)** — vérifie si le profit dépend de l'ensemble des trades ou d'une poignée de coups isolés.

Le score est strictement la somme de ces quatre piliers, borné 0-100. Rien d'autre ne le modifie en cachette.

### Fiabilité de l'évaluation — séparée du score
Un échantillon trop petit n'abaisse plus artificiellement le score — à la place, un badge indique si l'évaluation est "fiable", "limitée" ou "insuffisante", avec les raisons précises.

### Stress Pro — diagnostics complémentaires (n'affectent pas le score)
- **Test de fragilité** — retire aléatoirement 10% des trades, 1000 fois, pour voir à quel point le résultat tient sans certains trades.
- **Stress sur les coûts** — teste 4 niveaux de frais (0,5x / 1x / 1,5x / 2x le coût de référence), avec calcul du seuil de rentabilité.
- **Découpe temporelle (rolling)** — des fenêtres qui se chevauchent réellement pour tester la stabilité dans le temps, avec un vrai refus explicite ("historique insuffisant") plutôt qu'un faux résultat à 0% quand il n'y a pas assez de données.

### Diagnostic
- Score global avec détail des 4 piliers et fourchette de variabilité (bootstrap)
- Alertes de santé classées par sévérité, cliquables, qui renvoient directement au pilier concerné
- Biais de sélection multiple (nombre de variantes testées, à titre indicatif)
- Risque de ruine (signaux de payoff, d'escalade de taille, de queue de distribution — diagnostiques, sans impact sur le score)
- Seuils de surveillance pour le suivi en réel
- Analyse comportementale (jour de la semaine, achat/vente, comportement après une perte)
- Export PDF du rapport complet
- Rapport texte copiable, pensé pour être collé dans une IA ou partagé à un humain qui débogue le code
- Export CSV des trades annotés

### Personnalisation
- Mode sombre / clair
- Couleur d'accent personnalisable
- Couleur du trou noir (mode Analyse et mode Validation) personnalisable
- Couleur du rail de navigation et de l'en-tête personnalisables, indépendamment l'une de l'autre

### Persistance (app de bureau uniquement)
- Historique des stratégies et de leurs versions successives
- Comparaison entre versions
- Ajout du code source associé à chaque analyse
- Tout reste en local, rien n'est envoyé nulle part

### Démo web (`essayer.html`)
Même moteur de calcul, sans compte ni sauvegarde — on importe, on voit le résultat, rien n'est conservé après. Limite de 8 analyses par heure par visiteur pour éviter les abus.

### Ce qui existe dans le code mais reste désactivé pour la V1
Le **mode Validation Officielle** — un suivi dans le temps d'une stratégie en conditions réelles, avec ses propres étapes et son propre rail. Il n'a pas encore reçu le même travail de fond (audit méthodologique, refonte visuelle) que le mode Analyse, donc il reste invisible dans l'interface pour cette sortie. Rien à faire de ton côté pour ça — c'est déjà masqué.

---

## 4. Les questions à te poser avant de sortir la V1

Vu que tu dis avoir encore des doutes, voici ce qui, à mon avis, mérite vraiment réflexion avant d'appuyer sur "publier" — pas des bugs, des vraies questions de fond :

- **Le score /100 est-il présenté assez clairement comme "robustesse", pas "qualité" ?** Une stratégie peut avoir un score bas tout en étant rentable — est-ce assez clair pour un nouvel utilisateur qui découvre l'outil sans contexte ?
- **As-tu vraiment testé avec de vrais exports de chaque plateforme annoncée ?** On a un vrai fichier cTrader testé en profondeur. MT4, MT5, TradingView, NinjaTrader et Freqtrade ont été testés avec des données synthétiques construites pour ressembler à ces formats, mais pas encore, à ma connaissance, avec un export réel de chacune.
- **Le guide de mise en ligne te semble-t-il faisable seul ?** Si une étape reste floue une fois que tu t'y mets vraiment, mieux vaut le savoir maintenant qu'au milieu du déploiement.

Le reste — bugs, cohérence de l'interface, calculs — a été audité en profondeur au fil de cette conversation. Ces trois points sont les seuls qui, à mon avis, relèvent encore d'une vraie décision de ta part plutôt que d'un travail technique restant.
