
## Rendement Annualisé
**Formule:** `(1 + rendement_total)^(252/n_jours) - 1`

**Explication:** Combien tu gagnes en moyenne par an si tu gardes ton placement. C'est le premier indicateur à regarder, mais attention, il ne dit rien sur le risque pris pour l'obtenir.

**Interprétation:** Plus c'est haut, mieux c'est. Un rendement de 8% signifie que 100€ investis deviennent 108€ en un an.

---

## Rendement 1Y / 3Y / 5Y
**Formule:** Même formule mais sur les 1, 3 ou 5 dernières années

**Explication:** Permet de voir si le fonds performe bien sur différents horizons de temps. Un fonds peut être bon sur 1 an mais mauvais sur 5 ans (ou l'inverse).

**Interprétation:** Compare les rendements sur différentes périodes. Si 1Y >> 5Y, le fonds s'est amélioré récemment. Si 1Y << 5Y, il a peut-être des difficultés.

---

## Volatilité
**Formule:** `écart-type des rendements × √252`

**Explication:** Est-ce que le fonds fait les montagnes russes ou c'est tranquille ? La volatilité mesure l'amplitude des variations quotidiennes.

**Interprétation:** Plus c'est bas, plus c'est stable. Une vol de 15% signifie que le fonds peut facilement varier de +15% ou -15% sur l'année.

---

## Volatilité 1Y / 3Y / 5Y
**Formule:** Même formule mais sur les 1, 3 ou 5 dernières années

**Explication:** Permet de voir si le fonds est devenu plus ou moins risqué au fil du temps.

**Interprétation:** Si Vol 1Y < Vol 5Y, le fonds s'est calmé récemment. Si Vol 1Y > Vol 5Y, il est devenu plus nerveux.

---

## Sharpe Ratio
**Formule:** `(rendement - taux sans risque) / volatilité`

**Explication:** Est-ce que le risque pris en valait la peine ? C'est le rendement que tu obtiens pour chaque unité de risque. C'est l'indicateur roi en gestion de portefeuille.

**Interprétation:**
- < 0 : mauvais
- 0-1 : moyen
- 1-2 : bon
- > 2 : excellent

---

## Sortino Ratio
**Formule:** `(rendement - taux sans risque) / volatilité négative`

**Explication:** Comme le Sharpe mais on punit seulement les baisses, pas les hausses. C'est plus juste parce qu'on s'en fiche si le fonds monte beaucoup !

**Interprétation:** Mêmes seuils que le Sharpe. Souvent plus élevé car il ignore la "bonne" volatilité.

---

## Semi-Variance
**Formule:** `moyenne des (rendement - moyenne)² pour les rendements < moyenne`

**Explication:** Mesure la dispersion des rendements uniquement quand ça va mal (en-dessous de la moyenne). C'est une mesure du "risque de baisse".

**Interprétation:** Plus c'est bas, mieux c'est. Un fonds avec une faible semi-variance a des baisses plus prévisibles/limitées.

---

## Max Drawdown
**Formule:** `(creux - pic) / pic`

**Explication:** La pire dégringolade qu'on aurait pu subir si on avait acheté au pire moment. C'est le scénario catastrophe.

**Interprétation:** -20% signifie que dans le pire des cas, tu aurais perdu 20% de ton investissement avant que ça remonte.

---

## Beta
**Formule:** `Cov(fonds, marché) / Var(marché)`

**Explication:** Quand le marché bouge de 1%, le fonds bouge de combien ?

**Interprétation:**
- Beta = 1 : pareil que le marché
- Beta > 1 : plus nerveux que le marché
- Beta < 1 : plus calme que le marché

---

## Alpha
**Formule:** `Rendement - (Rf + Beta × (Rm - Rf))`

**Explication:** Le petit plus (ou moins) que le gérant apporte par rapport à ce qu'on attendait vu le risque pris.

**Interprétation:** Positif = le gérant crée de la valeur. Négatif = autant acheter l'indice directement.

---

## Calmar Ratio
**Formule:** `rendement annualisé / |max drawdown|`

**Explication:** Le rendement par rapport à la pire chute subie. Combine performance et risque extrême.

**Interprétation:** > 1 signifie que ton rendement annuel est supérieur à ta pire perte. Plus c'est haut, mieux c'est.

---

## Omega Ratio
**Formule:** `Σ gains au-dessus du seuil / Σ pertes en-dessous`

**Explication:** Pour chaque euro que tu risques de perdre, combien tu peux gagner ? Prend en compte toute la distribution des rendements.

**Interprétation:** Omega > 1 = tu gagnes plus que tu perds en moyenne. Plus c'est haut, mieux c'est.

---

## % Bat le Benchmark
**Formule:** `Nb(jours où rendement_fonds > rendement_benchmark) / Nb(jours total)`

**Explication:** Sur tous les jours de la période, combien de fois le fonds a fait un meilleur rendement quotidien que le benchmark ?

**Interprétation:** 50% = pareil que le marché en moyenne. > 50% = le fonds surperforme régulièrement au quotidien.

---

## Taux Sans Risque (Euribor/EONIA)
**Explication:** C'est le rendement qu'on peut obtenir "sans risque" (en théorie). On l'utilise comme référence : si un fonds fait moins bien que ce taux, autant laisser son argent à la banque !

**Note:** Dans ton fichier, c'est la colonne `DBDCONIA Index`.