# Risultati della Fase 1: benchmark di situazioni

*Matteo Preda — 16 ottobre 2026. Riferimento: `Piano_lavoro_tesi_Preda.md`, sezione 5 (Fase 1). Dati: run `bench_v1` del notebook `notebooks/fase_1.ipynb`, raccolti in `docs/results_fase1.xlsx`.*

## In breve

- La policy ufficiale `alpha_walking` è stata valutata su **68 condizioni**, 20 episodi ciascuna. Le condizioni coprono piano, terreno irregolare, gradini, pendenze, pavimento scivoloso, carico e transizioni durante la camminata. La variante della comunità `rough-walk-g` è stata valutata sugli stessi episodi come confronto.
- **Il criterio per proseguire è soddisfatto.** `alpha_walking` peggiora in modo misurabile rispetto al piano in **6 situazioni** (il piano ne chiede almeno 3): terreno irregolare, gradini, pendenze, attrito basso, carico e transizioni.
- Per ogni situazione la difficoltà copre l'intera gamma, da 0 a 20 cadute su 20, quindi si vede dove la policy inizia a cedere.
- `rough-walk-g` riesce in molte condizioni in cui `alpha_walking` cade. Esiste quindi, per lo stesso robot con le stesse osservazioni, un comportamento migliore da imparare: è la premessa delle policy specialiste della fase 2.
- **Non servono terreni più difficili.** Servono più risoluzione e più episodi nella zona di soglia, più due verifiche mirate (sezione "Considerazioni").

---

## 1. Metodo

### Simulatore

Si usa lo stesso simulatore validato nella **fase 0**: MuJoCo su CPU, il modello del robot di `microduck_rl` (commit `6cd45fc`, branch `hostile-terrain`) e i servomotori XL330 simulati con il modello `m6` del pacchetto `bam` (controllo in tensione, limite di corrente del firmware, attrito dipendente dal carico). Le osservazioni sono le stesse 61 della policy e il controllo gira a 50 Hz.

È il protocollo con cui la scheda di `rough-walk-g` riporta i suoi risultati. Nella fase 0 la batteria di 110 prove ricostruita su questo simulatore ha dato:

| Policy | Cadute (fase 0) | Cadute (scheda) | Potenza sul piano (fase 0) | Potenza (scheda) |
|---|---|---|---|---|
| `alpha_walking` | 32/110 | 32/110 | 1,01 W | 1,03 W |
| `rough-walk-g` | 13/110 | 12/110 | 1,42 W | 1,21 W |

Il divario tra le due policy è riprodotto. La potenza assoluta differisce: la scheda non specifica come la calcola, qui è la potenza meccanica Σ|coppia × velocità dei giunti|.

### Condizioni

| Situazione (piano) | Condizioni | Valori |
|---|---|---|
| Piano (riferimento) | `flat` | — |
| Terreno irregolare | griglia di blocchi, ciottoli, onde | griglia ±0,3–1,5 cm; ciottoli ±0,2–2,5 cm; onde 0,5–2 cm di ampiezza |
| Gradini | scale a piramide in discesa (partenza in cima) e in salita (partenza dal bordo) | gradini da 1,0 a 3,0 cm, pedata 12 cm |
| Pendenze | piramide del training percorsa dalla cima; piramide a **facce piane** percorsa in salita o in discesa | piramide 3–9°; facce 0–15° |
| Pavimento scivoloso | attrito impostato **sia sui piedi sia sul terreno** | da 0,6 a 0,1 (minimo del training: 0,7) |
| Carico | massa fissata al tronco, **sul centro di massa** (centro di massa invariato) oppure **spostata** in avanti, indietro, di lato, in alto | 10–30% della massa del robot (0,737 kg); centro di massa spostato fino a 11,5 mm in orizzontale e 18,5 mm in alto |
| Transizioni | piano → pendenza → piano; il pavimento diventa scivoloso a 5 s; un carico compare a 5 s | pendenze 5° e 10°; attrito 0,3 e 0,15; carico 30% sul centro di massa o dietro |

Note sulle scelte:
- **Piramide a facce piane.** La piramide del generatore di mjlab ha le pieghe sulle linee centrali delle facce: un robot che sale o scende al centro di una "faccia" ha i piedi su due inclinazioni laterali opposte, pari a circa metà della pendenza nominale (misurato in MuJoCo). Per le pendenze con direzione è stata quindi costruita una piramide con facce piane, verificata in simulazione: pendenza esatta e nessuna inclinazione laterale.
- **Rampa di `slope_terrain.py` non usata.** Era stata scritta per il task con i pattini e dava cadute anomale già a 2,5°. Le transizioni usano una striscia a heightfield, con la stessa fisica di contatto delle altre pendenze.
- **Attrito.** In MuJoCo, quando due superfici si toccano, l'attrito usato è il massimo dei due coefficienti. Per questo è stato impostato lo stesso valore su piedi e terreno, verificandolo sui contatti simulati.
- **Vibrazioni** (opzionali nel piano): non incluse.

### Episodi e metriche

- **Episodi:** 20 per condizione, da 20 s (30 s per le transizioni di terreno). Comando costante di 0,3 m/s in avanti, con mantenimento della direzione: il comando di rotazione corregge l'errore di direzione, come un operatore con il joystick.
- **Variabilità:** la simulazione è deterministica, quindi ogni episodio estrae una tensione della batteria (7,0–7,8 V), una piccola deviazione degli angoli iniziali e il punto di partenza. Le estrazioni dipendono solo da condizione ed episodio: **le due policy vedono esattamente gli stessi episodi** (confronto accoppiato).
- **Caduta:** inclinazione del tronco oltre 70°, la stessa terminazione del training.
- **Metriche:**
  - cadute, con intervallo di confidenza al 95% (Wilson);
  - errore di velocità, calcolato su finestre di 1 s per togliere l'oscillazione del passo;
  - velocità lungo l'asse del robot;
  - potenza meccanica.

  Velocità, errore e potenza sono mediati **solo sugli episodi completati**, perché durante una caduta misurerebbero la caduta stessa.
- **"Peggiore del piano":**
  - gli intervalli delle cadute non si sovrappongono; oppure
  - l'errore di velocità è maggiore di almeno 0,02 m/s con test di Welch p < 0,01; oppure
  - la potenza è almeno 1,1 volte quella sul piano, con p < 0,01.

---

## 2. Risultati

Nelle tabelle α indica `alpha_walking` e g indica `rough-walk-g`. Le celle vuote corrispondono a condizioni in cui tutti gli episodi cadono, quindi senza metriche di camminata.

### Riferimento: piano

| | Cadute | Errore vel. (m/s) | Velocità (m/s) | Rotazione (rad/s) | Potenza (W) |
|---|---|---|---|---|---|
| α | 0/20 | 0,198 | 0,103 | 0,000 | 1,12 |
| g | 0/20 | 0,175 | 0,126 | 0,039 | 1,38 |

Entrambe raggiungono circa un terzo o metà della velocità comandata (0,3 m/s), coerente con le schede (0,11–0,12 m/s). `rough-walk-g` consuma il 24% in più sul piano (la scheda dice +17%) e tende a girare a sinistra anche con il mantenimento della direzione.

### Terreno irregolare

| Condizione | Cadute α | Cadute g | Vel. α | Vel. g | Potenza α | Potenza g |
|---|---|---|---|---|---|---|
| griglia 0,3 cm | 0/20 | 0/20 | 0,107 | 0,127 | 1,11 | 1,35 |
| griglia 0,6 cm | 1/20 | 0/20 | 0,088 | 0,124 | 1,14 | 1,41 |
| griglia 0,9 cm | 4/20 | 1/20 | 0,098 | 0,111 | 1,19 | 1,40 |
| griglia 1,2 cm | **8/20** | 6/20 | 0,090 | 0,110 | 1,29 | 1,50 |
| griglia 1,5 cm | **10/20** | 5/20 | 0,094 | 0,123 | 1,35 | 1,41 |
| ciottoli 0,2 cm | 0/20 | 0/20 | 0,128 | 0,125 | 1,06 | 1,20 |
| ciottoli 0,75 cm | 0/20 | 0/20 | 0,085 | 0,102 | 0,97 | 1,16 |
| ciottoli 1,3 cm | 3/20 | 0/20 | **0,013** | **0,013** | 0,87 | 1,02 |
| ciottoli 1,9 cm | **16/20** | 0/20 | 0,000 | 0,007 | 0,98 | 1,05 |
| ciottoli 2,5 cm | **15/20** | 2/20 | 0,000 | 0,003 | 1,07 | 1,18 |
| onde 0,5–2,0 cm | 0/20 | 0/20 | 0,119–0,129 | 0,116–0,129 | 1,05–1,08 | 1,16–1,22 |

- Sui ciottoli da 1,3 cm in su **entrambe camminano sul posto**: velocità circa 0 e errore pari al comando. `alpha_walking` in più cade (16/20 a 1,9 cm), `rough-walk-g` resta in piedi.
- Le onde, fino a 2 cm, non creano difficoltà a nessuna delle due.

### Gradini

| Condizione | Cadute α | Cadute g | Vel. α | Vel. g | Potenza α | Potenza g |
|---|---|---|---|---|---|---|
| discesa 1,0 cm | 0/20 | 0/20 | 0,206 | 0,123 | 2,92 | 1,47 |
| salita 1,0 cm | 0/20 | 0/20 | 0,013 | 0,025 | 1,24 | 1,36 |
| discesa 1,5 cm | **16/20** | 1/20 | 0,248 | 0,168 | 4,49 | 1,89 |
| salita 1,5 cm | **18/20** | 11/20 | −0,001 | 0,085 | 1,28 | 1,69 |
| discesa 2,0 cm | **20/20** | 12/20 | | 0,226 | | 3,29 |
| salita 2,0 cm | **20/20** | 20/20 | | | | |
| 2,5 e 3,0 cm (salita e discesa) | **20/20** | 20/20 | | | | |

- **In discesa** `alpha_walking` accelera (0,21–0,25 m/s) e consuma 3–4 volte la potenza sul piano già con gradini da 1 cm, poi cade da 1,5 cm. `rough-walk-g` scende in modo più controllato.
- **In salita** nessuna delle due avanza già da 1 cm. Da 2 cm in su cadono entrambe: probabilmente è un limite fisico del robot, che solleva i piedi di 1–2 cm (in simulazione circa 12 mm).

### Pendenze

| Condizione | Cadute α | Cadute g | Vel. α | Vel. g | Potenza α | Potenza g |
|---|---|---|---|---|---|---|
| piramide 3° (dalla cima) | 0/20 | 0/20 | 0,159 | 0,168 | 1,36 | 1,43 |
| piramide 5° | 4/20 | 0/20 | 0,207 | 0,187 | 2,71 | 1,56 |
| piramide 7° | **15/20** | 0/20 | 0,257 | 0,215 | 4,71 | 2,18 |
| piramide 9° | **20/20** | 2/20 | | 0,249 | | 3,22 |
| discesa 0° | 0/20 | 0/20 | 0,124 | 0,131 | 1,10 | 1,26 |
| discesa 2,5° | 0/20 | 0/20 | 0,163 | 0,173 | 1,33 | 1,45 |
| discesa 5° | **12/20** | 0/20 | 0,194 | 0,202 | 2,34 | 1,62 |
| discesa 7,5° | **20/20** | 0/20 | | 0,267 | | 3,34 |
| discesa 10° | **20/20** | 3/20 | | 0,515 | | 9,04 |
| discesa 12,5° e 15° | **20/20** | 20/20 | | | | |
| salita 0° | 0/20 | 0/20 | 0,124 | 0,131 | 1,11 | 1,27 |
| salita 2,5° | 0/20 | 0/20 | 0,100 | 0,097 | 1,02 | 1,13 |
| salita 5° | 0/20 | 0/20 | 0,071 | 0,073 | 0,94 | 1,03 |
| salita 7,5° | 0/20 | 0/20 | **0,033** | 0,051 | 0,86 | 0,93 |
| salita 10° | 1/20 | 2/20 | **−0,001** | 0,135 | 0,81 | 2,93 |
| salita 12,5° | 0/20 | 2/20 | **−0,118** | −0,039 | 1,17 | 0,99 |
| salita 15° | **20/20** | 16/20 | | −0,187 | | 3,35 |

- **La discesa è la situazione più netta.** `alpha_walking`, mai addestrata su pendenze, cade già a 5° (12/20) e sempre da 7,5°. `rough-walk-g`, addestrata su pendenze fino a circa 9°, regge fino a 10°.
- **In salita** `alpha_walking` non cade fino a 12,5°, ma la velocità scende fino a zero (10°) e poi diventa negativa: scivola indietro. Il peggioramento qui è nella velocità, non nelle cadute.
- Le due piramidi danno risultati coerenti. La piramide del training percorsa dalla cima è un po' più facile delle facce piane, perché il robot parte da una piattaforma piana e incontra la pendenza gradualmente.

### Pavimento scivoloso

| Attrito | Cadute α | Cadute g | Vel. α | Vel. g | Potenza α | Potenza g |
|---|---|---|---|---|---|---|
| 0,6 | 0/20 | 0/20 | 0,101 | 0,123 | 1,08 | 1,38 |
| 0,5 | 0/20 | 0/20 | 0,101 | 0,123 | 1,09 | 1,40 |
| 0,4 | 0/20 | 0/20 | 0,100 | 0,128 | 1,16 | 1,37 |
| 0,3 | 0/20 | 0/20 | 0,123 | 0,139 | **1,97** | 1,50 |
| 0,2 | 0/20 | 0/20 | 0,131 | 0,110 | **2,38** | 1,90 |
| 0,1 | **20/20** | 2/20 | | −0,020 | | 1,42 |

- Fino a 0,4 `alpha_walking` non mostra effetti.
- A 0,3–0,2 non cade e cammina perfino più veloce, ma la **potenza raddoppia**: i piedi scivolano e la policy compensa.
- A 0,1 cade sempre. `rough-walk-g` resta in piedi ma non avanza.

### Carico

| Carico | Spostamento centro di massa (mm) | Cadute α / g | Vel. α | Vel. g | Potenza α | Potenza g | Rotazione g (rad/s) |
|---|---|---|---|---|---|---|---|
| 10% sul centro di massa | 0 | 0/20 / 0/20 | 0,108 | 0,131 | 1,17 | 1,44 | 0,03 |
| 20% sul centro di massa | 0 | 0/20 / 0/20 | 0,111 | 0,138 | 1,25 | 1,49 | 0,02 |
| 30% sul centro di massa | 0 | 0/20 / 0/20 | 0,115 | 0,139 | 1,33 | 1,51 | 0,01 |
| 20% avanti | +8,3 avanti | 0/20 / 0/20 | 0,124 | 0,166 | **2,02** | 2,14 | 0,36 |
| 20% dietro | −8,3 | 0/20 / 0/20 | 0,101 | 0,098 | 0,89 | 1,11 | 0,28 |
| 20% di lato | +8,3 a sinistra | 0/20 / 0/20 | 0,113 | 0,141 | 1,53 | 1,49 | **0,70** |
| 20% in alto | +13,3 in alto | 0/20 / 0/20 | 0,093 | 0,121 | 1,26 | 1,38 | 0,00 |
| 30% avanti | +11,5 avanti | 0/20 / 0/20 | 0,133 | 0,201 | **2,57** | 2,97 | 0,42 |
| 30% dietro | −11,5 | 0/20 / 0/20 | **0,078** | 0,072 | 0,87 | 1,07 | 0,05 |
| 30% di lato | +11,5 a sinistra | 0/20 / 0/20 | 0,115 | 0,148 | 1,75 | 1,54 | **1,02** |
| 30% in alto | +18,5 in alto | 0/20 / 0/20 | 0,091 | 0,125 | 1,31 | 1,49 | 0,04 |

- **Nessuna caduta**, per nessuna delle due policy.
- Il carico **sul centro di massa** aumenta solo la potenza, in proporzione alla massa (+19% al 30%).
- Il carico **spostato** pesa molto di più:
  - in avanti la potenza più che raddoppia (+130% al 30%);
  - indietro il robot rallenta;
  - di lato `rough-walk-g` gira fino a 1 rad/s, nonostante il mantenimento della direzione.
- La posizione del carico conta più della sua massa.

### Transizioni durante la camminata

| Condizione | Cadute α | Cadute g | Note |
|---|---|---|---|
| piano → discesa 5° → piano | 0/20 | 0/20 | α +19% di potenza |
| piano → salita 5° → piano | 0/20 | **16/20** | anomalia di g, da verificare (vedi sotto) |
| piano → discesa 10° → piano | **19/20** | 0/20 | α cade sulla pendenza (circa 9 s) |
| piano → salita 10° → piano | 0/20 | 0/20 | α quasi si ferma (0,031 m/s) |
| attrito 1,0 → 0,3 a 5 s | 0/20 | 0/20 | α +54% di potenza |
| attrito 1,0 → 0,15 a 5 s | **16/20** | 4/20 | tutte le cadute avvengono dopo il cambio |
| carico 30% sul centro di massa a 5 s | 0/20 | 0/20 | α +14% di potenza |
| carico 30% dietro a 5 s | 0/20 | 0/20 | velocità ridotta per entrambe |

Le transizioni confermano i risultati a condizione fissa: la stessa policy cede quando la situazione cambia durante la camminata. Saranno il banco di prova del riconoscitore e del router (fase 4).

### Criterio della fase 1

| Situazione | `alpha_walking` peggiora? | Come |
|---|---|---|
| Terreno irregolare | sì | cadute (griglia ≥ 1,2 cm, ciottoli ≥ 1,9 cm); si ferma sui ciottoli ≥ 1,3 cm |
| Gradini | sì | cadute da 1,5 cm; potenza ×2,6 già in discesa da 1 cm |
| Pendenze | sì | cadute in discesa da 5°; si ferma in salita da 10° |
| Pavimento scivoloso | sì | potenza ×2 a 0,2–0,3; cadute a 0,1 |
| Carico | sì | potenza (fino a ×2,3 col carico avanti), velocità (carico dietro); nessuna caduta |
| Transizioni | sì | cadute in discesa a 10° e con l'attrito che scende a 0,15 |

**6 situazioni su 6: criterio soddisfatto** (ne servono almeno 3).

---

## 3. Considerazioni

### Servono terreni più difficili?

**No.** Il materiale è sufficiente, per tre motivi:

1. **Per `alpha_walking` la scala di difficoltà è già esaurita.** Arriva a 20/20 cadute con scale da 2 cm, discesa da 7,5°, piramide a 9° e attrito 0,1. Terreni più difficili aggiungerebbero solo altri 20/20, senza informazione.
2. **Esiste già la prova che si può fare meglio.** `rough-walk-g`, stesso robot e stesse osservazioni, riesce dove `alpha_walking` cade:

   | Condizione | `rough-walk-g` | `alpha_walking` |
   |---|---|---|
   | discesa 7,5° | 0/20 | 20/20 |
   | discesa 10° | 3/20 | 20/20 |
   | transizione in discesa 10° | 0/20 | 19/20 |
   | attrito 0,1 | 2/20 | 20/20 |
   | ciottoli 1,9 cm | 0/20 | 16/20 |
   | scale in discesa 1,5 cm | 1/20 | 16/20 |

   Una specialista ottenuta con il fine-tuning di `alpha_walking` su queste situazioni ha buone probabilità di superare il criterio della fase 2.
3. **Dove falliscono entrambe** (scale da 2,5 cm, discesa a 12,5–15°, salita a 15°), il limite è probabilmente fisico: il robot solleva i piedi di 1–2 cm. Spingersi oltre produrrebbe situazioni in cui nemmeno una specialista migliora, che il piano scarterebbe comunque.

### Cosa conviene aggiungere prima della fase 2

1. **Più risoluzione nella zona di soglia**, dove le cadute passano da 0 a 20. È lì che si misurerà l'effetto graduale delle correzioni (prova di steering, sezione 3.5 del piano). Oggi i passi sono grossolani:
   - discesa: 2,5° → 0/20, 5° → 12/20, 7,5° → 20/20. Proposta: 3, 4, 5, 5,5, 6, 6,5, 7°;
   - scale: 1,0 → 0/20, 1,5 → 16–18/20. Proposta: 1,0, 1,25, 1,5, 1,75 cm;
   - attrito: tra 0,2 (0/20) e 0,1 (20/20). Proposta: 0,175, 0,15, 0,125;
   - ciottoli: tra 1,3 e 1,9 cm.
2. **Più episodi nelle condizioni di soglia** (da 20 a 50–100). Con 20 episodi lo 0/20 ha un intervallo di confidenza 0–16%, troppo largo per misurare il "vantaggio recuperato" della fase 3. Il piano ragiona su 100 prove. Basta farlo sulle condizioni di soglia, non su tutte le 68.
3. **Carico più pesante, come verifica.** `alpha_walking` non cade mai, nemmeno con il 30% spostato. Una prova al 40–50%, o con uno spostamento maggiore, dirà se esiste una zona di caduta. Se non c'è, il carico resta una questione di efficienza (potenza, velocità) e va deciso se merita una specialista.
4. **Verificare l'anomalia di `trans_up_5deg` per `rough-walk-g`**: 16/20 cadute a 5° e 0/20 a 10°. Prima di usare questa condizione va capito se è un comportamento della policy o un difetto del terreno, per esempio al cambio di pendenza in cima. I dati registrati a ogni passo (pendenza misurata sotto il robot al momento della caduta) e il video lo chiariscono.

### Candidate per le policy specialiste (fase 2)

| Situazione | Motivazione | Priorità |
|---|---|---|
| **Discesa** | il peggioramento più netto (cadute da 5°); `rough-walk-g` dimostra che si può fare meglio; è l'esempio guida del piano | alta |
| **Gradini in discesa** | cadute da 1,5 cm, potenza ×2,6 già a 1 cm; `rough-walk-g` migliora | alta |
| **Terreno irregolare (ciottoli, griglia)** | cadute ≥ 1,2–1,9 cm; `rough-walk-g` non cade | alta |
| **Attrito basso** | potenza ×2 a 0,2–0,3, cadute a 0,1; `rough-walk-g` resiste fino a 0,1 | media |
| **Carico spostato** | solo potenza e velocità, nessuna caduta: da decidere dopo la verifica con carichi maggiori | da decidere |
| **Salita (pendenze, gradini)** | il robot si ferma ma non cade; nemmeno `rough-walk-g` avanza sui gradini: forse è un limite fisico | da decidere |

Le **transizioni** non sono una situazione da specialista: servono a valutare il riconoscitore e il router.

### Limiti da dichiarare

- **Simulatore proxy.** MuJoCo su CPU con i servo BAM è il protocollo delle schede, non quello del training (mjlab su GPU). Il confronto è equo perché entrambe le policy sono state addestrate in mjlab, ma i valori assoluti possono differire da quelli del simulatore di training.
- **20 episodi per condizione:** intervalli di confidenza larghi (vedi sopra).
- **Comando costante** di 0,3 m/s con mantenimento della direzione: non copre i cambi di velocità e le rotazioni comandate.
- **Carico** modellato come massa puntiforme saldata al tronco.
- **Vibrazioni:** non incluse (opzionali nel piano).


