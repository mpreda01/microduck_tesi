# TO DO — Fase 2: policy specialiste

*Documento di lavoro, 16 ottobre 2026. Riferimento: `Piano_lavoro_tesi_Preda.md`, sezione 5 (Fase 2) e sezioni 3.2, 3.6, 4.6-4.11.*

## Obiettivo della fase (dal piano)

Una **policy specialista** per ogni situazione selezionata nella fase 1, ottenuta con fine-tuning PPO di `alpha_walking`. Il lavoro si articola in tre passi:
1. fine-tuning di `alpha_walking` in ciascuna situazione; se non basta, training da zero;
2. valutazione di ogni specialista nella propria situazione e sul piano;
3. registrazione dei dati: rollout di `alpha_walking` in tutte le situazioni (per probe, SAE e riconoscitore) e rollout di ciascuna specialista nella propria situazione (per la distillazione).

**Prodotto:** specialiste e dataset.
**Criterio per proseguire:** ogni specialista è nettamente migliore di `alpha_walking` nella propria situazione. Se non lo è, la situazione esce dalla libreria, perché non c'è un comportamento migliore da distillare.

## Punto di partenza (fasi 0 e 1)

- **Fase 0** (`notebooks/fase_0.ipynb`): la batteria su MuJoCo CPU con i servo BAM riproduce le schede (alpha 32/110, rough-walk-g 13/110; carta 32 e 12). La conversione ONNX → PyTorch è verificata, con differenza sotto 1e-5.
- **Fase 1** (`notebooks/fase_1.ipynb`): 68 condizioni, 20 episodi ciascuna, run `bench_v1` di 32 minuti. Criterio soddisfatto: `alpha_walking` peggiora in 6 situazioni. Il notebook ora registra anche i dati per passo della sezione 4.8, nel formato della sezione 4.9.

**Situazioni candidate per le specialiste**, dai risultati di `bench_v1`:

| Situazione | Dove `alpha_walking` peggiora | Note |
|---|---|---|
| Discesa | `face_5deg_downhill` 12/20 cadute, da 7,5° in su 20/20; `pyramid_7deg` 15/20; `trans_down_10deg` 19/20 | la candidata più chiara |
| Salita | `face_7.5-12.5deg_uphill`: non cade, ma non avanza (velocità ~0, a 12,5° scivola indietro) | il peggioramento è nella velocità, non nelle cadute |
| Gradini | `stairs_down_1.5cm` 16/20, `stairs_up_1.5cm` 18/20, da 2 cm in su 20/20 | da decidere: una specialista sola o salita e discesa separate |
| Terreno irregolare | `grid_1.2-1.5cm` 8-10/20; `rubble_1.9-2.5cm` 15-16/20; da 1,3 cm cammina sul posto | |
| Attrito basso | `friction_0.1` 20/20; a 0,2-0,3 non cade ma la potenza raddoppia (scivola); `trans_friction_0.15` 16/20 | |
| Carico | nessuna caduta; potenza fino a 2,6 W col carico davanti; velocità ridotta col carico dietro | peggioramento lieve: da valutare se vale una specialista |

Le **transizioni** non sono una situazione da specialista: servono a valutare riconoscitore e router (fase 4).

---

## Decisione bloccante: da quale checkpoint parte il fine-tuning

Il PPO di rsl_rl riparte da un **checkpoint**, che contiene attore, critico, normalizzazione delle osservazioni e stato dell'ottimizzatore. Il file ONNX non basta. Il checkpoint di `alpha_walking` (`model_9999.pt`, run W&B `yr25mna4`, citato dalla scheda di `rough-walk-e`) non è pubblico.

| Opzione | Pro | Contro |
|---|---|---|
| **A. Chiedere il checkpoint** a Pollen Robotics o a Rémi Fabre | la strada più pulita: si parte esattamente dalla policy ufficiale, con il suo critico | dipende da altri; tempi incerti |
| **B. Ricostruire il checkpoint dall'ONNX:** pesi dell'attore e normalizzazione copiati, critico nuovo | resta la policy ufficiale; non dipende da nessuno | il critico parte da zero: servono alcune iterazioni con l'attore **congelato** mentre il critico impara, altrimenti i primi aggiornamenti peggiorano la policy. Va verificato che la normalizzazione dell'ONNX corrisponda a quella attesa da rsl_rl |
| **C. Riaddestrare** un `alpha_walking` equivalente (strada B della sezione 4.5 del piano) | controllo completo, checkpoint disponibile | la "policy ufficiale" studiata non è più quella distribuita; costa circa un'ora su RTX 5090 |

Proposta: chiedere subito il checkpoint (A) e intanto preparare B. In ogni caso conviene verificare che la policy ripresa dal checkpoint, esportata in ONNX, coincida con `alpha_walking.onnx` (stesse uscite sugli stessi input, come nella fase 0) prima di usarla.

- [ ] Scrivere a Pollen Robotics / Rémi Fabre per il checkpoint `yr25mna4/model_9999.pt`
- [ ] In parallelo: prototipo dell'opzione B (attore + normalizzazione dall'ONNX, critico nuovo, attore congelato per N iterazioni)
- [ ] Verifica: checkpoint ripreso → export ONNX → uscite uguali ad `alpha_walking.onnx`
- [ ] Chiarire la discrepanza: `docs/velstand_policy.md` dice che `alpha_walking.onnx` viene dalla run `441tzs6d`, la scheda di `rough-walk-e` dalla `yr25mna4`

---

## I componenti da realizzare

Quattro pezzi. Il principio è che ogni confronto passi dallo **stesso codice**, così le differenze tra policy non nascono dagli script.

### 1. Valutazione e registrazione, comune a tutte le policy

Base: `fase1_bench.py`, che già garantisce stesse condizioni, episodi accoppiati (stesso terreno, partenza, tensione e rumore iniziale per ogni policy) e stesse metriche.

Da aggiungere:
- [ ] **Policy generica:** accettare non solo un file ONNX ma un "oggetto policy" (ONNX; PyTorch; PyTorch con hook che **modificano** le activation). Servirà nelle fasi 3-5 per valutare `alpha_walking` + vettore di steering, + LoRA, + router con lo stesso codice: è lì che un bias farebbe più danni
- [ ] **Specialiste:** valutarle nella propria situazione e **sul piano**, come chiede il passo 2 della fase 2, e se possibile in tutte le altre situazioni, per vedere che cosa perdono
- [ ] **Esperto che etichetta** (vedi componente 3): far camminare una policy e salvare, per ogni osservazione, anche l'azione che un'altra policy avrebbe scelto
- [ ] Mantenere invariata la registrazione per passo (sezione 4.8) e il formato `.npz` (sezione 4.9)

**Contro il bias:**
- [ ] **Fissare la lista delle condizioni di valutazione prima** di addestrare le specialiste, e non cambiarla dopo, per non regolarle sul test
- [ ] **Semi separati** tra training e valutazione
- [ ] **Valori tenuti fuori** dal training: per esempio discesa addestrata fino a 10° e valutata anche a 12,5° e 15°, oppure attrito addestrato fino a 0,2 e valutato a 0,1. Misura se la specialista generalizza
- [ ] Stesso numero di episodi per tutte le policy; intervalli di confidenza al 95% come nella fase 1

**Nota sui simulatori:** le specialiste si addestrano in mjlab (GPU), ma si valutano nel simulatore CPU con i servo BAM. Vale anche per `alpha_walking`, quindi il confronto resta equo, e in più è un controllo di robustezza tra simulatori. Va però scritto nella tesi.

### 2. Fine-tuning PPO comune, parametrizzato per situazione

Un solo script, in cui le specialiste differiscono **solo** per la situazione.

Uguale per tutte:
- [ ] punto di partenza (il checkpoint scelto sopra)
- [ ] iperparametri PPO (quelli di `MicroduckRlCfg`, eventualmente con tasso di apprendimento ridotto per il fine-tuning)
- [ ] numero di iterazioni (riferimento: 800 come `rough-walk-e`) e numero di copie parallele
- [ ] semi (almeno 2-3 per specialista: il piano chiede media, minimo e massimo sui semi)
- [ ] domain randomization di `alpha_walking` (attrito piedi 0,7-1,3, massa ±5%, spinte, tensione 6,5-8,2 V, ritardo 3-6 passi)
- [ ] reward della camminata di `alpha_walking`, invariati

Diversa per ogni specialista: **solo la configurazione dell'ambiente** (terreno, attrito, carico).

**Gli ambienti di training devono corrispondere alle condizioni del benchmark**, altrimenti la specialista impara una situazione diversa da quella su cui viene valutata:
- [ ] **Discesa e salita:** piramide a **facce piane** (non `HfPyramidSlopedTerrainCfg`, che ha le pieghe sotto i piedi); pendenze con un curriculum fino al massimo addestrato
- [ ] **Gradini e terreno irregolare:** gli stessi tipi del benchmark (`BoxPyramidStairsTerrainCfg`, `BoxRandomGridTerrainCfg`, ciottoli `HfRandomUniformTerrainCfg`), stesse forme, altezze nel range di addestramento
- [ ] **Attrito:** impostato **sia sui piedi sia sul terreno** (in MuJoCo vale il massimo dei due). L'evento `foot_friction` di training randomizza solo i piedi: va sostituito o affiancato
- [ ] **Carico:** corpo rigido saldato al tronco, posto nel centro di massa oppure spostato, come nel benchmark; massa e posizione estratte per episodio

**Restare vicini ad `alpha_walking`:** il piano conta sul fatto che la specialista cambi solo il necessario, così la differenza da colmare con un vettore è piccola.
- [ ] Valutare un tasso di apprendimento basso, oppure un termine che ancori la policy all'originale (come l'ancora di imitazione di `velstand`, `docs/velstand_policy.md`)
- [ ] Misurare quanto la specialista si allontana: differenza media tra le sue azioni e quelle di `alpha_walking` sugli stessi stati, e camminata sul piano

**Export:**
- [ ] ogni specialista esportata in ONNX con `scripts/export.py`, stesso formato di `alpha_walking` (61 → 14, normalizzazione inclusa), così entra direttamente nella valutazione comune
- [ ] registrare per ogni specialista: commit, task, semi, iterazioni, ore di GPU (servono per il "costo di una situazione nuova" della fase 5)

### 3. Raccolta ed etichettatura dei dati per la distillazione

Dal passo 3 della fase 2 e dalla sezione 3.2 (passi 2 e 4) del piano:

- [ ] **Rollout di `alpha_walking` in tutte le situazioni** (per probe, SAE e riconoscitore): già prodotti dalla fase 1 con `SAVE_STEPS = True`. Va verificato che siano abbastanza: il piano indica circa un milione di passi come punto di partenza per il SAE
- [ ] **Rollout di ogni specialista nella propria situazione**, con le coppie (osservazione, azione della specialista): il dataset del passo 2 della distillazione
- [ ] **Etichettatura per DAgger** (fase 3, passo 4 della sezione 3.2): far camminare `alpha_walking` (poi `alpha_walking` + vettore) e salvare, per ogni osservazione, l'azione che **avrebbe scelto la specialista**. Non serve una nuova simulazione: basta dare l'osservazione alla rete della specialista. Si può fare come opzione "esperto che etichetta" dello script di valutazione, senza scriverne uno nuovo
- [ ] Formato: lo stesso `.npz` della fase 1, con in più un campo `expert_action` (14 numeri) quando c'è un esperto

### 4. Script SAE (appartiene alla fase 3, ma si può preparare)

Nel piano il SAE è il primo passo della fase 3. Prepararlo ora va bene, tenendo presente che:
- [ ] si addestra sulle activation di **`alpha_walking`** (non delle specialiste) in **tutte** le situazioni, **bilanciate**: se il piano domina il dataset, domina anche il dizionario
- [ ] va scelto il layer: si parte da quello in cui si farà lo steering (il piano usa come esempio il secondo, 256 numeri), ma conviene un SAE per ciascuno dei tre
- [ ] libreria: `overcomplete` oppure un SAE scritto a mano (sezione 4.2)
- [ ] **controllo obbligatorio** (sezione 3.6): la policy con l'activation sostituita dalla sua ricostruzione, senza alcuna modifica, deve camminare come prima. Il controllo passa dallo script di valutazione comune (componente 1, con un hook che sostituisce l'activation)
- [ ] verificare che la quantità di dati basti: la qualità della ricostruzione deve smettere di migliorare aggiungendo dati (sezione 4.9)

---

## Ordine di lavoro proposto

1. [ ] **Checkpoint:** richiesta (A) e prototipo di ricostruzione (B); verifica di equivalenza con `alpha_walking.onnx`
2. [ ] **Scelta delle situazioni** per le specialiste, dalla tabella sopra; lista delle condizioni di valutazione **fissata** prima del training, con i valori tenuti fuori
3. [ ] **Ambienti di training in mjlab** per ogni situazione, allineati al benchmark (facce piane, attrito su piedi e terreno, carico)
4. [ ] **Script di fine-tuning comune**; prova breve su una situazione (la discesa) per misurare tempo e stabilità del primo avvio. È anche il punto 4 della fase 0, rimasto aperto
5. [ ] **Valutazione comune estesa** a policy generiche; valutazione della prima specialista nella sua situazione e sul piano
6. [ ] Se la prima specialista supera il criterio: le altre situazioni, con lo stesso script
7. [ ] **Raccolta dati:** rollout delle specialiste; opzione "esperto che etichetta" pronta per la fase 3
8. [ ] (Anticipo della fase 3) **script SAE** e controllo della ricostruzione

## Criteri e misure da riportare

- [ ] Per ogni specialista, nella propria situazione: cadute, errore di velocità, potenza, con IC 95%, contro `alpha_walking` sugli stessi episodi
- [ ] Sul piano: la specialista non deve peggiorare in modo significativo
- [ ] Sui valori tenuti fuori dal training: quanto generalizza
- [ ] Vicinanza ad `alpha_walking`: differenza media delle azioni sugli stessi stati
- [ ] Costo: ore di GPU per specialista (fine-tuning) e per dati
- [ ] Decisione per ciascuna situazione: **entra nella libreria** solo se la specialista è nettamente migliore

## Rischi e domande aperte

- **Checkpoint non disponibile e ricostruzione instabile:** il critico nuovo può destabilizzare i primi aggiornamenti. Mitigazione: attore congelato all'inizio, tasso di apprendimento basso.
- **Distanza tra simulatori:** una specialista può sfruttare dettagli di mjlab che il simulatore CPU con i servo BAM non riproduce. Da controllare valutando le specialiste anche in mjlab.
- **Specialista troppo lontana da `alpha_walking`:** il vettore non riuscirebbe a colmare la differenza (fase 3). Da monitorare con la vicinanza delle azioni.
- **Carico e salita:** il peggioramento di `alpha_walking` è nella potenza o nella velocità, non nelle cadute. Va deciso se basta per una specialista o se queste situazioni escono dalla libreria.
- **Gradini:** una specialista per salita e discesa insieme, oppure due separate?
- **Fase 0, punto 4 ancora aperto:** il tempo di un fine-tuning su RTX 3090 e RTX 5090 si misura con la prima specialista.
