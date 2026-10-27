# TO DO — Fase 2: policy specialiste

*Matteo Preda — 20 ottobre 2026. Riferimenti: [Piano di lavoro](Piano_lavoro_tesi_Preda.md), sezione 3.2 (passi 1-2) e sezione 5 (Fase 2); [risultati della fase 1](RESULTS_fase1.md); [struttura della repository](../REPO_STRUCTURE.md).*

## Obiettivo, prodotto e criterio

- **Obiettivo:** una policy specialista per ogni situazione della libreria. Ognuna si ottiene con un fine-tuning di `alpha_walking` con PPO nella sola situazione, e cammina in quella situazione nettamente meglio della policy ufficiale.
- **Prodotto:**
  1. le specialiste (`.pt` + `.onnx`), promosse in `models/`;
  2. la loro valutazione nella propria situazione e sul piano;
  3. i dataset per le fasi 3-4:
     - rollout di `alpha_walking` in tutte le situazioni (probe, SAE, riconoscitore);
     - rollout di ogni specialista nella propria situazione (distillazione).
- **Criterio per proseguire:** ogni specialista è nettamente migliore di `alpha_walking` nella propria situazione. Una situazione in cui la specialista non migliora esce dalla libreria, perché non c'è un comportamento migliore da distillare.

Il piano stima le settimane 5-7, con le specialiste addestrate in parallelo sui nodi di Faretra e su Moro43.

---

## 0. Prerequisiti da risolvere prima di tutto

### 0.1 Il checkpoint PyTorch di `alpha_walking` (bloccante)

Il fine-tuning con PPO riparte da un **checkpoint rsl_rl** (`model_XXXX.pt`), che contiene:
- l'attore (la policy);
- il **critic** (la rete che stima il valore, usata solo nel training);
- la normalizzazione delle osservazioni;
- lo stato dell'ottimizzatore.

Su Hugging Face c'è solo l'ONNX, cioè solo l'attore. Secondo il manifest della scheda di `rough-walk-e`, il checkpoint di `alpha_walking` è il run W&B `yr25mna4`, `model_9999.pt`, che non è pubblico. `docs/velstand_policy.md` di microduck_rl cita invece il run `441tzs6d`: la discrepanza va chiarita.

- [x] **Checkpoint `.pt` di `alpha_walking` trovato (27 ottobre 2026):** Pollen lo pubblica in `pollen-robotics/microduck-training`, bundle `velocity2/441tzs6d_3750` (run W&B `441tzs6d`, `model_3750.pt`), registrato come `checkpoint` di `alpha_walking` in `configs/policies.yaml`. Il run rilasciato è `441tzs6d`, non `yr25mna4`.
- [x] Verificato: l'attore del checkpoint dà le stesse azioni di `alpha_walking.onnx` (differenza massima 4·10⁻⁶ su 10.000 input casuali); lo controllano `train.py` e `test_checkpoint_actor_equivalence`. Il PPO parte dal checkpoint completo, senza riscaldamento del critic: il piano B qui sotto non serve più.
- [ ] **Piano B, se il checkpoint non arriva:** ricostruire un checkpoint rsl_rl dall'ONNX:
  1. copiare i pesi dell'attore e la normalizzazione degli input nell'`ActorCritic` di rsl_rl (la conversione ONNX → PyTorch di `steerduck/policy.py` già esiste);
  2. inizializzare il critic da zero;
  3. fare un **riscaldamento del critic**, cioè qualche centinaio di iterazioni PPO con l'attore congelato e solo il critic che impara, sul task piano;
  4. solo dopo sbloccare l'attore.

  Senza il riscaldamento, i primi aggiornamenti dell'attore usano vantaggi stimati da un critic casuale e possono rovinare la camminata. Va anche fissata la deviazione standard iniziale dell'azione (rumore di esplorazione), che l'ONNX non contiene: va scelta piccola, per non perturbare subito il passo.
- [ ] **Piano C:** training da zero con microduck_rl. Il piano lo prevede come alternativa (circa 1 h su RTX 5090 sul piano, 5,5 h su terreno irregolare). Da evitare come scelta principale, perché la specialista sarebbe molto diversa da `alpha_walking` e il vettore della fase 3 avrebbe una distanza molto maggiore da colmare (sezione 3.2, passo 1).

### 0.2 Stesso codice di microduck_rl per tutto

- [ ] Usare **`6cd45fc`** (branch `hostile-terrain`) sia per il training sia per la valutazione: è il commit delle fasi 0-1, quello delle schede, e contiene i task `Mjlab-Hostile-*` usati per `rough-walk-g` ed `-e`.
- [ ] **Rifare il run della fase 1 su `6cd45fc`.** Il run `20261020-0137` è stato fatto su `develop` (`273afe0`), come registrato nel suo `run.json`. Già che lo si rifà, usare più semi (`seeds: [0, 1, 2, 3, 4]` in `eval.yaml`) per avere più terreni casuali per situazione (vedi l'analisi dei risultati: le cadute su `rubble` e `grid` dipendono dal singolo terreno sorteggiato). Questo run diventa:
  - il riferimento della fase 2 (`alpha_walking` nelle stesse situazioni e con gli stessi semi delle specialiste);
  - il dataset di `alpha_walking` del punto 7.1.
- [ ] Registrarlo in `results/runs_used.yaml` sotto `fase1`.

### 0.3 Calcolo

- [ ] Misurare il costo di un'iterazione di fine-tuning sulla RTX 3090 e sulla RTX 5090 di Moro43, con il numero di ambienti che si userà (4096). La fase 0 doveva misurarlo (passo 4): riprendere quel numero se c'è. Serve a dimensionare il budget e, nella fase 5, la metrica "costo di una situazione nuova".

---

## 1. Scegliere le situazioni della libreria

Dalla fase 1 (sezione "Candidate per le policy specialiste" di [RESULTS_fase1.md](RESULTS_fase1.md)):

| Situazione | Priorità | Perché | Nota |
|---|---|---|---|
| Discesa | alta | Peggioramento più netto (cadute da 5°); `rough-walk-g` regge fino a 9-10° | L'esempio guida del piano |
| Gradini in discesa | alta | Cadute da 1,25-1,5 cm, potenza ×3-4 già a 1 cm; `rough-walk-g` scende senza cadere fino a 1,75 cm | |
| Terreno irregolare (rubble, griglia) | alta | Cadute da 1,2 cm (griglia) e 1,5 cm (rubble) | Sul rubble ≥ 1,3 cm entrambe le policy restano ferme: la specialista deve migliorare anche la velocità, non solo le cadute |
| Attrito basso | media | Soglia netta: 0,175 → 1/20, 0,15 → 20/20 cadute; `rough-walk-g` regge fino a 0,15 | |
| Carico | da decidere | Sul baricentro nessuna caduta fino al 50%, solo potenza; spostato in avanti al 40-50% cade subito, prima di camminare (equilibrio statico) | Se si tiene, usare il carico in avanti tra 30 e 40% e il carico laterale, dove `rough-walk-g` gira su se stessa |
| Salita (pendenze, gradini) | da decidere | Non cade ma si ferma; nemmeno `rough-walk-g` sale i gradini da 1,5 cm | Probabile limite fisico sui gradini; sulle pendenze la specialista potrebbe almeno avanzare |

- [ ] Decidere la lista finale, **una specialista per famiglia**. Discesa e salita restano separate: sono comportamenti opposti e situazioni diverse per il riconoscitore.
- [ ] Per ogni specialista fissare due intervalli distinti:
  - **intervallo di training:** la difficoltà su cui si addestra, per esempio discesa da 2° a 9°, coprendo la soglia e un po' oltre;
  - **intervallo di valutazione:** le situazioni di `situations.yaml` su cui la si giudica.

  Conviene che il training copra tutta la famiglia e non un solo valore, perché il router applicherà il vettore su un intervallo di pendenze.
- [ ] Aggiungere a `configs/situations.yaml`:
  - il campo `family` con l'etichetta della libreria (`slope_down`, `slope_up`, `steps_down`, `rough`, `slippery`, `payload`). Serve qui per collegare specialista e situazioni, e nella fase 4 come etichetta del riconoscitore. Oggi il codice chiama `family` il modello del nome del blocco: rinominare quel campo interno (per esempio `template`) prima di introdurre questa etichetta, e aggiornare `video_situations: one_per_family`;
  - la suite **`library`**, cioè le situazioni di valutazione delle specialiste scelte.

---

## 2. Codice da scrivere

Seguire [REPO_STRUCTURE.md](../REPO_STRUCTURE.md). Il codice nuovo della fase è circoscritto a questi punti.

### 2.1 `configs/training.yaml`, sezione `ppo`

- [ ] **Parametri comuni a tutte le specialiste:**
  - task mjlab di partenza (il task piano o hostile da cui si eredita la configurazione dell'ambiente);
  - checkpoint iniziale (`alpha_walking`, punto 0.1);
  - numero di ambienti, iterazioni, frequenza di salvataggio dei checkpoint, learning rate, coefficiente di entropia;
  - deviazione standard iniziale dell'azione;
  - semi di training;
  - parametri della regolarizzazione verso `alpha_walking` (punto 4.3).
- [ ] **Un blocco `overrides` per situazione,** solo con ciò che cambia: le situazioni (o l'intervallo) di training e, se servono, più iterazioni o un'altra ricompensa. Aggiungere una specialista significa aggiungere un nome lì, non un file nuovo.
- [ ] I valori numerici iniziali vanno ripresi dai task `Mjlab-Hostile-Finetune*` di microduck_rl e da `MicroduckHostileRlCfg`, che sono la ricetta già provata del fine-tuning di `alpha_walking` (`rough-walk-e`: 800 iterazioni).

### 2.2 `steerduck/mjlab_tasks.py`

Traduce **le stesse voci** di `situations.yaml` in un task mjlab per il training su GPU.

- [ ] **Terreno.** Un `TerrainGeneratorCfg` con i sub-terrain della famiglia, come "scala" di difficoltà: righe = difficoltà dentro l'intervallo di training, con la curriculum di `hostile_terrain_levels` di microduck_rl (promozione se l'episodio finisce in piedi seguendo il comando, retrocessione se cade). Accorgimenti:
  - **riusare le classi** `FlatFacePyramidTerrainCfg` e `ProfileStripTerrainCfg` di `steerduck/scene.py`: sono già `SubTerrainCfg` di mjlab, quindi lo stesso codice genera il terreno in valutazione e in training;
  - **discesa:** piramide a facce piane con partenza sulla piattaforma in cima. In ogni direzione si scende, ma lungo le diagonali ci sono le pieghe: verificare che non creino un caso troppo diverso dalla valutazione, che parte al centro di una faccia;
  - **gradini in discesa:** piramide a gradini con partenza in cima;
  - **irregolare:** i terreni `grid` e `rubble` degli stessi tipi della valutazione;
  - tenere una piccola quota di terreno piano (per esempio 10-15%), come `hostile_subterrains`, perché la specialista non disimpari a camminare sul piano;
  - mantenere `_soften_terrain_contacts` e la regola dei terreni "a fossa" (sezione 4.6 del piano).
- [ ] **Attrito.** Evento all'avvio che imposta lo stesso coefficiente sui piedi **e** sui geom del terreno, campionato nell'intervallo di training (per esempio 0,1-0,3). Lo stesso vincolo della valutazione: in MuJoCo vince il massimo dei due coefficienti.
- [ ] **Carico** (se si tiene): massa aggiunta al tronco con spostamento del baricentro, con `dr.body_mass` e lo spostamento del centro di massa già usato da microduck_rl per la randomizzazione.
- [ ] **Registrazione dei task** `Mjlab-Spec-<famiglia>-MicroDuck` con `register_mjlab_task`, dal nostro pacchetto, **senza modificare microduck_rl**.
- [ ] **Anti-rotazione.** Mantenere la condizione di promozione con spostamento reale (`min_progress`) e la ricompensa sulla rotazione più severa dei task `FinetuneFeetProgress`. Il run A di hostile ha imparato a girare in cerchio sulla piattaforma di partenza: la stessa trappola vale per noi, soprattutto in salita e sul rubble, dove la fase 1 mostra il robot fermo.

### 2.3 `scripts/train.py ppo --situation <famiglia>`

- [ ] Legge `training.yaml`, registra il task da `mjlab_tasks.py`, lancia il trainer di microduck_rl (`uv run train ...` con `--agent.resume True --agent.load-checkpoint ...`) e salva tutto in un run:
  - `runs/<data>_train_ppo_<famiglia>/` con `config.yaml`, `run.json` (ore di GPU, commit di microduck_rl) e `log.txt`;
  - i checkpoint `.pt`;
  - le curve (`curves.parquet`, oltre a W&B).
- [ ] Alla fine esporta il checkpoint scelto in ONNX con `scripts/export.py` di microduck_rl e lancia la valutazione del punto 5 sulla propria situazione e sul piano. REPO_STRUCTURE.md lo prevede: ogni training termina valutando il risultato.

### 2.4 `steerduck/policy.py` e `configs/policies.yaml`

- [ ] Una voce `specialist_<famiglia>` per ogni specialista, di tipo `onnx`, che punta a `models/specialist_<famiglia>/`. La valutazione funziona già con le policy ONNX.
- [ ] Il caricamento da checkpoint rsl_rl (previsto da REPO_STRUCTURE.md "dalla fase 2"), solo se serve: per la distillazione della fase 3 basta la conversione ONNX → PyTorch esistente.

### 2.5 `tests/test_sanity.py::test_conditions`

- [ ] Dalla stessa voce di `situations.yaml`, il terreno generato per la valutazione (CPU) e per il training (mjlab) hanno la stessa geometria (altezze a campione), lo stesso attrito sui piedi e sul terreno e la stessa massa del carico. È l'unico punto in cui una situazione viene tradotta due volte (REPO_STRUCTURE.md, sezione 9).

### 2.6 `steerduck/report.py` e `scripts/report.py --phase 2`

- [ ] **Tabella per specialista:** nella propria situazione e sul piano, cadute (con intervallo di Wilson), errore di velocità, velocità, potenza, distanza percorsa e velocità di rotazione, a confronto con `alpha_walking` sugli stessi episodi (stessi semi).
- [ ] **Decisione per specialista** secondo il criterio del punto 5.2, e il riepilogo "entra / non entra nella libreria".
- [ ] **Distanza dalla policy ufficiale** (punto 5.3).

### 2.7 Documentazione

- [ ] Aggiornare REPO_STRUCTURE.md (stato, `training.yaml`, `mjlab_tasks.py`, `train.py`, test), il README (comandi della fase 2) e `results/runs_used.yaml` (`fase2`).

---

## 3. Verifiche prima di addestrare

- [ ] **Guardare i terreni di training** con `visualize_terrain` di mjlab e con il viewer: pendenze, gradini e spigoli devono corrispondere alla valutazione, e il robot non deve partire sotto il pavimento.
- [ ] **`alpha_walking` nel task di training, senza addestrarla** (`play` di mjlab). Deve peggiorare come nella fase 1, per esempio cadere in discesa da circa 5°. Se in mjlab non cade, il simulatore di training non riproduce il problema e la specialista non avrà niente da imparare: va capito prima di spendere ore di GPU. È anche la prima misura del rischio "due simulatori" (REPO_STRUCTURE.md, sezione 9).
- [ ] **Fine-tuning di prova breve** (per esempio 50 iterazioni) su una sola famiglia:
  - la ricompensa di partenza è quella di una policy che cammina già, non quella di una policy casuale;
  - la lunghezza degli episodi non crolla nelle prime iterazioni;
  - nessun termine di penalità è positivo.

  Con il piano B del punto 0.1, verificare in più che il critic si sia stabilizzato prima di sbloccare l'attore.
- [ ] **Misurare il tempo per iterazione** (punto 0.3) e fissare il budget.

---

## 4. Training delle specialiste

### 4.1 Ordine

- [ ] **Prima la discesa,** l'esempio del piano, fino in fondo (training, valutazione, dati). Serve a mettere a punto la procedura.
- [ ] **Poi le altre in parallelo,** una per nodo (Faretra, Moro43).

### 4.2 Protocollo per ogni specialista

- [ ] **Fine-tuning da `alpha_walking`** con i parametri di `training.yaml`. Riferimento: `rough-walk-e` è `alpha_walking` più 800 iterazioni su terreni misti. Per una singola famiglia ne potrebbero bastare meno: partire da circa 500-1000 e guardare le curve.
- [ ] **Salvare un checkpoint ogni 100 iterazioni** (o meno) e non scegliere automaticamente l'ultimo: la scelta si fa con la valutazione del punto 5. AGENTS.md di microduck_rl insiste che un training "fallito" va prima misurato sui checkpoint reali.
- [ ] **Da guardare durante il training:**
  - la ricompensa di inseguimento della velocità che cresce;
  - la lunghezza degli episodi;
  - il livello medio della curriculum di terreno, che deve salire;
  - le penalità ≤ 0;
  - la distanza percorsa (contro la rotazione in cerchio).
- [ ] **Fissare le curricula a passi** (rampa della penalità sul tasso d'azione, intervalli della testa, randomizzazione del baricentro) allo stadio finale fin dall'inizio, come fa `_freeze_step_curricula`. Il passo esiste già e non si vuole che torni più brusco.
- [ ] **Annotare nel run** le ore di GPU, per la metrica "costo di una situazione nuova" della fase 5.

### 4.3 Tenere la specialista vicina ad `alpha_walking`

La specialista serve come modello da imitare per un vettore che si somma ad `alpha_walking` (sezione 3.2, passo 1). Più si allontana, più è difficile che un vettore la riproduca. Decisioni:

- [ ] **Prima prova con la ricompensa invariata** (come `FinetuneBase`) e solo il terreno della situazione. Cambiare la ricompensa, per esempio l'altezza dei piedi a 3,5 cm di `FinetuneFeet`, cambia il passo anche dove non serve. Usarla solo se senza non migliora.
- [ ] **Valutare un termine di regolarizzazione verso `alpha_walking`:** penalità sulla distanza tra le azioni della specialista e quelle di `alpha_walking` nello stesso stato, con un peso piccolo. Non c'è nei task di microduck_rl, andrebbe aggiunto in `mjlab_tasks.py`. Da introdurre solo se la distanza misurata al punto 5.3 risulta grande.
- [ ] **Meno iterazioni a parità di risultato:** tra due checkpoint ugualmente buoni nella situazione, scegliere il più vicino ad `alpha_walking`.

### 4.4 Se il fine-tuning non basta

Nell'ordine:
1. più iterazioni;
2. un intervallo di training più facile all'inizio (curriculum);
3. la ricompensa di `FinetuneFeet` / `FinetuneFeetProgress`;
4. training da zero (piano C).

Se nemmeno così la specialista migliora, la situazione esce dalla libreria (criterio della fase). Sui gradini in salita è un esito probabile, per il limite fisico dell'altezza del passo.

---

## 5. Valutazione delle specialiste

### 5.1 Protocollo

- [ ] **Stesso simulatore e stesso protocollo della fase 1:** CPU MuJoCo + bam, `configs/eval.yaml`, stessi semi, quindi episodi accoppiati con `alpha_walking`.
- [ ] **Per ogni specialista:**
  - la suite della propria famiglia;
  - il piano, per verificare che non sia peggiorata dove non serve;
  - le **altre famiglie**: non è richiesto dal criterio, ma serve a capire se è una specialista o una policy più robusta in generale. Costa poco ed è utile per la fase 5.
- [ ] **Valutare 3-5 checkpoint** per specialista e scegliere quello da promuovere con il criterio del punto 5.2.
- [ ] **Più episodi nelle situazioni di soglia** (50-100, come suggerito in RESULTS_fase1.md): sono quelle su cui la fase 3 misurerà il vantaggio recuperato, e 20 episodi danno intervalli troppo larghi.

### 5.2 Criterio "nettamente migliore"

Il piano non dà una soglia numerica: va fissata prima di vedere i risultati, e scritta in `eval.yaml` come per la fase 1. Proposta:

- [ ] **Nella propria famiglia, su almeno metà delle situazioni di soglia** (quelle in cui `alpha_walking` ha cadute fra il 20% e il 100%):
  - cadute ridotte con intervalli di Wilson al 95% che non si sovrappongono;
  - oppure, dove `alpha_walking` non cade (salita, carico, rubble fermo), velocità o potenza migliori con test di Welch p < 0,01 e una soglia minima, come per "peggiore del piano".
- [ ] **Sul piano** nessun peggioramento significativo: cadute 0, potenza ed errore di velocità entro una tolleranza (per esempio +10%).
- [ ] **Vantaggio disponibile:** riportare per ogni situazione la differenza di cadute fra `alpha_walking` e specialista. È il denominatore del "vantaggio recuperato" della fase 3 (sezione 3.2, passo 5): se è piccolo, anche un buon vettore mostrerà poco.

### 5.3 Distanza dalla policy ufficiale

- [ ] Sugli stati visitati da `alpha_walking` (dati del punto 7.1) e da ciascuna specialista nella propria situazione, misurare l'errore quadratico medio fra le azioni delle due policy:
  - **nella situazione:** è quanto il vettore dovrà spostare;
  - **sul piano:** dovrebbe essere piccola.

  Serve a scegliere fra checkpoint (punto 4.3) e a prevedere quanto sarà difficile la distillazione.

### 5.4 Controlli qualitativi

- [ ] **Video** del primo episodio della specialista e di `alpha_walking` in alcune situazioni della famiglia (`video_situations`). Vanno guardati: AGENTS.md avverte che le metriche possono passare con un comportamento visibilmente sbagliato.
- [ ] **Rotazione e distanza percorsa:** una specialista che "non cade" girando su se stessa o restando ferma non va promossa (vedi `rough_walk_g` con il carico laterale e sul rubble, nella fase 1).

---

## 6. Promozione

- [ ] Per ogni specialista che supera il criterio:
  1. copiare `.pt` e `.onnx` del checkpoint scelto in `models/specialist_<famiglia>/`, con un `.json` di metadati (run di origine, iterazione, situazioni di training, risultato della valutazione);
  2. aggiungere la voce in `configs/policies.yaml`.
- [ ] Registrare in `results/runs_used.yaml` (`fase2`) i run di training e di valutazione usati.
- [ ] Per le situazioni escluse, annotare il motivo e i numeri, che entrano nella tesi.

---

## 7. Raccolta dei dataset per le fasi 3-4

Sezioni 4.8, 4.9 e 4.11 del piano. Il formato è già quello della fase 1: `data/<run_id>/<controller>/<situazione>__seed<seme>.npz` più `fields.json`.

### 7.1 `alpha_walking` in tutte le situazioni (probe, SAE, riconoscitore)

- [ ] È il run del punto 0.2 (commit `6cd45fc`, più semi), con `record_steps` e `record_activations` attivi.
- [ ] **Dimensione:** il piano indica circa un milione di passi come punto di partenza per il SAE sul layer da 512. Con 59 situazioni × 20 episodi × 5 semi × circa 1000 passi si arriva a qualche milione: verificare lo spazio (circa 2 kB per passo con le activation) e, se serve, ridurre gli episodi per seme.
- [ ] **Transizioni:** riattivarle in `situations.yaml` (oggi sono commentate). Il riconoscitore della fase 4 ha bisogno proprio dei cambi di situazione durante l'episodio. Correggere prima l'anomalia di `trans_up_5deg` segnalata in RESULTS_fase1.md.

### 7.2 Ogni specialista nella propria situazione (distillazione)

- [ ] `evaluate.py specialist_<famiglia> --suite <famiglia>` con la registrazione attiva: salva per ogni passo l'osservazione (61) e l'azione della specialista (14). Sono le coppie del passo 2 della sezione 3.2.
- [ ] **Quantità:** il piano suggerisce circa 100.000 coppie per situazione (da verificare); il vettore da imparare ha al massimo 512 numeri. Con circa 1000 passi per episodio servono un centinaio di episodi per famiglia, distribuiti su tutti i valori della famiglia.
- [ ] Le activation della specialista non servono alla distillazione, perché il vettore si somma alle activation di `alpha_walking`. Si possono spegnere con `--no-activations` per risparmiare spazio, ma potrebbero servire a confrontare le rappresentazioni: decidere.
- [ ] Il ciclo DAgger (passo 4 della sezione 3.2) appartiene alla fase 3. In questa fase serve solo il dataset iniziale.

---

## 8. Prodotti finali della fase

- [ ] Specialiste promosse in `models/` e voci in `policies.yaml`.
- [ ] Tabella della fase 2 (`results/tables/fase2_*.csv`) generata da `report.py --phase 2`.
- [ ] Dataset di `alpha_walking` e delle specialiste in `data/`, con i run elencati in `runs_used.yaml`.
- [ ] `docs/RESULTS_fase2.md`, sul modello di RESULTS_fase1.md:
  - metodo (fine-tuning, intervalli di training, iterazioni, ore di GPU);
  - risultati per specialista nella propria situazione e sul piano;
  - distanza da `alpha_walking`;
  - situazioni escluse e motivi;
  - limiti (due simulatori, checkpoint di partenza).

---

## 9. Rischi e domande aperte

| Rischio | Effetto | Cosa fare |
|---|---|---|
| Checkpoint di `alpha_walking` non disponibile | Il fine-tuning parte da un critic casuale | Piano B del punto 0.1 (riscaldamento del critic), oppure piano C |
| Simulatore di training (mjlab GPU) diverso da quello di valutazione (CPU + bam) | Una specialista buona in mjlab potrebbe non esserlo sullo stack CPU, o viceversa | Verifica del punto 3 (`alpha_walking` in mjlab), `test_conditions` e scelta del checkpoint fatta sullo stack CPU |
| La specialista impara a girare in cerchio o a restare ferma | Ricompensa alta, comportamento inutile | Promozione con spostamento reale, ricompensa sulla rotazione più severa, controllo di distanza e rotazione nella valutazione |
| Specialista troppo diversa da `alpha_walking` | Distillazione difficile nella fase 3 | Ricompensa invariata, poche iterazioni, eventuale regolarizzazione, misura della distanza (punto 5.3) |
| Limite fisico (gradini in salita, pendenze ≥ 12,5°) | Nessuna specialista migliora | Escludere la situazione, come prevede il criterio |
| Velocità reale circa 1/3 del comando anche sul piano | L'errore di velocità di riferimento è già 0,2 m/s | È coerente con le schede (0,11-0,12 m/s): confrontare le specialiste soprattutto su cadute, velocità, distanza e potenza, non sull'errore assoluto |

**Decisioni da prendere prima di iniziare:**
1. Quali famiglie entrano nella libreria (punto 1).
2. Da dove parte il fine-tuning: checkpoint di Pollen, piano B o piano C (punto 0.1).
3. La soglia numerica di "nettamente migliore" (punto 5.2).
4. Se usare la regolarizzazione verso `alpha_walking` fin dall'inizio o solo se serve (punto 4.3).
