# TO DO — dalle specialiste di pendenza alla libreria completa

*Matteo Preda — 28 ottobre 2026. Riprende i punti ancora aperti di [TO_DO_fase2.md](TO_DO_fase2.md) e quelli emersi lavorando sulle specialiste di pendenza. Riferimenti: [piano di lavoro](Piano_lavoro_tesi_Preda.md) (sezione 3.2 e sezione 5, fasi 2-4), [struttura della repository](../REPO_STRUCTURE.md), [Notes.txt](../Notes.txt).*

## Stato al 28 ottobre 2026

- **Specialiste:** `slope_down-20261027-2240` e `slope_up-20261028-0021`. Sono `alpha_walking` con un fine-tuning PPO di 1000 iterazioni, a partire dal checkpoint di Pollen, sulle famiglie direzionali di `configs/training.yaml`.
- **Valutazione:** run `20261028-1648_evaluate_slope_up-20261028-0021+slope_down-20261027-2240_all_only`, con 41 situazioni (piano, piramidi, fosse, rampe `face` in discesa e in salita) e i dati per passo.
  - Il criterio della fase 2 è soddisfatto per entrambe.
  - In discesa `face` a 6-8°, alpha ha lo 0% di successi e slope_down il 100%.
  - In salita a 7-10°, alpha resta ferma e slope_up ha il 100% di successi.
- **Dataset di alpha_walking:** run `20261028-1239_evaluate_alpha_walking_all`, con 87 situazioni (comprese le transizioni), un seme e le activation.
- **Distanza da alpha:** `scripts/action_distance.py`. I risultati e le considerazioni sono in Notes.txt (commit `ec799b0`). slope_up è più lontana di slope_down.

L'ordine proposto è A → B, con C in parallelo dove la GPU è libera, poi D ed E. Prima di aggiungere famiglie conviene che la pipeline sia provata da cima a fondo su una famiglia sola (punto B).

---

## A. Chiudere la fase 2 sulle pendenze

- [ ] **Registrare i run** in `results/runs_used.yaml`, oggi vuoto:
  - sotto `fase1`, la valutazione di alpha e di rough_walk_g (`20261028-1239…`, `20261028-1320…`);
  - sotto `fase2`, i due training e la valutazione `20261028-1648…`.
- [ ] **Scrivere il criterio "nettamente migliore"** in `configs/eval.yaml` (TO_DO_fase2, punto 5.2). La soglia è stata fissata dopo aver visto i risultati: va dichiarato nella tesi. Con queste differenze l'esito non cambia.
- [ ] **Generare la tabella della fase 2:** `report.py --run 20261028-1648… 20261028-1239…`. Gli episodi sono accoppiati, perché i semi dipendono dal nome della situazione.
- [ ] **Scrivere `docs/RESULTS_fase2.md`** sul modello di RESULTS_fase1.md. Deve contenere:
  - il metodo: checkpoint di partenza, famiglie direzionali, comando, iterazioni, ore di GPU;
  - i risultati per specialista nella propria situazione e sul piano;
  - la distanza da alpha;
  - i limiti.

  La base sono le considerazioni di Notes.txt.
- [ ] **Annotare cosa emerge dalle curve di training:**
  - la std delle azioni collassa da 0.19 a 0.03, perché `entropy_coef: 0`;
  - per slope_down il curriculum è saturo (90% di promozioni, livello massimo raggiunto all'iterazione 200);
  - l'altezza dei piedi in volo cala del 15-20%.
- [ ] **Guardare i video** delle specialiste. Su slope_up controllare la caviglia sinistra: sul piano è il motore più distante da alpha, con un errore quadratico medio di 0.044 contro circa 0.01 della destra.
- [ ] **Facoltativo: scelta del checkpoint** (TO_DO_fase2, punti 4.3 e 5.1). L'ipotesi di lavoro è che il checkpoint 1000 sia equivalente al 400. Il reward è costante, ma la rete continua a cambiare: la std cala e il KL resta intorno a 0.01 a ogni aggiornamento. Per verificarlo:
  - [ ] aggiungere a `train.py` l'export in ONNX di un checkpoint scelto, perché oggi esporta solo l'ultimo;
  - [ ] valutare `model_400.pt` di slope_up, la più lontana da alpha, sulle stesse 41 situazioni;
  - [ ] confrontare successi e `action_distance.py` con il checkpoint 1000. Se sono simili, scriverlo come risultato e non più come ipotesi.
- [ ] **Facoltativo:** fare 50-100 episodi nelle situazioni di soglia (5-7° in discesa, 6-8° in salita), perché 20 episodi danno intervalli di Wilson larghi. Sono le situazioni su cui la fase 3 misurerà il vantaggio recuperato.

---

## B. Fase 3, pilota: vettore costante per slope_down

Serve a provare tutta la pipeline (training → valutazione → dataset → distillazione → valutazione con il vettore) prima di aggiungere famiglie. È anche il rischio principale della tesi. Il dataset c'è già: i dati per passo di slope_down nel run `1648`.

- [ ] **`steerduck/steering.py`:**
  - il forward hook che somma α·v all'uscita di `h1`, `h2` o `h3`;
  - la forma `const`, con l'API comune delle forme di adapter (REPO_STRUCTURE.md, sezione 4).
- [ ] **`steerduck/distill.py`:**
  - [ ] il training di v sulle coppie (osservazione, azione della specialista), con i pesi di alpha bloccati (piano, sezione 3.2, passo 3). Usare solo i passi fuori dai 2 s prima di una caduta (`falls_within_2s`);
  - [ ] il ciclo DAgger (passo 4): alpha + v cammina con `rollout.py`, la specialista etichetta gli stati con `policy.relabel`, e si ripete finché cadute ed errore di velocità smettono di migliorare.
- [ ] **Sezione `distill` in `configs/training.yaml`:** layer, learning rate, epoche, numero di cicli DAgger, episodi per ciclo, situazioni.
- [ ] **`train.py distill --form const --layer h1|h2|h3|all --situation slope_down`**, che salva il vettore in un run.
- [ ] **Il Controller `alpha+const_<layer>_slope_down`** in `build_controller` e in `policies.yaml`, così si valuta con `evaluate.py` come ogni altra policy.
- [ ] **`test_zero_adapter`:** con α = 0 le azioni devono essere esattamente quelle di alpha.
- [ ] **In `report.py`:**
  - il **vantaggio recuperato** (sezione 3.2, passo 5): (cadute di alpha − cadute con il vettore) / (cadute di alpha − cadute della specialista), e lo stesso calcolo sui successi;
  - la **quota imitata**: 1 − distanza(alpha + v, specialista) / distanza(alpha, specialista), riusando `action_distance`.
- [ ] **Scegliere il layer** (`h1`, `h2`, `h3` o tutti) con la migliore imitazione e il miglior vantaggio recuperato.
- [ ] **Verificare che il vettore non peggiori il piano** quando è applicato per errore.
- [ ] **`evaluate.py --alpha 0,…,1.5`:** prova di steering e limite di sicurezza (sezione 3.2, passo 6; sezione 3.5).
- [ ] **Ripetere per slope_up.** È la più lontana da alpha: se il vettore costante non basta, provare LoRA di rango 1 e la piccola rete (sezione 3.3).

**Criterio della fase 3:** almeno due situazioni in cui una forma recupera circa metà del vantaggio della specialista. slope_down e slope_up possono bastare.

---

## C. Training delle altre situazioni

Per una nuova famiglia di **pendenze** basta un blocco `families` in `configs/training.yaml`, senza codice. Per le altre il training su mjlab oggi non funziona e va completato `steerduck/mjlab_tasks.py`.

### C.1 Codice comune a tutte le famiglie

- [ ] **Famiglie senza direzione** (piano, `grid`, `rubble`, `waves`, attrito, carico). `family_heading` accetta solo `direction: downhill | uphill` e oggi si ferma con un `ValueError`. Serve la variante della valutazione:
  - partenza al centro del patch;
  - heading di partenza casuale;
  - heading tenuto fisso per tutto l'episodio.

  Per le famiglie senza direzione `set_straight_walking` deve quindi sorteggiare l'heading per ogni ambiente invece di usarne uno solo.
- [ ] **Partenze specifiche** come `face_start`, coerenti con `scene.start_pose`:
  - **gradini in salita:** partenza vicino al bordo, rivolti verso la cima (in valutazione a 0.35 m dal bordo);
  - **gradini in discesa:** partenza sulla piattaforma in cima.
- [ ] **Controllare che `LadderTerrainCfg` funzioni con i terreni a box** (`grid`, `stairs` di mjlab, non heightfield): dimensione del patch, origine, `border_width`.
- [ ] **`test_conditions`** (TO_DO_fase2, punto 2.5): dalla stessa voce di `situations.yaml`, il simulatore CPU e mjlab devono dare la stessa geometria a campione, lo stesso attrito su piedi e terreno e la stessa massa del carico. Diventa indispensabile con attrito e carico.

### C.2 Attrito basso

- [ ] **Evento di reset** in `mjlab_tasks.py` che imposta lo stesso coefficiente sui piedi **e** sui geom del terreno. In MuJoCo vince il massimo dei due, come in `scene.set_friction`.
- [ ] **Intervallo `[facile, difficile]` lungo la scala**, per esempio da 0.3 a 0.12. La valutazione ha 0.175, 0.15 e 0.125, e alpha cade del tutto a 0.15.
- [ ] **Famiglia `slippery`** in `training.yaml`.

### C.3 Carico

- [ ] **Decidere se tenerlo** (TO_DO_fase2, punto 1). Sul baricentro alpha non cade fino al 50%; con il carico spostato in avanti al 40-50% cade subito. Se si tiene, usare il carico in avanti o laterale tra il 30% e il 40%.
- [ ] **Evento di reset** che aggiunge al tronco la massa e lo spostamento del baricentro come `scene.build_model` (`mass_pct`, `offset_m`), con `dr.body_mass` e lo spostamento del baricentro già usati da microduck_rl.
- [ ] **Famiglia `payload`** in `training.yaml`.

### C.4 Terreno irregolare (`grid`, `rubble`, `waves`) e gradini

- [ ] **Famiglie `rough`** (grid + rubble, eventualmente waves) **e `steps_down`** in `training.yaml`. Gli intervalli di training devono coprire la soglia e un po' oltre:
  - griglia: cadute da 1.2 cm;
  - rubble: cadute da 1.5 cm;
  - gradini in discesa: cadute da 1.25-1.5 cm.
- [ ] **Attenzione ai limiti fisici.** Sul rubble ≥ 1.3 cm nessuna policy avanza: i piedi si alzano di circa 12 mm. La specialista potrebbe al massimo evitare le cadute, e il criterio la può escludere. Lo stesso vale per i gradini in salita.
- [ ] **Anti-rotazione** (TO_DO_fase2, punto 2.2): controllare che la specialista non impari a girare in cerchio o a restare ferma. In valutazione lo mostrano `off_course` e `in_place`.

### C.5 Per ogni nuova famiglia, prima di addestrare

- [ ] **Guardare il terreno** con `evaluate.py --preview` (valutazione) e il viewer di mjlab (training).
- [ ] **Far camminare alpha nel task di training, senza addestrarla:** deve peggiorare come nella fase 1. Se in mjlab non peggiora, la specialista non ha niente da imparare.
- [ ] **Training di prova breve:** `--num-envs 32 --finetune-iterations 2`, poi circa 50 iterazioni su GPU. Controllare:
  - il reward iniziale è quello di una policy che cammina;
  - la durata degli episodi non crolla;
  - le penalità restano ≤ 0.
- [ ] **Training completo:** `train.py ppo --situation <famiglia>` (oppure `--situation all`).
- [ ] **Valutazione** con gli stessi semi di alpha, sulla propria famiglia e sul piano, poi `action_distance.py`.
- [ ] **Registrare il run** in `runs_used.yaml`, oppure escludere la famiglia scrivendo il motivo (criterio della fase 2).

---

## D. Libreria e situazioni

- [ ] **Decidere la lista finale delle famiglie** della libreria, una specialista per famiglia. Discesa e salita restano separate. Candidate, dalla fase 1:
  - alta priorità: gradini in discesa, terreno irregolare;
  - media priorità: attrito basso;
  - da decidere: carico.
- [ ] **In `configs/situations.yaml`:**
  - rinominare il campo interno `family`, che oggi è il modello del nome del blocco, per esempio in `template`, e aggiornare `video_situations: one_per_family`;
  - introdurre `family` come etichetta della libreria (`slope_down`, `slope_up`, `steps_down`, `rough`, `slippery`, `payload`). Collega specialista e situazioni, ed è l'etichetta del riconoscitore della fase 4;
  - aggiungere la suite `library`.
- [ ] **Suite per famiglia** (per esempio `slope_down`, `slope_up`), così valutazione e raccolta dati di una specialista si lanciano con `--suite` invece di un lungo `--only`.

---

## E. Dataset per SAE e riconoscitore (fasi 3-4)

- [ ] **Quando la libreria è decisa,** ripetere la valutazione di alpha in tutte le situazioni con più semi (`seeds: [0, 1, 2, 3, 4]`, TO_DO_fase2 punto 0.2). Le cadute su `rubble` e `grid` dipendono dal singolo terreno sorteggiato. Con un solo seme ci sono circa 1.7 milioni di passi: bastano per un primo SAE, non per il riconoscitore.
- [ ] **Verificare le transizioni** prima di usarle per il riconoscitore. Va corretta l'anomalia di `trans_up_5deg` segnalata in RESULTS_fase1.md.
- [ ] **Spazio:** circa 2 GB per policy con le activation. Con 5 semi conviene ridurre gli episodi per seme, oppure registrare le activation solo per alpha.

---

## Note

- **Video registrati prima della correzione del clipping** (`video_znear_m`): tagliavano il robot sulla piramide `face`. Per riaverli corretti va rifatta la valutazione.
- **Il run `20261028-1112…` non è più in `runs/`:** restano solo i suoi dati in `data/`. Il riferimento della fase 2 è ora il run `1648`.
- **Le soglie di successo** (`min_progress_m_s`) `report.py` le legge dall'`eval.yaml` corrente, non dal run: cambiarle cambia le tabelle anche dei run vecchi.
