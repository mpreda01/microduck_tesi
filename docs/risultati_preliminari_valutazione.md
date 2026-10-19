# Risultati preliminari della valutazione in simulazione

*Fase 0-1 del piano di lavoro: suite di valutazione di `alpha_walking` e `rough-walk-g`.*

## Impostazione

- **Ambiente:** camminata del Microduck in microduck_rl (mjlab 1.3.0, attuatori BAM). Le randomizzazioni del training restano attive: centro di massa, massa ±5%, attrito dei giunti, bias degli encoder, disallineamento dell'IMU. Spinte disattivate.
- **Condizioni:** ciascuna fissa per tutto l'episodio. Piano; pendenze costanti in salita e in discesa (sulla piramide, vedi sezione 3); attrito di pavimento e piedi ridotto; carichi rigidi sul tronco con centro di massa spostato; 14 tipi di terreno generato.
- **Episodi:** 100 per condizione, da 20 s ciascuno, con seed 0. Le due policy vengono valutate sulle stesse condizioni.
- **Metriche per episodio:** caduta, cioè inclinazione del tronco oltre la soglia di mjlab, con intervallo di confidenza di Wilson al 95%; velocità; potenza meccanica dei motori; inclinazione massima.
- **Dati salvati:** per la policy da studiare, anche i dati per passo (osservazioni, activation, ground truth) per SAE e probe.

`A` = `alpha_walking`, `B` = `rough-walk-g`.

---

## 1. Le policy seguono solo in parte il comando di velocità

### Il problema

L'errore di velocità |v − comando| risultava di circa 0,2 m/s in quasi tutte le condizioni, per entrambe le policy, anche sul piano. Abbiamo verificato le cause possibili una alla volta.

| Ipotesi | Verifica | Esito |
|---|---|---|
| La metrica è istantanea e include l'oscillazione del tronco dentro ogni passo | Scomposizione della velocità: oscillazione laterale con std di vy pari a 0,15 m/s | **In parte vera.** Abbiamo aggiunto un errore calcolato su velocità e comando mediati su circa 1 s |
| Il comando non è quello impostato | Valori del comando registrati: 0,2 e 0,3 invece di solo 0,2 | **Vera, era un nostro errore, corretto.** Il meccanismo `rel_forward_envs` di mjlab porta vx ad almeno 0,3 m/s in una parte delle copie. Ora viene disattivato nelle condizioni a comando fisso |
| Il robot resta fermo nei primi secondi | Velocità media per intervalli di 2 s | **Esclusa.** La velocità si stabilizza entro circa 4 s e poi resta costante, sotto il comando |
| I contatti morbidi dei terreni generati rallentano il robot | Stesso test sul piano esatto del training (`plane`: contatti e solver predefiniti) | **Esclusa.** `plane` e piano generato danno valori quasi uguali |

### Il risultato

Velocità reale in avanti (m/s), con comando costante in avanti, wz = 0, 100 episodi per valore:

| Comando | A, piano generato | A, `plane` | B, piano generato | B, `plane` |
|---|---|---|---|---|
| 0,1 | **−0,001** | – | – | – |
| 0,2 | 0,079 | 0,059 | – | 0,085 |
| 0,3 | 0,153 | 0,146 | 0,153 | 0,141 |
| 0,4 | 0,205 | 0,189 | 0,209 | 0,190 |

- **Zona morta:** con 0,1 m/s `alpha_walking` resta ferma in piedi (potenza 0,009 W) e non fa nemmeno passi sul posto.
- **Guadagno di circa 0,5:** tra 0,3 e 0,4 m/s entrambe le policy vanno a circa metà della velocità comandata; a 0,2 m/s meno della metà.
- **Il comportamento è comune alle due policy.** È coerente con la scheda di `rough-walk-g`, che riporta un problema analogo sulla rotazione: *"weak turn response (0.29 rad/s achieved for 1.0 commanded)"*. Anche il repository segnala una zona morta di `alpha_walking` sui comandi di rotazione.

### Conseguenze

- È una proprietà delle policy in simulazione, non un errore di misura: va riportata come tale.
- L'errore assoluto rispetto al comando è dominato da questo guadagno e distingue poco le condizioni. Per misurare l'effetto di una condizione conviene la velocità reale relativa al piano con lo stesso comando, per esempio salita a 10° contro piano.
- Le condizioni a comando fisso (pendenze) vanno eseguite a 0,4 m/s: a 0,2 m/s il robot quasi non avanza.

---

## 2. `rough-walk-g` cammina in cerchio

### L'osservazione

Con un comando costante in avanti e rotazione nulla, nel video `rough-walk-g` cambia direzione e torna verso il punto di partenza. Nei dati, la distanza percorsa in linea retta è molto minore della strada percorsa in avanti:

| Comando 0,3 m/s | Avanti (m/s) | In linea retta (m/s) | Rapporto |
|---|---|---|---|
| A, piano generato | 0,153 | 0,130 | 0,85 |
| B, piano generato | 0,153 | **0,051** | **0,33** |
| B, `plane` | 0,141 | 0,118 | 0,84 |

### La misura

Velocità di rotazione media per episodio, piano generato, 0,3 m/s. Un episodio è contato a sinistra o a destra se la sua rotazione media supera 0,02 rad/s.

| | Rotazione media in valore assoluto | Episodi a sinistra | Episodi a destra |
|---|---|---|---|
| A, `alpha_walking` | 0,066 rad/s | 32 | 31 |
| B, `rough-walk-g` | **0,32 rad/s**, cioè mezzo giro in circa 10 s | **92** | **4** |

### Interpretazione

- **Perché la deriva si accumula.** Il comando contiene una velocità di rotazione, non una direzione da mantenere. Se il robot ruota, la policy non ha un riferimento per tornare alla direzione iniziale, e con un controllo della rotazione debole la deriva si somma.
- **L'ambiente di valutazione è simmetrico.** Nelle stesse condizioni `alpha_walking` ha una deriva piccola e distribuita in modo uguale tra sinistra e destra (32 contro 31). È quanto ci si aspetta dalle randomizzazioni, che sono centrate sullo zero. L'ambiente quindi non introduce un'asimmetria.
- **La rotazione di `rough-walk-g` è una sua proprietà.** 92 episodi contro 4, con una velocità di rotazione circa 5 volte quella di `alpha_walking`, indicano un'asimmetria dell'andatura di `rough-walk-g` a vx = 0,3 e wz = 0, che la policy non corregge.
- **È più forte sul piano generato che su `plane`** (rapporto 0,33 contro 0,84), anche se `rough-walk-g` è stata addestrata su terreni generati.
- **Un dettaglio da chiarire:** 0,32 rad/s è circa la rotazione massima che la scheda attribuisce alla policy con il comando massimo (0,29 rad/s per 1,0 comandato).

### Conseguenze

- Sul piano le metriche restano valide: cadute, velocità e potenza sono nel riferimento del robot.
- **Sulle pendenze la deriva invalida la condizione, per entrambe le policy.** Con `rough-walk-g`, a 0,3 rad/s, dopo circa 5 s il robot "in discesa" sta già attraversando la pendenza. Anche la deriva di `alpha_walking` (0,066 rad/s) in un episodio di 20 s porta il robot a circa 75° dalla direzione iniziale. Per questo nelle condizioni di pendenza è stato introdotto il mantenimento della direzione (sezione 3).
- La deriva di direzione diventa una metrica da riportare per entrambe le policy.

---

## 3. Altri risultati

**Riproduzione della scheda di `rough-walk-g` (Fase 0).** La scheda riporta +17% di potenza rispetto ad `alpha_walking` sul piano (1,21 W contro 1,03 W). Noi misuriamo +17% sul piano generato (1,25 W contro 1,07 W) e +10% su `plane`. La metrica di potenza è quindi coerente con quella della scheda. Resta da riprodurre il confronto delle cadute su terreno irregolare: 12/110 contro 32/110.

**Pendenze: la rampa produceva un artefatto, sostituita dalla piramide.**

- **Problema.** Le prime pendenze usavano `FlatRampTerrainCfg`, la rampa del task RollerSlope: un unico parallelepipedo inclinato lungo 14 m. Il confronto diretto, con lo stesso comando costante di 0,3 m/s e la stessa direzione, mostra che produce un artefatto:

  | Cadute / 100 | A | B |
  |---|---|---|
  | Rampa 0° (controllo) | 0 | 0 |
  | **Rampa 2,5° in discesa / salita** | **85 / 79** | **83 / 73** |
  | **Piramide 5° in discesa** | **0** | **0** |
  | Piramide 10° in discesa | 26 | 3 |

  Il codice della rampa funziona finché è piatta, ma già a 2,5° fa cadere entrambe le policy, anche `rough-walk-g`, addestrata su pendenze fino a circa 5,7°. Sulla piramide (`HfPyramidSlopedTerrainCfg`, il terreno del training su terreno irregolare) i risultati sono quelli attesi e coerenti con la scheda: B sopravvive a circa 9-10°, A molto meno. La causa precisa nella rampa non è stata isolata. Due errori precedenti, già corretti, avevano reso illeggibili i primi risultati: episodi interrotti al primo passo da `out_of_terrain_bounds`, e il comando alterato da `rel_forward_envs` (sezione 1).

- **Soluzione.** La rampa viene abbandonata; le pendenze si misurano sulla piramide:
  - un'unica piramide di 40 × 40 m invece di pezzi da 8 × 8. Con 8 m la faccia era lunga solo 3,5 m e dopo circa 17 s il robot risaliva la piramide vicina. Con 40 m la faccia è lunga circa 19 m;
  - mjlab costruisce la piramide come prodotto di due funzioni a tenda (H·tent(x)·tent(y)), quindi la pendenza nominale vale solo sugli assi per il centro. Il robot viene posato **sull'asse**, senza spostamenti laterali casuali;
  - **salita e discesa sono lo stesso compito in direzioni opposte:** stessa partenza a metà faccia (10 m dal centro), all'altezza della superficie calcolata con la stessa formula di mjlab, con circa 9 m disponibili in ciascuna direzione. In discesa il robot è rivolto verso il bordo della piramide, in salita verso la cima;
  - **mantenimento della direzione** con il controllore di mjlab: a ogni passo il comando di rotazione è proporzionale all'errore di direzione rispetto a quella iniziale, limitato a ±1 rad/s. Così la deriva della sezione 2 non porta il robot ad attraversare la pendenza. Modifica il comando ricevuto dalla policy e va dichiarato nei risultati;
  - pendenze 0, 2,5, 5, 7,5, 10 e 15°, in salita e in discesa, con 0° come controllo.

**Candidate per la libreria (criterio della Fase 1).** Sono le condizioni in cui A cade più di B con intervalli di confidenza al 95% che non si sovrappongono. Le cadute delle condizioni a comando casuale restano valide dopo le correzioni.

| Condizione | Cadute A / 100 | Cadute B / 100 |
|---|---|---|
| Gradini a piramide, 1,5 cm | 24 | 1 |
| Superficie irregolare, 2 cm | 16 | 1 |
| Selciato, 1 cm | 14 | 2 |
| Gradini casuali, 1,5 cm | 9 | 0 |
| Pendenza a piramide, circa 10° | 26 | 10 |
| Attrito 0,1 | 92 | 65 |

Sono più di tre situazioni, quindi il criterio della Fase 1 sembra soddisfatto.
- **Carichi fino a 160 g** (+20% circa) e **attrito ≥ 0,3** non peggiorano in modo misurabile: servono livelli più duri.
- **Alcuni terreni hanno dimensioni non verificate,** perché sono i valori predefiniti di mjlab o parametri ipotizzati, e fanno cadere molto entrambe le policy: pietre distanziate (88 / 89), griglia inclinata (82 / 86), gradini aperti (82 / 80), travi strette (79 / 77), blocchi sparsi (45 / 37). Vanno verificati sui parametri realmente usati, registrati in `summary.json`, oppure esclusi.

**Limiti della misura**
- Una caduta è un evento di inclinazione: scivolare o camminare sul posto non conta come caduta.
- Alcune cadute avvengono nei primi secondi dopo la nascita del robot (per esempio con il carico).
- La simulazione su GPU non è perfettamente deterministica: i confronti vanno fatti con gli intervalli di confidenza, non episodio per episodio.

---

## 4. Punti da discutere

1. **Velocità di riferimento del benchmark.** Si propone 0,4 m/s per le condizioni a comando fisso, e di riportare l'inseguimento come rapporto tra velocità reale e comandata.
2. **Mantenimento della direzione sulle pendenze.** Il controllore della direzione modifica il comando ricevuto dalla policy: è accettabile per il benchmark, o si preferisce misurare le pendenze senza correzioni e riportare la deriva a parte?
3. **Situazioni da tenere nella libreria:** quelle della tabella sopra, più le pendenze misurate sulla piramide.
4. **Un primo banco di prova per lo steering.** La zona morta, il guadagno di circa 0,5 e la deriva di direzione sono difetti della policy frozen misurabili già sul piano, senza terreni speciali. Un vettore di steering che migliori l'inseguimento della velocità o corregga la deriva potrebbe essere un primo test semplice del metodo, prima delle situazioni della libreria.
