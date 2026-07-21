# Gedit — SPEC v1

Drugi model w rodzinie [[microg|MicroG]]: lokalny, w 100% własny model do edycji zdjęć, wywoływany głosem z [[gzowo-ai]] ("zrób mi zdjęcie i spraw żeby wyglądało jak X"), edytujący klatkę z kamery LOKALNIE (nie przez zewnętrzne API) i pokazujący wynik w oknie asystenta.

Status: **spec do zatwierdzenia, przed pierwszym kodem** (ta sama reguła co przy MicroG i gzowo-ai).

## 0. Nazwa i marka (2026-07-21)

**Gedit** — "G" (Gzowo, ten sam rdzeń co MicroG) + "edit" (dokładnie to, co model robi). Świadoma kolizja nazwy z linuksowym edytorem tekstu `gedit` — inny kontekst, mało prawdopodobne pomylenie w praktyce, zaakceptowana przez Jurka.

**Logo: monogram, ten sam szkielet geometrii co µG.** Lewy znak nie jest literą — to **crop-mark** (dwa przeciwległe narożniki L, uniwersalna ikona kadrowania/edycji zdjęć), stojący w miejscu µ dokładnie tak, jak µ stoi za słowem "micro". Zajmuje IDENTYCZNY bounding box co µ w MicroG (x 60-132, y 74-172), więc para czyta się na tej samej wadze optycznej i nie wymaga przeliczania transformu w ikonie. Prawa strona (G, promień 50, środek 200,90) jest **dosłownie tą samą ścieżką SVG** co w MicroG — ta sama grubość kreski 15, ta sama linia bazowa 140.

Źródła: `Design/gedit-{mark,mark-inverse,icon}.svg` w tym folderze, zbudowane wg dokładnie tego samego wzorca co `MicroG/Design/microg-*.svg` (skopiowana konstrukcja, podmieniony tylko lewy glif). Rasteryzacja do `.icns` — ten sam pipeline co MicroG (`Design/build-icons.sh` + `rasterize.html`/`raster-server.py`), do skopiowania i podmiany nazw przy realnym buildzie.

Slogan — do ustalenia razem z liczbą parametrów finalnego modelu (wzorem MicroG "100M parameters. 100% Gzowo."); robocza wersja: **"Small pixels. 100% Gzowo."**

## 1. Zakres funkcjonalny

- Edycje: **style artystyczne/filtry** (vintage, cyberpunk-glow, czarno-białe itd.) + **proste dodawanie/usuwanie obiektów** (inpainting).
- Komenda: dowolny opis słowny, mapowany na tekstowy embedding — bez zamkniętej listy nazwanych stylów.
- Każda edycja zawsze wychodzi od **świeżego zdjęcia z kamery** (bez łańcuchowania edycji na edycjach).
- Wynik zawsze się pokazuje, bez automatycznego odrzucania słabych wyników (ocena "na oko" przez Jurka) — na start.
- iMessage (wysyłanie zdjęcia do kogoś) — **odłożone**, dorabiamy po tym, jak core (kamera→edycja→podgląd) działa stabilnie.

## 2. Model — architektura i trening

- **W 100% własny model dyfuzyjny** (mała liczba kroków odszumiania), trenowany **od zera**, jak przy MicroG — ale świadomie w ograniczonej skali:
  - niska rozdzielczość treningowa/inference (64–128 px), upscaling do pełnej rozdzielczości **decyzja odłożona** do zobaczenia jakości pierwszych wyników (Lanczos vs gotowy model super-res).
  - koder tekstu (zamiana opisu słownego na embedding) — **gotowy, zamrożony pretrenowany CLIP text encoder**, NIE trenowany przez nas. Cały obrazowy U-Net/diffusion model, który faktycznie edytuje zdjęcie, jest w 100% własny i trenowany od zera — to standardowa praktyka (tak samo robi Stable Diffusion).
- **Dataset**: [huggingface.co/datasets/timbrooks/instructpix2pix-clip-filtered](https://huggingface.co/datasets/timbrooks/instructpix2pix-clip-filtered) (pary przed/po + instrukcja edycji, wygenerowane przez Prompt-to-Prompt). Dataset jest duży (setki tysięcy par) — bierzemy **mały podzbiór (rząd 10-30 tys. par)** w niskiej rozdzielczości, żeby trening był wykonalny w kilka-kilkanaście godzin, nie dni/tygodnie.
- **Trening: Kaggle** (darmowe GPU T4x2, limit ~30h/tydzień), ten sam workflow co przy MicroG — checkpointy + wznawianie sesji co ~12h.
- **Jakość**: brak automatycznej metryki na start, ocena wizualna przez Jurka po każdym checkpoincie (tak jak przy MicroG).

## 3. Inference i integracja z gzowo-ai

Zweryfikowane w realnym kodzie gzowo-ai (`~/Downloads/Claude/Projects/Gzowo AI/v1`):

- **Eksport**: PyTorch (trening) → **ONNX** → inference w bridge'u przez `onnxruntime-node`. Bridge to Node.js (porty 8787/8788), zero Pythona w runtime — `bridge/package.json` obecnie ma tylko `@google/genai`, trzeba dodać `onnxruntime-node`.
- **Rozszerzenie istniejącego trybu Vision** w gzowo-ai, nie nowy osobny tryb:
  - kamera: `js/vision/camera.js` (`cameraVision` singleton) — nowa funkcja `captureFrame()` (już istnieje) do jednorazowego zdjęcia, bez zmian w istniejącym 1fps live-feed do Gemini.
  - nowe narzędzie w `js/vision/vision-tools.js`, zarejestrowane przez `toolRouter.registerTool({name: 'edit_camera_photo', description, parameters}, handler)` — wzorzec identyczny jak istniejące `create_3d_model` (capture → POST do bridge → poll job → widget wyniku).
  - **wzorzec job-queue** z `bridge/model3d-jobs.js` (`createModelJob()` zwraca job id od razu, `setImmediate` robi robotę w tle, klient polluje `GET /.../jobs/:id`) — bo inference dyfuzyjny może zająć kilka-kilkanaście sekund na CPU/MPS Twojego Maca, nie może blokować requesta.
  - nowy plik `bridge/photo-edit.js` z handlerem inference (analogicznie do `bridge/whisper.js`, ale ONNX zamiast subprocess) + nowe endpointy dodane do dispatcher-a w `bridge/server.js` (wzorzec przy linii ~1247 i ~1296-1351).
- **Plik modelu (ONNX)**: poza paczką .app, w `~/Library/Application Support/Gzowo AI/` (tak jak dziś sekrety/certy — patrz `desktop/main.js`, `userDataPath`/`userData` pattern) — bridge ładuje go stamtąd przy starcie. Model NIE wchodzi do `desktop/package.json` `extraResources` whitelisty razem z kodem.
- **Historia edycji**: nowy IndexedDB store wzorowany 1:1 na `js/model3d/store.js` (`gzowo-models` → analogicznie `gzowo-photo-edits`, rekordy `{id, prompt, before(dataURL), after(dataURL), createdAt}`). To **świadomy wyjątek** od zasady "klatki Vision nigdy nie trafiają na dysk" — dotyczy WYŁĄCZNIE finalnego zdjęcia przed/po tej jednej funkcji, reszta trybu Vision (co widzisz, OCR, itd.) zostaje efemeryczna jak dziś.
- **Wyświetlanie wyniku**: w tym samym oknie asystenta, jako panel/widget obok rozmowy (spójne z resztą UI, styl Gzowo Aperture — mono, `.glass` gdzie pasuje).

## 4. Co NIE wchodzi w ten etap

- iMessage / wysyłanie zdjęcia (osobna runda po core).
- Automatyczna kontrola jakości wyników (heurystyki odrzucające szum/czarny obraz).
- Łańcuchowe edycje (edycja na edycji).
- Upscaling ponad Lanczos (decyzja po pierwszych wynikach treningu).

## 5. Otwarte pytania do rozstrzygnięcia PO pierwszych wynikach treningu

- Upscaling: Lanczos vs gotowy model super-res.
- Czy 64px czy 128px daje sensowną jakość przy tej skali datasetu — wybór finalnej rozdzielczości.
- Dokładna liczba kroków odszumiania (trade-off jakość vs czas inference na CPU/MPS).
- Finalny slogan po ustaleniu liczby parametrów modelu.

## Kolejne kroki

1. Jurek zatwierdza tę specę ("go").
2. Zbudowanie `Design/gedit-{mark,mark-inverse,icon}.svg` → PNG → `.icns` (skopiować i zaadaptować pipeline z MicroG).
3. Przygotowanie podzbioru datasetu (10-30k par, niska rozdzielczość) + notebook treningowy na Kaggle (wzorzec z MicroG).
4. Trening + iteracja jakości (checkpointy, ocena wizualna).
5. Eksport ONNX, dodanie `onnxruntime-node` do bridge'a gzowo-ai, `bridge/photo-edit.js` + endpointy.
6. `edit_camera_photo` tool + widget wyniku + IndexedDB historia w gzowo-ai.
7. Test end-to-end na realnym Macu, potem iMessage jako runda 2.

Powiązane: [[microg]], [[gzowo-ai]], [[stack]], [[identity]]
