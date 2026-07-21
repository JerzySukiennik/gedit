# Trening Gedit na Kaggle — instrukcja

Ten sam plan co przy [[microg|MicroG]]: **30 h GPU tygodniowo za darmo** (T4 ×2), sesja ucinana po 12 h, wszystko poniżej zbudowane pod wznawianie.

Czego potrzebujesz: konta na kaggle.com z **zweryfikowanym numerem telefonu**.

---

## Dlaczego nie wysyłamy danych z domu

`timbrooks/instructpix2pix-clip-filtered` jest ogromny — bierzemy tylko pierwsze ~20k par w 128px, ale nawet to lepiej ściągnąć na szybkim łączu Kaggle niż z domu. **Pobieramy i pakujemy dane na Kaggle**, raz, zapisujemy jako Dataset. Każda kolejna sesja treningowa startuje z gotowych binariów w kilka sekund.

---

## Krok 1 — przygotowanie danych (raz, ~30-60 min)

1. **New Notebook**
2. Prawy panel → **Accelerator: None**, **Internet: On**
3. *(opcjonalnie, przyspiesza pobieranie)* **Add-ons → Secrets → Add secret**, nazwa `HF_TOKEN`, wartość = Twój token z HuggingFace
4. Wklej zawartość [`01-prep.py`](01-prep.py) do komórki i uruchom
5. **Save Version → Save & Run All**, poczekaj aż się wykona
6. Na stronie wersji: **Output → New Dataset**, nazwij **`gedit-data`**

Powstaną `gedit_images.bin` (obrazy przed/po, uint8), `gedit_text.bin` (embeddingi CLIP instrukcji edycji, float32) i `gedit_meta.json`.

## Krok 2 — trening (czas nieznany do pierwszego pomiaru)

1. **New Notebook**
2. **Accelerator: GPU T4 x2**, **Internet: On**, **Persistence: Variables and Files**
3. **Add Input** → dataset `gedit-data`
4. Wklej [`02-train.py`](02-train.py) i uruchom
5. Gdy sesja się skończy (albo padnie): **Save Version**, potem **Output → New Dataset**, nazwij **`gedit-ckpt`**

## Krok 3 — wznowienie (jeśli 12 h nie starczyło)

Nowy notebook jak w kroku 2, ale **Add Input** dwa razy: `gedit-data` **oraz** `gedit-ckpt`. Skrypt sam wykryje `ckpt.pt` i podejmie od ostatniego kroku — razem z momentami Adama, więc bez skoku loss.

Po każdej sesji nadpisuj `gedit-ckpt` nowym outputem (**Output → Update Dataset**).

---

## Czego się spodziewać

`BATCH, ACCUM, STEPS` w `02-train.py` to **placeholder** — w przeciwieństwie do MicroG (gdzie harmonogram był już zmierzony), tu jeszcze nie mamy pomiaru realnego throughputu na obrazach 128px. Pierwsza sesja: patrz na tokeny/s... właściwie kroki/s w logu, dopiero potem doprecyzuj `STEPS` tak, żeby zmieściło się w rozsądnej liczbie sesji 12h.

**Loss startowy to najważniejsza liczba w pierwszej minucie** — dla straty MSE na przewidywaniu szumu (wariancja jednostkowa) startowy loss powinien być **w okolicach 1.0**. Wartość dużo wyższa albo `nan` od razu = coś jest zepsute (zły zakres pikseli, zła normalizacja) — sprawdź przed inwestowaniem godzin GPU.

## Jak sprawdzić, czy idzie dobrze

Val loss (log co `--eval-every`) powinien systematycznie spadać poniżej 1.0 w pierwszych ~500 krokach. Jeśli stoi w miejscu przez kilkaset kroków — learning rate za niski albo dane zepsute. Jeśli skacze do `nan` — zbij `--lr`.

## Jak coś nie działa

- **`CUDA out of memory`** → zbij `BATCH` w `02-train.py`, podnieś `ACCUM` (ta sama liczba przykładów na krok)
- **DataParallel się sypie** → dopisz `"--single-gpu"` do listy `cmd` w `02-train.py`
- **Notebook nie widzi danych** → sprawdź, czy dataset jest faktycznie dodany w **Add Input**, nie tylko utworzony
- **Val loss dużo wyższy niż train loss** → 300 przykładów walidacyjnych (domyślne `--val-n`) to mało przy małym batchu, trochę szumu jest normalne
