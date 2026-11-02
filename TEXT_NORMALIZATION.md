# Speech text normalization

The engine expands written forms before sentence splitting and tokenization.
This runs for legacy and v2 models, the CLI, HTTP server, and shared-library
streaming. No model re-export is needed. English and French are supported;
French uses the France number conventions. English spellings use the library's
default US convention (for example, "four hundred fifty").

## Rules and examples

| Input (English) | Spoken text passed to the model |
| --- | --- |
| `$5.2M` | five point two million dollars |
| `$450K` | four hundred fifty thousand dollars |
| `$1,500` | one thousand five hundred dollars |
| `$1.50` | one dollar and fifty cents |
| `$14 million`, `£14 billion` | fourteen million dollars, fourteen billion pounds |
| `$1.5 trillion` | one point five trillion dollars |
| `5.6 × 10^-34`, `5.6e-34` | five point six times ten to the minus thirty-four |
| `5.3e1,200`, `5.3e+1200` | five point three times ten to the one thousand two hundred |
| `Jan. 1, 1980` | January first, nineteen eighty |
| `12 - Mar - 09` | March twelfth, two thousand nine |
| `Mon Jan 1, 1980` | Monday, January first, nineteen eighty |
| `03/12/2026`, `2026-03-12` | three twelve twenty twenty-six, twenty twenty-six three twelve |
| `11 : 06 P. M.` | eleven oh six pee em |
| `2026`, `1943` | twenty twenty-six, nineteen forty-three |
| `1,450,000` | one million four hundred fifty thousand |
| `(212) 555-0142 ext. 402` | two one two, five five five, zero one four two, extension four zero two |
| `87M-34KEL6` | eight seven em, three four kay ee el six |
| `12-34`, `12 - 34` | twelve to thirty-four |
| `554-33-2998` | five five four, three three, two nine nine eight |
| `150-milligrams`, `150-mg` | one hundred fifty milligrams |
| `3-D-printer` | three dee printer |
| `4D`, `9800XT`, `IN2BACH` | four D, nine thousand eight hundred XT, IN two BACH |
| `78oC`, `78°C`, `78ºC` | seventy-eight degrees Celsius |
| `12h`, `30s` | twelve hours, thirty seconds |
| `400 mbps` | four hundred megabits per second |
| `45 lbs` | forty-five pounds |
| `007`, `3.050` | zero zero seven, three point zero five zero |

French accepts space, nonbreaking space, and narrow nonbreaking space grouping,
and a comma or period decimal separator: `1 450 000`, `1,50 €`, `28°C`, `1h`
become `un-million-quatre-cent-cinquante-mille`, `un euro et cinquante centimes`,
`vingt-huit degrés Celsius`, `une heure`. French phone numbers and `poste`
extensions are read digit by digit, following the requested digit policy.

Currency symbols `$`, `€`, `£` work before or after the amount. Compact K/M/B/T
magnitudes are supported directly after a prefix-currency amount.
Written magnitudes (`thousand`, `mille`, `million`, `milliard`, `billion`, `trillion`, with
optional plural s) also stay before the currency name: `$14 million` becomes
"fourteen million dollars" and French `€14 milliards` becomes "quatorze milliards
d'euros". Full-word magnitudes also work before a trailing currency symbol.
French written `billion` retains its French meaning; compact B means milliard.
Compact T means English trillion / French billion (`$1.5T` → "one point five trillion dollars").
Written French `trillion` likewise retains the French long-scale meaning.
Two decimal
places without a magnitude are read as currency subunits. Decimal fractions
otherwise retain every written digit. No floating-point conversion is used.
Integers longer than 18 digits are read digit by digit.

Units include time, weight, distance, volume, percentages, frequency, storage,
and data rates. Matching is case-sensitive: `Mbps`/`mbps` are megabits per second,
`MB/s` is megabytes per second. A bare `m` means meters; minutes use `min`.
Units and known currency forms take priority over identifiers. Compact ASCII
tokens containing digits and uppercase letters only split into alphabetic and
numeric runs: `9800XT` → `9800 XT` → "nine thousand eight hundred XT";
`IN2BACH` → "IN two BACH". Alphabetic runs stay intact; numeric runs are cardinal
values, not year readings or individual digits (`A007` → "A seven"). The previous
spelling branch is kept commented in `read_token()` for easy restoration. Lowercase or
mixed-case tokens such as `mp3` and `iPhone15` are left to the model. Numeric
codes containing hyphens/parentheses are read digit by digit. Structured uppercase
codes such as `87M-34KEL6` and `R2-D2` retain digit/letter spelling, with hyphens
replaced by comma pauses. Number–word compounds instead use ordinary numbers:
`150-milligrams` becomes "one hundred fifty milligrams" and `3-D-printer`
becomes "three dee printer". Hyphenated unit abbreviations such as `150-mg`
expand as units. Parenthesized digit groups and common phone layouts are spelled.
Phone groups separated by spaces, dots, hyphens, or parentheses get comma pauses:
`(310) 799-1029` → "three one zero, seven nine nine, one zero two nine".
Extensions get a preceding comma as well.
English numeric ordinals and common French ordinal suffixes are supported.

Bare, ungrouped English integers from 1000 through 2099 are assumed to be years.
They use natural year readings: `1943` → "nineteen forty-three", `1900` →
"nineteen hundred", `1905` → "nineteen oh five", `2000` → "two thousand", and
`2005` → "two thousand five". From 2010 onward the two-pair reading applies,
as in `2026` → "twenty twenty-six". Year-range endpoints use the same policy.
Recognized currencies, units, ordinals and identifiers take priority; `$2026`,
`2026 kg`, grouped `2,026`, and decimal `2026.5` retain ordinary number readings.
This intentionally assumes a year without interpreting surrounding prose;
French years retain their usual cardinal readings.

Outside recognized phone formats, two unsigned integer groups separated by a
hyphen (with optional spaces) are read as a range: `12-34` becomes "twelve to
thirty-four" / "douze à trente-quatre". A following hyphen, even after spaces,
prevents range matching. Three-or-more-group numeric codes are spelled with
comma pauses, so `554-33-2998` is not partially converted into a range.

Scientific notation accepts `e`/`E` or `x`/`×` followed by `10^`, with optional
spaces around the multiplication and power signs. Mantissas may be signed and
decimal; exponents are signed integers with optional comma thousands grouping.
An explicit positive exponent sign is silent. Fractional mantissa digits are
preserved, and exponents use ordinary number readings rather than years.
French accepts decimal commas or periods: `2,7E-35` becomes "deux virgule sept
fois dix à la puissance moins trente-cinq". Scientific notation takes priority
over compact uppercase identifiers, so `2E3` is a power of ten.

English written-month dates accept full names and common case-insensitive
abbreviations, with optional dots: `Jan. 1, 1980`, `12-MAR-09`, `12 March 2009`,
and `May 5`. Days become ordinals and month names expand. Optional weekday
names/abbreviations expand only before recognized written-month dates; standalone
`Mon` and `Fri` remain unchanged. Weekdays are read as supplied, not recalculated.
Two-digit years use 2000–2069 for `00`–`69` and 1970–1999 for `70`–`99`.

Numeric dates accept `MM/DD/YYYY`, `MM-DD-YYYY`, and year-first forms
`YYYY-MM-DD`/`YYYY/MM/DD`, also with two-digit trailing years. Month/day components
stay cardinal and in written order; the year uses the existing language reading.
Ambiguous month-first dates always use MM/DD, including in French. Calendar
validation includes leap years. French written-month dates are not yet recognized.

Times accept `HH:MM`, optional `:SS`, and optional `AM`/`PM` with dots/spaces
between the letters. Hours must be 0–23, minutes/seconds 0–59, and AM/PM hours
1–12. English minutes 01–09 use "oh"; whole hours use "o'clock" without AM/PM,
or omit the zero minutes with AM/PM. AM/PM becomes "ay em"/"pee em". Seconds
are explicit: `23:05:09` → "twenty-three oh five and nine seconds". French uses
hours/minutes and optional seconds, with `du matin`/`de l'après-midi` for AM/PM.
Spaces and tabs around date/time separators are accepted.

Unrecognized colon/slash forms containing numbers, and unrecognized hyphenated
written-month forms, fall back to alphabetic/numeric runs separated by spaces:
`25:99` → "twenty-five ninety-nine", `32-Mar-09` → "thirty-two Mar nine".
Existing phone, range, unit and structured-code rules retain their meanings;
URLs and emails remain protected. Numeric hyphen chains that fail date validation
retain the earlier identifier policy (digits and comma pauses).

These are deterministic reading policies, not a general understanding of text.
Other date/time formats, fractions and arbitrary
country-specific phone formats do not yet have dedicated rules. French space
grouping can make a sequence such as `81 200 201` look like one large number;
use punctuation between independent values. URLs with a protocol or `www.`
and email addresses are preserved for a future dedicated reading policy.

## External pronunciation lexicons

The default files are `lexicon.english.csv` and `lexicon.french.csv`, searched in
the process working directory and loaded once per engine. They are optional;
missing default files mean no substitutions. Use `--lexicon /path/custom.csv`
to select one explicitly; a missing or malformed explicit file is an error.
Restart the server/engine after editing a lexicon, or use the FFI setter below.

```csv
source,replacement
Mr.,Mister
Dr.,Doctor
A.I.,eigh eye
AI,eigh eye
MacOS,mac oh ess
"Acme, Inc.",Acme Incorporated
/r\bFY(20\d\d)\b,fiscal year $1
```

The header is optional. Files must be UTF-8; a BOM and CRLF are accepted.
Blank lines and lines starting with `#` are ignored. CSV quoting supports
commas, doubled quotes, and embedded newlines. Empty replacements delete a
match; empty sources and duplicate sources are errors.

Source prefixes select matching behavior:

| Prefix | Match mode | Boundaries |
| --- | --- | --- |
| none | Case-sensitive literal text | Implicit |
| `/i` | Case-insensitive literal text | Implicit |
| `/r` | Case-sensitive ECMAScript regex | Explicit |
| `/R` | Case-insensitive ECMAScript regex | Explicit |
| `//` | Case-sensitive literal beginning with `/` | Implicit |

Prefixes are adjacent to the text/pattern; any following whitespace belongs to
the match. Ordinary entries are literal and case-sensitive: `AI` will not change
`RAIN`, `xAI`, or `ai`. `/iAI` also matches `ai` and `Ai`, with the same implicit
boundaries. `/i` compares Unicode uppercase forms, including French accents
(`café`/`CAFÉ`); it does not normalize composed/decomposed Unicode spellings.

```csv
/iMacOS,mac oh ess
/R\bFY(20\d\d)\b,fiscal year $1
//api,slash api
```

The last entry matches literal `/api`, not `api` or `/API`. The escape removes
one leading slash: `///api` matches literal `//api`. To match a leading slash
case-insensitively, use `/i/api`. Unknown prefixes and empty text/patterns are
errors. Migrate old `//pattern` regex rows to `/rpattern`; `//` is now exclusively
a literal slash escape.

For `/r` and `/R`, everything after the prefix is the regex; there is no closing
slash delimiter. Regex boundaries are **explicit**: use `/r\bFY(20\d\d)\b`
for complete words, or `/rFY(20\d\d)` to allow substring matches. `/R` matches
case-insensitively while preserving original text in captured groups. Anchors `^`
and `$` refer to the original input, not each suffix searched internally.
Regex syntax uses UTF-8 bytes: `\b`/`\w` do not provide Unicode word segmentation.
`/R` uses `std::regex::icase`, not full Unicode case folding; ASCII letter case
is supported. For accented case-insensitive words, prefer `/i` literal entries.
Literal entries retain the engine's existing boundary handling for Latin text.

Regex replacements support `$1` through `$99` for captured groups, `$&` for the
whole match, and `$$` for a literal dollar sign. Literal replacements do not
interpret capture syntax. Backslashes in the CSV are written once, as shown
above. Fields containing commas must be CSV-quoted, including regex quantifiers:

```csv
"/r\b([A-Z]{2,3})-(\d+)\b",$1 number $2
```

Rules are compiled/loaded once per engine. Invalid regexes fail with the filename
and CSV record number. At each text position, the first matching CSV row wins
across literal and regex entries, regardless of match length. Put specific
exceptions before broad rules; for example, put `AI Lab,the laboratory` before
`AI,eigh eye` if you want the whole phrase replaced. Each regex uses its normal
ECMAScript matching semantics internally. Empty regex matches insert once at
that position and advance over the next character, preventing infinite loops.
Matched/replacement UTF-8 must remain valid.

Replacements are applied once, without recursively matching their output, then
numeric rules run. Thus `FY2026` becomes "fiscal year 2026" and then "fiscal year
twenty twenty-six". Phonetic spellings are ordinary model input, not IPA or phoneme tokens.
The provided `etc.` entry uses the standard spelling "et cetera".

Lexicons apply before chunking, so an expanded `Dr.` cannot falsely end a
sentence. Lexicon selection is based on the bundle's language (legacy defaults
to English). Ship the CSV beside your launch directory, or use an absolute path;
the number-spelling rules themselves are embedded in the binary.

## Controls and inspection

```sh
./ptts --models-dir models-v2-int8-linear-decoder/english \
  --inspect-text 'Dr. Smith paid $1,500 for 12h.'
./ptts --lexicon /path/lexicon.english.csv 'Your text' narrator output.wav
```

`--inspect-text` includes `normalized_text` as well as prepared chunks and token
IDs (v2 bundles). `--trace` logs changed text during synthesis.
`--no-lexicon` disables only CSV replacements. `--no-normalize-text` bypasses
this entire new layer, preserving the previous engine preprocessing.

The shared library enables the same defaults. Configure an existing handle
before synthesis, never concurrently with a request:

```c
int ptt_set_text_options(void* handle, int enabled, const char* lexicon_path);
// enabled = 0: bypass everything; nonzero: enable normalization.
// lexicon_path = NULL: automatic language file; "-": no CSV; otherwise a path.
// Returns 0 on success, -1 on failure; failure preserves the previous options.
```

## Implementation and validation

`native_text/normalizer.cpp` recognizes context, then uses libnumbertext's
English/French Soros rules for number spelling. The small interpreter and two
rule files are vendored at a pinned revision under the BSD license option; see
`native_text/numbertext/README.md` and `COPYING`. There is no Python, ICU, Boost,
installed locale, or runtime rule-file dependency. Compiled rules are initialized
on the first numeric conversion and shared across engines.

```sh
cmake -S . -B .build-v2 -DBUILD_SHARED_LIB=ON \
  -DPOCKET_OUTPUT_DIR="$PWD/.cache/text-normalization/bin"
cmake --build .build-v2 -j4 --target ptts pocket-tts-lib normalizer-probe
python scripts/test_text_normalization.py \
  --probe .cache/text-normalization/bin/normalizer-probe \
  --cli .cache/text-normalization/bin/ptts
```

The probe tests written-to-spoken meaning and CSV behavior independently of
model inference. CLI integration checks verify the language-specific expansions
reach chunking and that the bypass preserves raw input. Generated speech still
depends on the checkpoint; normalization cannot guarantee every phonetic
spelling or every word will be pronounced correctly.

For Apple attention bundles, configure with `-DPOCKET_ACCELERATE=ON` explicitly.
Editing an `option()` default in CMakeLists.txt does not override an existing
cached OFF value. Configure Android with `-DPOCKET_ACCELERATE=OFF`. CMake prints
the effective Accelerate setting during configuration. Run the executable in
your configured `POCKET_OUTPUT_DIR` (the test command above uses the cache path).

Build `benchmark-text-normalization` and run it with an optional repeat count
(default 1,000). It reports constructor/CSV-loading time, first call, and warm
mean/median/p95 separately, including default lexicons. With the output directory
above, run `.cache/text-normalization/bin/benchmark-text-normalization 1000`.

On the development Mac with the default FY regex enabled, 1,000 repetitions per
case measured warm medians of 0.072 ms (98-byte prose), 0.139 ms (100-byte
currency example), 0.252 ms (84-byte range/unit example), 0.041 ms (phone/codes),
0.203 ms (FY capture replacements), 0.834 ms (440-byte dense numeric paragraph),
and 0.088 ms (French). First numeric rule initialization cost 14 ms for English
and 9 ms for French, once per language per process. Constructor/CSV loading cost
0.06–0.46 ms in this run. These are normalization
timings only; Android latency has not been measured. The full report is
`.cache/text-normalization/timing-latest.json`.
