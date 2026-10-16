"""Oracle for tests/test_dates_oracle.rs (C4, collection-date parsing).

Every expected value is computed here by calling the Python reference:

* headers: validation/run_danno.py parse_date (EXTERNAL_DATES empty), i.e.
  parse_dc then parse_underscore_year, with ymd_to_decimal.
* metadata values: validation/fetch_dates.py parse_collection_date (the
  GenBank collection_date forms run_danno.py --dates consumes). That function
  does not validate the calendar (it raises on a bad month name or month
  number and returns a value for 2013-02-30), so for day-precision strings the
  expected value is run_danno.ymd_to_decimal of the same fields, which returns
  None for impossible dates; the generator asserts the two agree whenever both
  give a value.

Two Rust rules have no Python counterpart and are labelled as such in the
fixture ("rule" column), with values still computed here, not typed:

* "pipe": a header '|' field holding a metadata-form date (ID|2013-10-22),
  value = parse_collection_date(field) under the rule above, accepted only in
  calendar years 1950-2035 (value in [1950, 2036)), so numeric ID fields such
  as PT|1001 or PT|0042 are not dates. This is a Rust rule, not an oracle:
  run_danno.parse_date leaves every such header undated;
* "decimal": a metadata value that is a decimal year in [1950, 2035]
  (DC meta.csv form, 2013.806849), value = float(s).

Also writes dc_headers.csv: every DC FASTA header with run_danno.parse_date.

Run from the repo root:
  python3 crates/trace50-core/tests/fixtures/dates/gen_oracle.py [DC msa.fasta]
"""
import csv
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, *(['..'] * 5)))
VALIDATION = os.path.join(os.path.dirname(REPO), 'validation')
sys.path.insert(0, VALIDATION)
import run_danno  # noqa: E402
import fetch_dates  # noqa: E402

assert not run_danno.EXTERNAL_DATES


def meta_expected(s):
    """(value or None, rule)."""
    try:
        v, prec = fetch_dates.parse_collection_date(s)
    except (KeyError, IndexError):
        v, prec = None, 'missing'
    t = (s or '').strip().strip('"')
    if '/' in t:
        t = t.split('/')[0].strip()
    m = re.fullmatch(r"(\d{1,2})-([A-Za-z]{3})-(\d{4})", t)
    ymd = None
    if m and m.group(2).upper() in fetch_dates.MONTHS:
        ymd = (int(m.group(3)), fetch_dates.MONTHS[m.group(2).upper()], int(m.group(1)))
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", t)
    if m:
        ymd = tuple(int(g) for g in m.groups())
    if ymd is not None:
        r = run_danno.ymd_to_decimal(*ymd)
        if r is not None and v is not None:
            assert r == v, (s, r, v)
        return r, 'fetch_dates+ymd_to_decimal'
    m = re.fullmatch(r"(\d{4})-(\d{2})", t)
    if m and not 1 <= int(m.group(2)) <= 12:
        return None, 'fetch_dates (month out of range)'   # fetch_dates: cum[-1]
    if v is not None:
        return v, 'fetch_dates'
    try:
        f = float(t)
        if 1950.0 <= f <= 2035.0:
            return f, 'decimal'
    except ValueError:
        pass
    return None, 'none'


def header_expected(h):
    v, _ = run_danno.parse_date(h)
    if v is not None:
        return v, 'run_danno.parse_date'
    if '|' in h:
        for field in h.strip().split('|'):
            fv, _ = meta_expected(field)
            if fv is not None and 1950.0 <= fv < 2036.0:
                return fv, 'pipe'
    return None, 'none'


HEADERS = [
    'B.US.2013.DC1000_0131022_PR_RT.MF455515',   # day
    'B.US.2013.DC1001_0130111_PR_RT.MF455516',   # day
    'B.US.2012.DC5_0120229_PR_RT.X1',            # leap day
    'B.US.2013.DC6_0130229_PR_RT.X2',            # 29 Feb in a common year -> year
    'B.US.2013.DC7_0121022_PR_RT.X3',            # YY disagrees with year -> year
    'B.US.2013.DC8_0131301_PR.X4',               # month 13 -> year
    'B.US.2013.DC9_PR_RT.X5',                    # no block -> year
    'B.US.2013.DC10_01310221_PR.X6',             # 7 digits after _0 -> no block
    'B.US.2014.DC11_0141231',                    # block at end of field
    'b.US.2013.DC1_0131022_PR.X',                # lower-case subtype still DC
    'B.us.2013.DC1_0131022_PR.X',                # not DC (country), no _YYYY
    'MH074765.1 HIV-1 isolate p6s6_2013 from Germany pol gene',
    'X_2009.86 clone',
    'X_2009.86',
    'X_2009.5_y',
    'X_2009.861',
    'X_1975',
    'X_1975_2001',
    'X_2031.5_2012',
    'seq1_2015.0',
    'seq4_2018.0',
    'ID|2013-10-22',
    'ID|2013.81',
    'ID|2013',
    'ID|22-Oct-2013',
    'ID|2013-02-30',
    'PT|1001|clone0',                            # numeric ID field, not a year
    'PT|0042',
    'HXB2|1234|x',
    'ID|1949-12-31',                             # outside 1950-2035
    'ID|2036',
    'ID|2035',
    'ID|1950-01-01',
    'seq1',
    'A_B_C',
]

META = [
    '2013-10-22', '2013-01-11', '2012-02-29', '2013-02-29', '2013-02-30',
    '2016-12-31', '2013-10', '2013-02', '2012-02', '2013', '22-Oct-2013',
    '1-mar-2012', 'Oct-2013', 'Feb-2012', '2011/2012', '01-Jan-2011/31-Dec-2011',
    '2013.806849', '2013.028767', '"2013-10-22"', ' 2013-10-22 ', '',
    'NA', '10/22/2013', '22/10/2013', '2013-13', '2013-00-10', '2013-1-5',
    '1949.5', '2040.1', 'Foo-2013',
]


def enc(v):
    return None if v is None else repr(v)


out = {'headers': [], 'meta': []}
for h in HEADERS:
    v, rule = header_expected(h)
    out['headers'].append({'input': h, 'expected': enc(v), 'rule': rule})
for s in META:
    v, rule = meta_expected(s)
    out['meta'].append({'input': s, 'expected': enc(v), 'rule': rule})
with open(os.path.join(HERE, 'oracle.json'), 'w') as fh:
    json.dump(out, fh, indent=1)
print(json.dumps(out, indent=1))

if len(sys.argv) > 1:
    with open(sys.argv[1]) as fh, open(os.path.join(HERE, 'dc_headers.csv'), 'w', newline='') as oh:
        w = csv.writer(oh)
        w.writerow(['header', 'expected'])
        for line in fh:
            if line.startswith('>'):
                h = line[1:].strip()
                v, _ = run_danno.parse_date(h)
                w.writerow([h, '' if v is None else repr(v)])
