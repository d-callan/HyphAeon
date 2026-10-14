"""
aeon_core/temporal.py
--------------------
Date parsing utilities for temporal analysis.

These are shared by hyphaeon (temporal selection dynamics) and
chronaeon (molecular clock dating, triage).
"""

import json
import re
import datetime
from pathlib import Path
from typing import Dict, Union, Any

import numpy as np
import pandas as pd


def parse_date_to_decimal(val: Any, time_units: str = "years") -> float:
    """
    Converts various date representations into a float time coordinate.
    Calendar (time_units='years'):
      - float or int: 2021.25 -> 2021.25
      - ISO string: "2021-04-15" -> 2021.2868
      - Partial ISO: "2021-04" or "2021-04-XX" -> 2021.2868 (mid-month)
      - Year only: "2021" or "2021-XX-XX" -> 2021.5 (mid-year)
      - Slash formatted: "2021/04/15" -> 2021.2868
      - Non-year-first: "15/04/2021" (DD/MM/YYYY, tried first), "04/15/2021"
        (MM/DD/YYYY), "04/2021" (MM/YYYY -> mid-month)
    Non-calendar (time_units in {'generations','days','arbitrary'}):
      - any non-negative number, or the first number embedded in a string
        ("gen_5000" -> 5000.0). No [1800,2100] gate.
    """
    if val is None or pd.isna(val):
        return np.nan

    # Non-calendar time coordinates: accept any non-negative real (no year gate)
    if time_units in ("generations", "days", "arbitrary"):
        try:
            val_f = float(val)
            return val_f if val_f >= 0.0 else np.nan
        except (ValueError, TypeError):
            pass
        m = re.search(r'(\d+(?:\.\d+)?)', str(val))
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return np.nan
        return np.nan

    if isinstance(val, (int, float)):
        val_f = float(val)
        if 1800.0 <= val_f <= 2100.0:
            return val_f
        return np.nan

    val_str = str(val).strip()
    if not val_str or val_str.lower() in ['unknown', 'nan', 'none', 'na', '?']:
        return np.nan

    # Try direct float conversion (e.g., "2021.25")
    try:
        val_f = float(val_str)
        if 1800.0 <= val_f <= 2100.0:
            return val_f
    except ValueError:
        pass

    # Normalize delimiters
    clean_str = val_str.replace('/', '-').replace('.', '-')
    parts = clean_str.split('-')

    if len(parts) >= 1 and parts[0].isdigit() and len(parts[0]) == 4:
        try:
            year = int(parts[0])
            if not (1800 <= year <= 2100):
                return np.nan

            month = 6
            if len(parts) >= 2 and parts[1].isdigit():
                m = int(parts[1])
                if 1 <= m <= 12:
                    month = m

            day = 15
            if len(parts) >= 3 and parts[2].isdigit():
                d = int(parts[2])
                if 1 <= d <= 31:
                    day = d

            dt = datetime.date(year, month, min(day, 28 if month == 2 else 30))
            start_of_year = datetime.date(year, 1, 1)
            days_in_year = 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365
            return float(year + (dt - start_of_year).days / days_in_year)
        except Exception:
            return np.nan

    # Non-year-first formats (coverage inherited from r0.parse_calendar_date):
    # 'DD/MM/YYYY', 'MM/DD/YYYY', and 'MM/YYYY' ('/' normalized to '-' above).
    # Ambiguous D/M orders keep the historical r0 precedence: day-first is
    # attempted before month-first.
    def _frac(year: int, month: int, day: int) -> float:
        dt = datetime.date(year, month, min(day, 28 if month == 2 else 30))
        start = datetime.date(year, 1, 1)
        days_in_year = 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365
        return float(year + (dt - start).days / days_in_year)

    if (
        len(parts) == 3
        and all(p.isdigit() for p in parts)
        and len(parts[2]) == 4
        and 1800 <= int(parts[2]) <= 2100
    ):
        d0, m0, y0 = int(parts[0]), int(parts[1]), int(parts[2])
        for day, month in ((d0, m0), (m0, d0)):  # DD/MM/YYYY, then MM/DD/YYYY
            if 1 <= month <= 12 and 1 <= day <= 31:
                return _frac(y0, month, day)

    if (
        len(parts) == 2
        and all(p.isdigit() for p in parts)
        and len(parts[1]) == 4
        and 1800 <= int(parts[1]) <= 2100
    ):
        month, year = int(parts[0]), int(parts[1])
        if 1 <= month <= 12:
            return _frac(year, month, 15)  # mid-month, canonical convention

    return np.nan


def parse_dates_from_auspice_json(json_path: Union[str, Path]) -> Dict[str, float]:
    """Recursively extracts tip node dates from Nextstrain Auspice JSON v2."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    tree_root = data.get('tree', data)
    dates: Dict[str, float] = {}

    def recurse(node: Dict[str, Any]):
        children = node.get('children', [])
        if not children:
            name = node.get('name')
            if not name:
                return
            attrs = node.get('node_attrs', {})
            val = np.nan

            if 'num_date' in attrs and isinstance(attrs['num_date'], dict) and 'value' in attrs['num_date']:
                val = parse_date_to_decimal(attrs['num_date']['value'])
            elif 'date' in attrs and isinstance(attrs['date'], dict) and 'value' in attrs['date']:
                val = parse_date_to_decimal(attrs['date']['value'])
            elif 'year' in attrs and isinstance(attrs['year'], dict) and 'value' in attrs['year']:
                val = parse_date_to_decimal(attrs['year']['value'])
            elif 'num_date' in attrs and isinstance(attrs['num_date'], (int, float, str)):
                val = parse_date_to_decimal(attrs['num_date'])
            elif 'date' in attrs and isinstance(attrs['date'], (int, float, str)):
                val = parse_date_to_decimal(attrs['date'])

            if np.isnan(val):
                # Try parsing timestamp from the tip name itself
                val = extract_date_from_string(name)

            if not np.isnan(val):
                dates[name] = val
        else:
            for child in children:
                recurse(child)

    recurse(tree_root)
    return dates


def extract_date_from_string(name: str, time_units: str = "years") -> float:
    """Extracts a time coordinate from a string / FASTA header using standard patterns."""
    if not name:
        return np.nan

    # Non-calendar: match embedded generation/day tokens e.g. _gen2000, |gen_5000,
    # _20000gen, _g50000, or a bare trailing number after a delimiter (|5000).
    # Prioritize explicit unit prefix/suffix before matching bare delimiter-bound numbers.
    if time_units in ("generations", "days", "arbitrary"):
        m = re.search(r'(?:[\|/_\-\s]|^)(?:gen|generation|g|day|d|t)[\-_]?(\d+(?:\.\d+)?)(?:[\|/_\-\s]|$)', name, re.IGNORECASE)
        if not m:
            m = re.search(r'(?:[\|/_\-\s]|^)(\d+(?:\.\d+)?)(?:gen|g|d)(?:[\|/_\-\s]|$)', name, re.IGNORECASE)
        if not m:
            m = re.search(r'(?:[\|/_\-\s]|^)(\d+(?:\.\d+)?)(?:[\|/_\-\s]|$)', name)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
        return np.nan

    # Pattern 1: ISO full date e.g. |2021-05-14 or /2021-05-14 or _2021-05-14
    m = re.search(r'(?:[\|/_\s]|^)(\d{4}-\d{2}-\d{2})(?:[\|/_\s]|$)', name)
    if m:
        d = parse_date_to_decimal(m.group(1))
        if not np.isnan(d):
            return d

    # Pattern 2: Decimal year e.g. |2021.35 or /2021.35
    m = re.search(r'(?:[\|/_\s]|^)(\d{4}\.\d{2,4})(?:[\|/_\s]|$)', name)
    if m:
        d = parse_date_to_decimal(m.group(1))
        if not np.isnan(d):
            return d

    # Pattern 3: Year-month e.g. |2021-05
    m = re.search(r'(?:[\|/_\s]|^)(\d{4}-\d{2})(?:[\|/_\s]|$)', name)
    if m:
        d = parse_date_to_decimal(m.group(1))
        if not np.isnan(d):
            return d

    # Pattern 4: Trailing year e.g. /2021 or |2021
    m = re.search(r'(?:[\|/_\s])(\d{4})$', name)
    if m:
        d = parse_date_to_decimal(m.group(1))
        if not np.isnan(d):
            return d

    return np.nan
