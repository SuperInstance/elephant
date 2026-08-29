#!/usr/bin/env python3
"""JEPA monitor — kappa/loss curves and dial state dashboard from logged field data.

Loads roomd-field-log.jsonl (or other data sources) and generates an interactive
HTML dashboard with:
  - Kappa concentration curve over time (SVG line chart)
  - Current dial state gauges (mood, volume, earnestness, cynicism, joke_landing, panic, presence)
  - Predictions-vs-actuals table

Usage: python tools/jepa-monitor.py
  Regenerate: python tools/jepa-monitor.py

Output: tools/jepa-dashboard.html (open in browser)
"""
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path


def load_field_log():
    """Load roomd-field-log.jsonl or return None if not found."""
    # Try multiple possible paths
    possible_paths = [
        Path("data/roomd-field-log.jsonl"),
        Path(__file__).parent.parent / "data" / "roomd-field-log.jsonl",
    ]

    path = None
    for p in possible_paths:
        if p.exists():
            path = p
            break

    if not path:
        return None

    entries = []
    bad_count = 0
    try:
        with open(path) as f:
            for line in f:
                if line.strip():
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        bad_count += 1
                        # Skip malformed lines but keep reading
                        continue
    except IOError:
        return None

    if bad_count > 0:
        print(f"⚠  Skipped {bad_count} malformed JSON lines")

    return entries if entries else None


def extract_timeseries(entries):
    """Extract (timestamp, kappa, room) tuples from field log.

    Returns list of dicts: {ts, kappa_avg, kappas_by_room, dials_by_room}
    """
    timeseries = []
    for entry in entries:
        ts = entry.get("ts", 0.0)
        rooms = entry.get("rooms", {})

        kappas = []
        dials_all = {}

        for room_name, room_data in rooms.items():
            kappa = room_data.get("kappa", 0.0)
            kappas.append(kappa)
            dials = room_data.get("dials", {})
            dials_all[room_name] = dials

        kappa_avg = sum(kappas) / len(kappas) if kappas else 0.0
        timeseries.append({
            "ts": ts,
            "kappa_avg": kappa_avg,
            "kappas_by_room": dict(zip(rooms.keys(), kappas)),
            "dials_by_room": dials_all,
        })

    return timeseries


def generate_synthetic_data():
    """Generate clearly-marked synthetic demo data."""
    now = datetime.now().timestamp()
    entries = []

    # Generate 24 synthetic entries with realistic-looking curves
    for i in range(24):
        ts = now - (23 - i) * 3600  # hourly, last 24h

        # Synthetic kappa curve: oscillates between 1.5 and 4.0
        base_kappa = 2.5 + 1.2 * math.sin(i / 6.0)

        rooms = {
            "demo-room-A": {
                "room": "demo-room-A",
                "warmth": 0.1 + 0.3 * math.sin(i / 8.0),
                "kappa": base_kappa + 0.3 * (i % 3),
                "dials": {
                    "mood": 0.5 + 0.3 * math.cos(i / 7.0),
                    "volume": 0.3 + 0.4 * math.sin(i / 5.0),
                    "earnestness": 0.6 + 0.2 * math.sin(i / 9.0),
                    "cynicism": 0.2 + 0.15 * math.cos(i / 6.0),
                    "joke_landing": 0.4 + 0.3 * math.sin(i / 8.0),
                    "panic": 0.1 + 0.2 * math.cos(i / 10.0),
                    "presence": 0.5 + 0.3 * math.sin(i / 4.0),
                },
                "messages": 10 + i,
                "ts": ts,
            },
            "demo-room-B": {
                "room": "demo-room-B",
                "warmth": -0.2 + 0.25 * math.sin(i / 7.0),
                "kappa": base_kappa * 1.1 - 0.2 * (i % 4),
                "dials": {
                    "mood": -0.3 + 0.4 * math.sin(i / 6.0),
                    "volume": 0.7 + 0.2 * math.cos(i / 8.0),
                    "earnestness": 0.5 + 0.25 * math.cos(i / 7.0),
                    "cynicism": 0.3 + 0.2 * math.sin(i / 9.0),
                    "joke_landing": 0.1 + 0.25 * math.cos(i / 5.0),
                    "panic": 0.5 + 0.3 * math.sin(i / 8.0),
                    "presence": 0.4 + 0.35 * math.cos(i / 6.0),
                },
                "messages": 20 + i * 2,
                "ts": ts,
            },
        }

        entries.append({"ts": ts, "rooms": rooms})

    return entries


def svg_line_chart(timeseries, width=800, height=300, label="Kappa"):
    """Generate SVG line chart for kappa over time.

    Returns SVG string with time series plotted.
    """
    if not timeseries:
        return '<svg></svg>'

    kappas = [entry["kappa_avg"] for entry in timeseries]
    if not kappas:
        return '<svg></svg>'

    # Find min/max for scaling
    min_kappa = min(kappas)
    max_kappa = max(kappas)
    kappa_range = max_kappa - min_kappa or 1.0
    padding = 60

    # Scale functions
    def scale_x(i):
        return padding + (i / (len(kappas) - 1)) * (width - 2 * padding) if len(kappas) > 1 else padding

    def scale_y(k):
        return height - padding - ((k - min_kappa) / kappa_range) * (height - 2 * padding)

    # Build path
    points = []
    for i, k in enumerate(kappas):
        points.append((scale_x(i), scale_y(k)))

    path_d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in points)

    # Create SVG
    svg = f'''<svg width="{width}" height="{height}" style="border: 1px solid #e0e0e0; border-radius: 4px;">
    <defs>
        <linearGradient id="gradient-kappa" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" style="stop-color: #1e90ff; stop-opacity: 0.3" />
            <stop offset="100%" style="stop-color: #1e90ff; stop-opacity: 0" />
        </linearGradient>
    </defs>

    <!-- Grid lines -->
    <line x1="{padding}" y1="{height - padding}" x2="{width - padding}" y2="{height - padding}"
          stroke="#e0e0e0" stroke-width="1" />

    <!-- Y-axis -->
    <line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height - padding}"
          stroke="#999" stroke-width="1" />

    <!-- Chart area background -->
    <rect x="{padding}" y="{padding}" width="{width - 2*padding}" height="{height - 2*padding}"
          fill="#fafafa" />

    <!-- Grid lines (horizontal) -->
    <line x1="{padding}" y1="{height - padding - (height - 2*padding) * 0.25}" x2="{width - padding}"
          y2="{height - padding - (height - 2*padding) * 0.25}" stroke="#e0e0e0" stroke-width="0.5" stroke-dasharray="2,2" />
    <line x1="{padding}" y1="{height - padding - (height - 2*padding) * 0.5}" x2="{width - padding}"
          y2="{height - padding - (height - 2*padding) * 0.5}" stroke="#e0e0e0" stroke-width="0.5" stroke-dasharray="2,2" />
    <line x1="{padding}" y1="{height - padding - (height - 2*padding) * 0.75}" x2="{width - padding}"
          y2="{height - padding - (height - 2*padding) * 0.75}" stroke="#e0e0e0" stroke-width="0.5" stroke-dasharray="2,2" />

    <!-- Y-axis labels -->
    <text x="{padding - 10}" y="{height - padding + 5}" text-anchor="end" font-size="12" fill="#666">
        {min_kappa:.2f}
    </text>
    <text x="{padding - 10}" y="{height - padding - (height - 2*padding) * 0.5 + 5}" text-anchor="end" font-size="12" fill="#666">
        {(min_kappa + max_kappa) / 2:.2f}
    </text>
    <text x="{padding - 10}" y="{padding + 5}" text-anchor="end" font-size="12" fill="#666">
        {max_kappa:.2f}
    </text>

    <!-- Path and fill -->
    <path d="{path_d}" stroke="#1e90ff" stroke-width="2" fill="none" />
    <circle cx="{points[-1][0]:.1f}" cy="{points[-1][1]:.1f}" r="4" fill="#1e90ff" />

    <!-- X-axis -->
    <line x1="{padding}" y1="{height - padding}" x2="{width - padding}" y2="{height - padding}"
          stroke="#999" stroke-width="1" />

    <!-- Title -->
    <text x="{width / 2}" y="25" text-anchor="middle" font-size="16" font-weight="bold" fill="#333">
        {label} Over Time
    </text>
</svg>'''

    return svg


def svg_gauge(value, name, min_val=0.0, max_val=1.0, width=120, height=120):
    """Generate SVG circular gauge for a dial value.

    Returns SVG string with gauge representation.
    """
    # Clamp value to range
    v = max(min_val, min(max_val, value))
    ratio = (v - min_val) / (max_val - min_val) if max_val > min_val else 0.5

    # Determine color based on value
    if ratio < 0.33:
        color = "#d32f2f"  # red
    elif ratio < 0.67:
        color = "#f57c00"  # orange
    else:
        color = "#388e3c"  # green

    center_x, center_y = width / 2, height / 2
    radius = 40
    arc_start = 225  # degrees
    arc_end = 225 + (ratio * 270)  # 0-270 degree arc

    # Convert to radians for calculation
    start_rad = math.radians(arc_start)
    end_rad = math.radians(arc_end)

    x1 = center_x + radius * math.cos(start_rad)
    y1 = center_y + radius * math.sin(start_rad)
    x2 = center_x + radius * math.cos(end_rad)
    y2 = center_y + radius * math.sin(end_rad)

    large_arc = 1 if (arc_end - arc_start) > 180 else 0

    svg = f'''<svg width="{width}" height="{height}" style="display: inline-block; margin: 10px;">
    <!-- Background circle -->
    <circle cx="{center_x}" cy="{center_y}" r="{radius}" fill="none" stroke="#e0e0e0" stroke-width="8" />

    <!-- Arc -->
    <path d="M {x1} {y1} A {radius} {radius} 0 {large_arc} 1 {x2} {y2}"
          fill="none" stroke="{color}" stroke-width="8" stroke-linecap="round" />

    <!-- Center circle -->
    <circle cx="{center_x}" cy="{center_y}" r="8" fill="#333" />

    <!-- Value label -->
    <text x="{center_x}" y="{center_y + 35}" text-anchor="middle" font-size="14" font-weight="bold" fill="#333">
        {value:.2f}
    </text>

    <!-- Name label -->
    <text x="{center_x}" y="{height - 5}" text-anchor="middle" font-size="11" fill="#666">
        {name}
    </text>
</svg>'''

    return svg


def generate_dashboard_html(timeseries, synthetic=False):
    """Generate the complete HTML dashboard."""
    if not timeseries:
        timeseries = []

    # Get current state (last entry)
    current = timeseries[-1] if timeseries else {}
    dials_by_room = current.get("dials_by_room", {})

    # Calculate average dial values
    dial_names = ["mood", "volume", "earnestness", "cynicism", "joke_landing", "panic", "presence"]
    dial_values = {}

    for dial_name in dial_names:
        values = []
        for room_dials in dials_by_room.values():
            if dial_name in room_dials:
                values.append(room_dials[dial_name])
        dial_values[dial_name] = sum(values) / len(values) if values else 0.5

    # Generate gauge SVGs
    gauge_svgs = {}
    for dial_name in dial_names:
        val = dial_values[dial_name]
        gauge_svgs[dial_name] = svg_gauge(val, dial_name.replace("_", " ").title(),
                                          min_val=-1.0 if dial_name == "mood" else -1.0 if dial_name == "joke_landing" else 0.0,
                                          max_val=1.0)

    # Generate kappa chart
    kappa_chart = svg_line_chart(timeseries, label="Kappa Concentration")

    # Build table rows for predictions vs actuals
    table_rows = ""
    for i, entry in enumerate(timeseries[-10:]):  # Last 10 entries
        ts = entry.get("ts", 0)
        dt = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        kappa_avg = entry.get("kappa_avg", 0)

        # Predicted (using persistence: assume it stays the same as previous)
        if i > 0:
            predicted = timeseries[-10 + i - 1].get("kappa_avg", 0)
        else:
            predicted = kappa_avg

        actual = kappa_avg
        error = abs(actual - predicted)

        table_rows += f'''
        <tr>
            <td>{dt}</td>
            <td>{predicted:.3f}</td>
            <td>{actual:.3f}</td>
            <td>{error:.3f}</td>
        </tr>
        '''

    # Build the HTML
    synthetic_note = "<div style='background: #fff3cd; border: 1px solid #ffc107; border-radius: 4px; padding: 12px; margin-bottom: 20px;'><strong>⚠ Demo data (synthetic)</strong> — no real field log found. This is clearly-marked demo data for visualization testing.</div>" if synthetic else ""

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>JEPA Monitor — Field Dynamics Dashboard</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
            background: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}

        header {{
            background: #fff;
            border-bottom: 1px solid #e0e0e0;
            padding: 20px 0;
            margin-bottom: 30px;
        }}

        h1 {{
            font-size: 28px;
            font-weight: 600;
            color: #1a1a1a;
            margin-bottom: 5px;
        }}

        .subtitle {{
            font-size: 14px;
            color: #999;
        }}

        .section {{
            background: #fff;
            border-radius: 6px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid #e0e0e0;
        }}

        .section h2 {{
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #f0f0f0;
        }}

        .gauge-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            justify-content: center;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}

        thead {{
            background: #f5f5f5;
        }}

        th {{
            padding: 12px;
            text-align: left;
            font-weight: 600;
            border-bottom: 2px solid #e0e0e0;
        }}

        td {{
            padding: 12px;
            border-bottom: 1px solid #f0f0f0;
        }}

        tr:hover {{
            background: #fafafa;
        }}

        .footer {{
            text-align: center;
            color: #999;
            font-size: 12px;
            margin-top: 30px;
        }}
    </style>
</head>
<body>
    <header>
        <div class="container">
            <h1>🐘 JEPA Monitor</h1>
            <p class="subtitle">Field dynamics dashboard — kappa concentration, dial states, predictions</p>
        </div>
    </header>

    <div class="container">
        {synthetic_note}

        <!-- Kappa Curve -->
        <div class="section">
            <h2>Kappa Over Time</h2>
            {kappa_chart}
            <p style="margin-top: 15px; font-size: 13px; color: #666;">
                Kappa (κ) measures the concentration of the room's field.
                <strong>High κ</strong> = cold room (one way to be),
                <strong>low κ</strong> = warm room (many ways to be).
            </p>
        </div>

        <!-- Dial States -->
        <div class="section">
            <h2>Current Dial States</h2>
            <div class="gauge-container">
                {gauge_svgs['mood']}
                {gauge_svgs['volume']}
                {gauge_svgs['earnestness']}
                {gauge_svgs['cynicism']}
                {gauge_svgs['joke_landing']}
                {gauge_svgs['panic']}
                {gauge_svgs['presence']}
            </div>
        </div>

        <!-- Predictions vs Actuals -->
        <div class="section">
            <h2>Predictions vs Actuals (Last 10)</h2>
            <table>
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Predicted κ</th>
                        <th>Actual κ</th>
                        <th>Error</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        </div>

        <div class="footer">
            Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            | Regenerate: <code>python tools/jepa-monitor.py</code>
        </div>
    </div>
</body>
</html>'''

    return html


def main():
    """Load data, generate dashboard HTML."""
    # Try to load real data
    entries = load_field_log()
    synthetic = False

    if not entries:
        print("⚠  No field log found (data/roomd-field-log.jsonl), generating synthetic demo data...")
        entries = generate_synthetic_data()
        synthetic = True

    print(f"📊 Loaded {len(entries)} entries")

    # Extract timeseries
    timeseries = extract_timeseries(entries)

    if not timeseries:
        print("❌ No valid timeseries data")
        sys.exit(1)

    # Generate HTML
    html = generate_dashboard_html(timeseries, synthetic=synthetic)

    # Write output
    output_path = Path("tools/jepa-dashboard.html")
    output_path.write_text(html)

    print(f"✅ Dashboard generated: {output_path}")
    print(f"   Open in browser: file://{output_path.resolve()}")


if __name__ == "__main__":
    main()
