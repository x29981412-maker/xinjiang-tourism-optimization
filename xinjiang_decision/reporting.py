"""Portable SVG figures and UTF-8 reports; no browser or plotting backend needed."""

from html import escape


NAVY, TEAL, GRAY, LIGHT = "#172c46", "#087f8c", "#8290a3", "#f2f5f9"


def canvas(title, subtitle, height=360):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" role="img" width="1040" height="{height}" viewBox="0 0 1040 {height}">',
            f'<title>{escape(title)}</title><desc>{escape(subtitle)}</desc>',
            f'<rect width="1040" height="{height}" rx="18" fill="{LIGHT}"/>',
            '<g font-family="Arial,Microsoft YaHei,sans-serif">',
            f'<text x="36" y="43" font-size="23" font-weight="700" fill="{NAVY}">{escape(title)}</text>',
            f'<text x="36" y="70" font-size="13" fill="#536378">{escape(subtitle)}</text>']


def portfolio_figure(baseline, optimized, gain):
    svg = canvas("Route portfolio | 63 candidates, 18 selected", "Fixed product quotas, regional coverage, capacity floor and quality floor")
    metrics = [("Model score", baseline["score"], optimized["score"], ".3f"),
               ("12-day capacity (model persons)", baseline["capacity"], optimized["capacity"], ",.0f"),
               ("Quality-proxy routes", baseline["quality_pass_count"], optimized["quality_pass_count"], ".0f")]
    for panel, (label, old, new, fmt) in enumerate(metrics):
        x = 36 + panel * 337
        svg.append(f'<text x="{x}" y="112" font-size="15" font-weight="700" fill="{NAVY}">{label}</text>')
        for i, (value, color, name) in enumerate([(old, GRAY, "Baseline"), (new, TEAL, "Optimized")]):
            y = 145 + i * 64
            width = 220 * value / max(old, new)
            svg.extend([f'<text x="{x}" y="{y}" font-size="12" fill="#536378">{name}</text>',
                        f'<rect x="{x}" y="{y + 9}" width="{width:.2f}" height="22" rx="4" fill="{color}"/>',
                        f'<text x="{x + width + 8:.2f}" y="{y + 25}" font-size="12" fill="{NAVY}">{format(value,fmt)}</text>'])
    svg.extend([f'<text x="36" y="310" font-size="18" font-weight="700" fill="{TEAL}">Score improvement: {gain:.4%} | MILP gap: {optimized["mip_gap"]:.0%}</text>',
                '<text x="36" y="337" font-size="12" fill="#536378">The optimum applies only to the fixed candidate set and stated policy, not all possible travel routes.</text>', '</g></svg>'])
    return "\n".join(svg)


def schedule_figure(schedule):
    svg = canvas("Three-team cultural fieldwork | exact partition", "14 sites; two approval sites assigned to the designated team; all tours start and end in Urumqi")
    max_time = schedule["max_completion_hours"]
    for i, group in enumerate(schedule["groups"]):
        y = 122 + i * 64
        width = 660 * group["completion_hours"] / max_time
        svg.extend([f'<text x="36" y="{y+19}" font-size="14" fill="{NAVY}">Team {i+1} | {len(group["site_ids"])} sites</text>',
                    f'<rect x="190" y="{y}" width="{width:.2f}" height="28" rx="5" fill="{TEAL if i == 0 else GRAY}"/>',
                    f'<text x="{200+width:.2f}" y="{y+19}" font-size="14" fill="{NAVY}">{group["completion_hours"]:.3f} h</text>'])
    svg.extend([f'<text x="36" y="325" font-size="13" fill="#536378">Makespan {max_time:.3f} h | Imbalance {schedule["balance_gap_hours"]:.3f} h | {schedule["checked_canonical_partitions"]:,} canonical partitions checked</text>', '</g></svg>'])
    return "\n".join(svg)


def markdown_table(rows, columns):
    labels = [label for _, label in columns]
    lines = ["| " + " | ".join(labels) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        values = []
        for key, _ in columns:
            v = row.get(key)
            values.append("—" if v is None else (f"{v:.4f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v)))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)
