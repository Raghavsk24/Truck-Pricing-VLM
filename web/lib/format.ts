export function formatUsd(n: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(n);
}

export function formatUsdSigned(n: number): string {
  const abs = formatUsd(Math.abs(n));
  if (n > 0) return `+${abs}`;
  if (n < 0) return `−${abs}`;
  return abs;
}

export function typeLabel(type: string): string {
  if (type === "day_cab" || type === "day-cab-truck") return "Day cab";
  if (type === "sleeper" || type === "sleeper-truck") return "Sleeper";
  if (type === "dump" || type === "dump-truck") return "Dump";
  return type.replace(/_/g, " ");
}

export function eraLabel(era: string): string {
  const map: Record<string, string> = {
    pre_2010: "Pre-2010",
    "2010_2015": "2010–2015",
    "2016_2020": "2016–2020",
    "2021_plus": "2021+",
    unknown: "Era unknown",
  };
  return map[era] ?? era;
}
