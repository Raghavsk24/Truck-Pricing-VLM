"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { Contribution } from "@/lib/types";
import { formatUsd, formatUsdSigned } from "@/lib/format";

const COLORS: Record<string, string> = {
  type: "#5b9bd5",
  brand: "#d4892a",
  era: "#2dd4bf",
  chassis_and_frame: "#a78bfa",
  front_end_engine_compartment_hood: "#fb7185",
  tires_wheels_suspension: "#38bdf8",
  cab_sleeper_aero_fairings: "#f472b6",
  condition: "#94a3b8",
};

type Props = {
  contributions: Contribution[];
};

export function ContributionChart({ contributions }: Props) {
  const row: Record<string, string | number> = { name: "Asking price" };
  for (const c of contributions) {
    row[c.key] = c.usd;
  }

  return (
    <div>
      <div className="h-[180px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={[row]}
            layout="vertical"
            margin={{ top: 12, right: 24, left: 8, bottom: 8 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
            <XAxis
              type="number"
              tickFormatter={(v: number) => formatUsd(v)}
              stroke="#64748b"
              tick={{ fill: "#94a3b8", fontSize: 11 }}
            />
            <YAxis
              type="category"
              dataKey="name"
              width={96}
              stroke="#64748b"
              tick={{ fill: "#cbd5e1", fontSize: 12 }}
            />
            <Tooltip
              cursor={{ fill: "rgba(148,163,184,0.08)" }}
              contentStyle={{
                background: "#0f172a",
                border: "1px solid #334155",
                borderRadius: 8,
                color: "#e2e8f0",
              }}
              formatter={(value, name) => {
                const n = typeof value === "number" ? value : Number(value);
                const label =
                  contributions.find((c) => c.key === name)?.label ?? String(name);
                return [formatUsdSigned(n), label];
              }}
            />
            <Legend
              wrapperStyle={{ color: "#cbd5e1", fontSize: 12 }}
              formatter={(value) =>
                contributions.find((c) => c.key === value)?.label ?? String(value)
              }
            />
            {contributions.map((c) => (
              <Bar
                key={c.key}
                dataKey={c.key}
                stackId="price"
                fill={COLORS[c.key] ?? "#64748b"}
                maxBarSize={42}
              />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
      <ul className="mt-4 grid gap-2 sm:grid-cols-2">
        {contributions.map((c) => (
          <li
            key={c.key}
            className="flex items-center justify-between rounded-lg border border-slate-800 bg-slate-950/50 px-3 py-2 text-sm"
          >
            <span className="flex items-center gap-2 text-slate-300">
              <span
                className="h-2.5 w-2.5 rounded-sm"
                style={{ background: COLORS[c.key] ?? "#64748b" }}
              />
              {c.label}
            </span>
            <span className={c.usd < 0 ? "text-rose-400" : "text-slate-100"}>
              {formatUsdSigned(c.usd)}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
