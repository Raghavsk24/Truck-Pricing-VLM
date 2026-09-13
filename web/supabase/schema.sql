-- Kamion pricing app schema. Run once in the Supabase SQL editor.

create table if not exists public.analyses (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  status text not null check (status in ('rejected', 'needs_brand', 'ok')),
  image_path text,
  vlm_output jsonb,
  truck_type text,
  brand text,
  era text,
  overall_score integer,
  center numeric,
  low numeric,
  high numeric,
  contributions jsonb,
  user_message text
);

alter table public.analyses enable row level security;

insert into storage.buckets (id, name, public)
values ('truck-uploads', 'truck-uploads', false)
on conflict (id) do nothing;
