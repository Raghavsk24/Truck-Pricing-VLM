# Kamion web app

Photo in, asking range out. Next.js app for Vercel with a Supabase store and a server-side Claude call.

## Setup

```bash
cd web
cp .env.example .env.local
npm install
```

Fill `.env.local`:

- `ANTHROPIC_API_KEY` — never expose this to the browser
- `ANTHROPIC_MODEL` — defaults to `claude-sonnet-4-5-20250929` (Claude Sonnet 4.5 / 5 family)
- `NEXT_PUBLIC_SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`

In the Supabase SQL editor, run [`supabase/schema.sql`](supabase/schema.sql) to create the `analyses` table and the private `truck-uploads` bucket.

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Deploy

Create a Vercel project with **Root Directory** `web`, then set the same env vars. Point DNS as usual.
