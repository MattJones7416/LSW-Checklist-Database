# Market Refresh Proxy

GitHub Actions cannot be dispatched anonymously. This Cloudflare Worker provides a public POST endpoint for the app and keeps the GitHub token server-side.

## Request

`POST /`

```json
{
  "item_type": "set",
  "number": "75355-1",
  "request_id": "uuid"
}
```

## Required secrets / vars

- `GITHUB_OWNER`
- `GITHUB_REPO`
- `GITHUB_TOKEN`
- optional: `GITHUB_WORKFLOW_FILE` (defaults to `refresh-single-market-item.yml`)
- optional: `GITHUB_REF` (defaults to `main`)

## Deploy

```bash
cd workers/market-refresh-proxy
cp .dev.vars.example .dev.vars
# fill in values
npx wrangler deploy
```

After deploy, write the public URL into `dist/client-config.json` as `marketRefreshDispatchURL` and rebuild sync artifacts with `MARKET_REFRESH_DISPATCH_URL` set.
