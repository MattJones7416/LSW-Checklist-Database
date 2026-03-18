const ITEM_TYPES = new Set(["set", "minifig", "piece"]);

function jsonResponse(payload, status = 200) {
  return new Response(JSON.stringify(payload), {
    status,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type",
      "Cache-Control": "no-store",
    },
  });
}

function normalizeText(value) {
  return String(value ?? "").trim();
}

function sanitizeNumber(value) {
  const number = normalizeText(value);
  if (!number || number.length > 120) return "";
  if (!/^[A-Za-z0-9 ._\/-]+$/.test(number)) return "";
  return number;
}

function sanitizeRequestID(value) {
  const requestID = normalizeText(value).toLowerCase();
  if (!requestID) return crypto.randomUUID();
  if (!/^[a-z0-9-]{8,64}$/.test(requestID)) return crypto.randomUUID();
  return requestID;
}

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") {
      return jsonResponse({ ok: true }, 204);
    }

    if (request.method === "GET") {
      return jsonResponse({ ok: true, service: "market-refresh-proxy" });
    }

    if (request.method !== "POST") {
      return jsonResponse({ ok: false, error: "method_not_allowed" }, 405);
    }

    let payload;
    try {
      payload = await request.json();
    } catch {
      return jsonResponse({ ok: false, error: "invalid_json" }, 400);
    }

    const itemType = normalizeText(payload.item_type).toLowerCase();
    const number = sanitizeNumber(payload.number);
    const requestID = sanitizeRequestID(payload.request_id);

    if (!ITEM_TYPES.has(itemType)) {
      return jsonResponse({ ok: false, error: "invalid_item_type" }, 400);
    }
    if (!number) {
      return jsonResponse({ ok: false, error: "invalid_number" }, 400);
    }

    const owner = normalizeText(env.GITHUB_OWNER);
    const repo = normalizeText(env.GITHUB_REPO);
    const token = normalizeText(env.GITHUB_TOKEN);
    const workflow = normalizeText(env.GITHUB_WORKFLOW_FILE) || "refresh-single-market-item.yml";
    const ref = normalizeText(env.GITHUB_REF) || "main";

    if (!owner || !repo || !token) {
      return jsonResponse({ ok: false, error: "server_not_configured" }, 500);
    }

    const githubResponse = await fetch(
      `https://api.github.com/repos/${owner}/${repo}/actions/workflows/${workflow}/dispatches`,
      {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${token}`,
          "Accept": "application/vnd.github+json",
          "Content-Type": "application/json",
          "X-GitHub-Api-Version": "2022-11-28",
          "User-Agent": "lsw-market-refresh-proxy",
        },
        body: JSON.stringify({
          ref,
          inputs: {
            item_type: itemType,
            number,
            request_id: requestID,
          },
        }),
      }
    );

    if (!githubResponse.ok) {
      const body = (await githubResponse.text()).slice(0, 800);
      return jsonResponse({
        ok: false,
        error: "github_dispatch_failed",
        status: githubResponse.status,
        request_id: requestID,
        body,
      }, githubResponse.status);
    }

    return jsonResponse({
      ok: true,
      state: "queued",
      request_id: requestID,
    }, 202);
  },
};
