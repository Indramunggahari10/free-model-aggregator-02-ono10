// free-model-aggregator-02: one server, two client wires, and a real free model behind every
// answer — the best free model the brief rates for each prompt, with failover when one refuses.
//
// WHAT A TRIAL GIVES YOU
//   A trial runs this server as `node /bot/main.ts` from your repository's root, with Node 22's
//   standard library alone (nothing is installed; TypeScript runs directly, and a relative import
//   needs its .ts extension). The env is at $MAKEFUN_ENV_URL; your model calls go through the
//   grader's gateway at $OPENAI_BASE_URL, an OpenAI-compatible endpoint that reaches OpenRouter on
//   the key you pushed to the competition. The brief at $MAKEFUN_PROMPT_FILE carries two documents:
//     - the free model list: each free OpenRouter model's gateway id
//       (`openrouter/<vendor>/<model>:free`), its published rating in milli-points, what it can do
//       (tools, vision, JSON mode, context length) and its free flag;
//     - the six prompts: each one's id, its api (openai or anthropic), whether it streams, its
//       weight (1 or 2) and the requirements a model must meet to serve it. Each prompt carries a
//       unique token the answer must contain.
//   The server:
//     1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//     2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//     3. serves GET /v1/models (answering in the shape of the auth header it got: x-api-key means
//        Anthropic, Authorization means OpenAI), POST /v1/chat/completions and POST /v1/messages,
//        each with a JSON body or with server-sent events;
//     4. answers each prompt by dialling ONE free model through the gateway. OPENAI_API_KEY in the
//        sandbox is a dummy the gateway swaps for the OpenRouter key you pushed; the real key never
//        enters the sandbox;
//     5. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}.
//        A process still running when the wall ends scores zero, so never wait for anything else.
//   `node main.ts --list` prints the same catalog JSON and exits 0 before any server starts; the
//   register body's catalog is that same document, byte for byte.
//
// ROUTING AND FAILOVER (quality 60%, failover 10%)
//   Candidates are ranked by the brief's rating_milli, best first, among the models that meet the
//   request's requirements (read off the request body itself: response_format json_object means
//   JSON mode, a non-empty tools array means tools, image blocks mean vision — so routing never
//   depends on arrival order, and a replay sees the same choice from the same request). The best
//   model serves first; a 429 or 5xx (or a timed-out call) fails over to the next model down the
//   ranking, up to three attempts a prompt, and a model that refused once is cooled down for the
//   rest of the trial — one rate-limited model never takes the later prompts down with it. Every
//   refusal a prompt recovered from is reported on the answer in the
//   `x-makefun-upstream-failures` header, the only way the failover column can see it. A 400 or
//   401 is not retried: no other model fixes a refused model id or a missing key.
//   Streaming answers are collected in full before the client sees a byte, so a refusal
//   part-way through a stream can still fail over cleanly, and the client always gets one
//   well-formed stream.
//
// THE MODEL RULE
//   Whatever this server reports in a response's `model` field must be the model it actually
//   dialled. The house records every gateway call and scores the model that really answered. This
//   server sends the brief's model id in the gateway request and puts that same id back on every
//   answer and every streamed chunk (the provider's own answer spells it without the gateway's
//   `openrouter/` prefix), so its claim is the call it made, never another. A prompt whose
//   reported model has no matching successful call in the house's record earns nothing.
//
// DETERMINISM
//   Every choice is a function of the brief, the requests, and the gateway's answers: no wall
//   clock and no unseeded randomness in anything sent. Timeouts only bound how long a call may
//   hang; the per-prompt attempt budget is counted in attempts, not seconds, and the model
//   listings carry fixed timestamps. A replay of the same trial therefore makes the same gateway
//   requests and leaves the same state export.
//
// FILES
//   main.ts            this server: the brief, the catalog, the routes, the register and the poll.
//   lib/providers.ts   the brief's types and parser, the free model list, the catalog, the ranking.
//   lib/wire.ts        Anthropic <-> Chat Completions translation, the model field, the SSE writers.
//   LICENSE            MIT.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  capabilitiesOf,
  freeModels,
  isRecord,
  meetsRequirements,
  rankedCandidates,
  readBrief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAIText,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIJsonToStreamText,
  openAIJsonWithModel,
  openAIModelList,
  openAIStreamTextWithModel,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** The most bytes of one upstream SSE body this server collects before relaying it. */
const MAX_UPSTREAM_BYTES = 4 * 1024 * 1024;
/** The most attempts one prompt gets across the ranked failover chain. */
const MAX_ATTEMPTS = 3;
/** One gateway call may hang this long before it counts as a refusal (ms). */
const DIAL_TIMEOUT_MS = 14000;
/** No new attempt starts once the prompt has been in flight this long (ms); the env caps a prompt at 30 s. */
const PROMPT_BUDGET_MS = 26000;
/** The output cap sent when a request names none: the gateway refuses a request with no cap. */
const DEFAULT_MAX_TOKENS = 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the entry's gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** Models that refused a gateway call this trial: skipped for later prompts unless no other candidate remains. */
const cooledDown = new Set<string>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on the
 * trial network); a base that does not already end in `/v1` gets one, so both spellings reach the route.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return freeModels(brief).map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

/** Requirements read off a request body: the body's own shape is the source, never arrival order. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** Requirements for an Anthropic-shaped body: tools, and image blocks in its messages. */
function requirementsFromAnthropicBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && block.type === 'image') vision = true;
    }
  }
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: false };
}

/**
 * The ranked failover chain for a request: the brief's rating order among the models that meet it,
 * with models that refused earlier in this trial moved to the back — unless that would leave no
 * candidate at all, in which case the full ranking stands (a cooled-down model is still better
 * than no model).
 */
function chainFor(requires: Requirement): ModelRow[] {
  const ranked = rankedCandidates(brief, requires);
  if (ranked.length === 0) return [];
  const fresh = ranked.filter((m) => !cooledDown.has(m.id));
  return fresh.length > 0 ? fresh : ranked;
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    // Node's http stream yields Buffers; a web stream yields Uint8Array views. Buffer.from
    // copies either one's bytes; String(chunk) would turn a view into comma-joined numbers.
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk as Uint8Array);
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** One upstream response body, collected whole, or null when it is too large to hold. */
async function readUpstreamText(res: Response): Promise<string | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  const body = res.body;
  if (body === null) return '';
  for await (const chunk of body) {
    // A web stream yields Uint8Array views, not Buffers: Buffer.from copies the bytes, while
    // String(chunk) would turn a view into comma-joined numbers and corrupt the answer.
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk as Uint8Array);
    size += buffer.length;
    if (size > MAX_UPSTREAM_BYTES) return null;
    chunks.push(buffer);
  }
  return Buffer.concat(chunks).toString('utf8');
}

/** The failure trail for the answer's header: at most 8 rows, as the rules spell them. */
function failuresHeader(failures: { status: number; code: string; model: string | null }[]): string {
  return JSON.stringify(failures.slice(0, 8));
}

function sendJson(res: ServerResponse, status: number, body: unknown, failures: { status: number; code: string; model: string | null }[] = []): void {
  const text = JSON.stringify(body);
  const headers: Record<string, string> = { 'content-type': 'application/json', 'content-length': String(Buffer.byteLength(text)) };
  if (failures.length > 0) headers['x-makefun-upstream-failures'] = failuresHeader(failures);
  res.writeHead(status, headers);
  res.end(text);
}

function sendSse(res: ServerResponse, text: string, failures: { status: number; code: string; model: string | null }[] = []): void {
  const headers: Record<string, string> = { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' };
  if (failures.length > 0) headers['x-makefun-upstream-failures'] = failuresHeader(failures);
  res.writeHead(200, headers);
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** One gateway call's outcome: a collected body on success, or the refusal that ended it. */
type DialOutcome =
  | { ok: true; model: string; status: number; text: string }
  | { ok: false; model: string; status: number; code: string };

/**
 * POST one Chat Completions body to the gateway, naming `model`, and collect the whole answer.
 * A non-2xx status, a body too large to hold, or a thrown fetch is a refusal, never an answer.
 */
async function dial(model: string, body: Json): Promise<DialOutcome> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  let upstream: Response;
  try {
    upstream = await fetch(chatCompletionsUrl(), {
      method: 'POST',
      headers,
      body: JSON.stringify({ ...body, model }),
      signal: AbortSignal.timeout(DIAL_TIMEOUT_MS),
    });
  } catch (err) {
    const timedOut = err instanceof Error && err.name === 'TimeoutError';
    return { ok: false, model, status: timedOut ? 504 : 502, code: timedOut ? 'gateway_timeout' : 'upstream_unreachable' };
  }
  if (!upstream.ok) {
    const text = await upstream.text().catch(() => '');
    let code = 'upstream_error';
    try {
      const parsed: unknown = JSON.parse(text);
      if (isRecord(parsed) && isRecord(parsed.error) && typeof parsed.error.code === 'string') code = parsed.error.code;
      else if (isRecord(parsed) && typeof parsed.code === 'string') code = parsed.code;
    } catch {
      // The refusal carries no JSON code: keep the generic one.
    }
    return { ok: false, model, status: upstream.status, code };
  }
  const text = await readUpstreamText(upstream);
  if (text === null) return { ok: false, model, status: 502, code: 'upstream_too_large' };
  return { ok: true, model, status: upstream.status, text };
}

/** Whether a refusal is worth another attempt: 429s and 5xx are transient; 400s and 401s are not. */
function retryable(status: number): boolean {
  return status === 429 || status >= 500;
}

/**
 * Dial a prompt down its failover chain. Returns the first successful call with the refusals it
 * took to get there, or the last refusal when every attempt failed. A model that refuses is
 * cooled down for the rest of the trial.
 */
async function dialWithFailover(chain: ModelRow[], body: Json): Promise<DialOutcome & { failures: { status: number; code: string; model: string | null }[] }> {
  const failures: { status: number; code: string; model: string | null }[] = [];
  const started = Date.now();
  let attempt = 0;
  for (const model of chain) {
    if (attempt >= MAX_ATTEMPTS) break;
    if (attempt > 0 && Date.now() - started > PROMPT_BUDGET_MS) break;
    attempt += 1;
    const outcome = await dial(model.id, body);
    if (outcome.ok) return { ...outcome, failures };
    failures.push({ status: outcome.status, code: outcome.code, model: outcome.model });
    cooledDown.add(outcome.model);
    if (!retryable(outcome.status)) break;
  }
  const last = failures[failures.length - 1] ?? { status: 502, code: 'no_free_model', model: null };
  return { ok: false, model: last.model ?? '', status: last.status, code: last.code, failures };
}

/** A refused prompt as a 502 in the client's wire, carrying the failure trail it took. */
function refused(
  res: ServerResponse,
  wire: 'openai' | 'anthropic',
  outcome: { status: number; code: string; failures: { status: number; code: string; model: string | null }[] },
): void {
  const message = `the gateway refused every free model that meets this request (last: HTTP ${outcome.status} ${outcome.code})`;
  if (wire === 'anthropic') {
    sendJson(res, 502, anthropicError('api_error', message), outcome.failures);
  } else {
    sendJson(res, 502, openAIError(message, 'upstream_error', null), outcome.failures);
  }
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that meets it. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const chain = chainFor(requirementsFromBody(body));
  if (chain.length === 0) return sendError(res, 'openai', 503, 'no_free_model', 'no free model in the brief meets this request');
  const upstreamBody: Json = { ...body, model: 'auto' };
  if (typeof upstreamBody.max_tokens !== 'number') upstreamBody.max_tokens = DEFAULT_MAX_TOKENS;
  if (upstreamBody.stream === true) {
    upstreamBody.stream_options = { ...(isRecord(upstreamBody.stream_options) ? upstreamBody.stream_options : {}), include_usage: true };
  }
  const outcome = await dialWithFailover(chain, upstreamBody);
  if (!outcome.ok) return refused(res, 'openai', outcome);
  let parsed: unknown = null;
  try {
    parsed = JSON.parse(outcome.text);
  } catch {
    // A stream body is not JSON: only a non-stream answer reaches this parse.
    if (body.stream !== true) return sendError(res, 'openai', 502, 'upstream_error', 'the gateway answered with a body that is not JSON');
  }
  if (body.stream === true) {
    if (outcome.text.includes('data:')) return sendSse(res, openAIStreamTextWithModel(outcome.text, outcome.model), outcome.failures);
    if (!isRecord(parsed)) return sendError(res, 'openai', 502, 'upstream_error', 'the gateway answered a stream request with a body that is neither SSE nor JSON');
    return sendSse(res, openAIJsonToStreamText(parsed, outcome.model), outcome.failures);
  }
  if (!isRecord(parsed) || !Array.isArray(parsed.choices)) {
    return sendError(res, 'openai', 502, 'upstream_error', 'the gateway answered with a body that holds no choices');
  }
  sendJson(res, 200, openAIJsonWithModel(parsed, outcome.model), outcome.failures);
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const chain = chainFor(requirementsFromAnthropicBody(body));
  if (chain.length === 0) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model in the brief meets this request');
  // The model id here is a placeholder: dialWithFailover names the model each attempt dials.
  const outcome = await dialWithFailover(chain, anthropicToOpenAI(body, 'auto'));
  if (!outcome.ok) return refused(res, 'anthropic', outcome);
  let parsed: unknown = null;
  try {
    parsed = JSON.parse(outcome.text);
  } catch {
    if (body.stream !== true) return sendError(res, 'anthropic', 502, 'api_error', 'the gateway answered with a body that is not JSON');
  }
  if (body.stream === true) {
    if (outcome.text.includes('data:')) return sendSse(res, anthropicEventsFromOpenAIText(outcome.text, outcome.model), outcome.failures);
    if (!isRecord(parsed)) return sendError(res, 'anthropic', 502, 'api_error', 'the gateway answered a stream request with a body that is neither SSE nor JSON');
    return sendSse(res, anthropicEventsFromOpenAIText(openAIJsonToStreamText(parsed, outcome.model), outcome.model), outcome.failures);
  }
  if (!isRecord(parsed) || !Array.isArray(parsed.choices)) {
    return sendError(res, 'anthropic', 502, 'api_error', 'the gateway answered with a body that holds no choices');
  }
  sendJson(res, 200, openAIToAnthropic(parsed, outcome.model), outcome.failures);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
