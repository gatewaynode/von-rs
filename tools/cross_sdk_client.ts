// Protocol-parity check, client side: drives a running `von serve` with the unmodified JS SDK
// (bug-fix-fork-von/js/src). Run by cross_sdk_check.sh:
//   bun run cross_sdk_client.ts BASE_URL API_KEY SDK_DIR
// Env: CROSS_SDK_TIMEOUT, seconds per request (default 300, as in cross_sdk_client.py).
const [base, apiKey, sdkDir] = process.argv.slice(2);
const { VonClient, VonError, choice, noul, score } = await import(`${sdkDir}/src/index.ts`);

function check(ok: boolean, what: string, detail: unknown) {
  if (!ok) {
    console.error(`FAILED: ${what}: ${JSON.stringify(detail)}`);
    process.exit(1);
  }
}

const timeout = Number(process.env.CROSS_SDK_TIMEOUT ?? "300") * 1000;
const client = new VonClient({ baseURL: base, apiKey, timeout });

const resp = await client.systemOne({
  state: "The user clicked the checkout button but received a credit card decline error.",
  questions: {
    error_type: choice("What type of error occurred?", {
      payment_error: "Payment or card transaction failure",
      ui_bug: "Layout or display bug",
    }),
    is_payment: noul("Is this a payment failure?"),
    severity: score("How severe is this?", ["Cosmetic", "Degraded", "Blocking"]),
  },
});
check(resp.model === "von-1.2.0", "model id", resp);
check(resp.answers.error_type.choice === "payment_error", "choice", resp);
check(resp.answers.is_payment.noul > 0.5, "noul", resp);
check(Object.keys(resp.answers.severity.legend).length === 3, "score legend", resp);

const decided = await client.decide("Refund my duplicate charge", ["refund", "bug_report"]);
check(decided.choice === "refund", "decide helper", decided);
const p = await client.judge("Prod database is down, all requests failing", "Is there an outage?");
check(p > 0.5, "judge helper", p);
const rated = await client.rate("Minor typo on the pricing page", ["Low", "Medium", "High"]);
check(typeof rated.score === "number" && rated.score < 1.0, "rate helper", rated);

let status = 0;
try {
  await new VonClient({ baseURL: base, apiKey: "wrong" }).decide("x", ["a", "b"]);
} catch (e) {
  status = e instanceof VonError ? e.status : -1;
}
check(status === 401, "401 surfaces as VonError", status);

console.log("js SDK: systemOne + decide/judge/rate + 401 ok");
