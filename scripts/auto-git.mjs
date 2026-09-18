#!/usr/bin/env node
/**
 * Commit whenever the working tree is dirty; push to origin every 5 minutes.
 * Respects .gitignore (no .env, MaleCNS feathers, outputs, individuals, venv).
 */
import { spawnSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const COMMIT_EVERY_MS = Number(process.env.AUTO_COMMIT_MS || 8_000);
const PUSH_EVERY_MS = Number(process.env.AUTO_PUSH_MS || 300_000);
const FORBIDDEN = /(\.env($|\.)|\.feather$|(^|\/)(\.venv|connectome-weights|syn-points|syn-partners|body-annotations|body-neurotransmitters))/i;

let busy = false;

function log(...args) {
  console.log(new Date().toISOString(), ...args);
}

function git(args) {
  return spawnSync("git", args, {
    cwd: ROOT,
    encoding: "utf8",
    env: { ...process.env, GIT_TERMINAL_PROMPT: "0" },
  });
}

function fail(result, action) {
  const err = (result.stderr || result.stdout || "").trim();
  log(`${action} failed (${result.status}):`, err || "(no output)");
  return false;
}

function stagedNames() {
  const result = git(["diff", "--cached", "--name-only"]);
  if (result.status !== 0) return [];
  return result.stdout.split("\n").map((line) => line.trim()).filter(Boolean);
}

function isDirty() {
  const result = git(["status", "--porcelain"]);
  if (result.status !== 0) return fail(result, "status");
  return result.stdout.trim().length > 0;
}

function aheadCount() {
  git(["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"]);
  const result = git(["rev-list", "--count", "@{u}..HEAD"]);
  if (result.status !== 0) {
    const local = git(["rev-parse", "--abbrev-ref", "HEAD"]);
    const branch = (local.stdout || "main").trim();
    const count = git(["rev-list", "--count", `origin/${branch}..HEAD`]);
    if (count.status !== 0) return 1;
    return Number(count.stdout.trim() || "0");
  }
  return Number(result.stdout.trim() || "0");
}

function commitIfDirty() {
  if (!isDirty()) return false;
  const add = git(["add", "-A"]);
  if (add.status !== 0) return fail(add, "add");

  const bad = stagedNames().filter((name) => FORBIDDEN.test(name));
  if (bad.length) {
    log("unstage forbidden files:", bad.join(", "));
    const reset = git(["reset", "HEAD", "--", ...bad]);
    if (reset.status !== 0) return fail(reset, "unstage");
  }

  const files = stagedNames();
  if (!files.length) {
    log("skip commit: nothing allowed to stage");
    return false;
  }

  const summary = files.length <= 5 ? files.join(", ") : `${files.length} files`;
  const message = `auto: ${summary}`;
  const commit = git(["commit", "-m", message]);
  if (commit.status !== 0) {
    if (/nothing to commit/i.test(`${commit.stdout}\n${commit.stderr}`)) return false;
    return fail(commit, "commit");
  }
  log("committed", message);
  return true;
}

function pushIfAhead() {
  const n = aheadCount();
  if (!n) {
    log("push skip: already on origin");
    return false;
  }
  const push = git(["push", "origin", "HEAD"]);
  if (push.status !== 0) return fail(push, "push");
  log(`pushed ${n} commit(s) to origin`);
  return true;
}

async function withLock(fn) {
  if (busy) return;
  busy = true;
  try {
    await fn();
  } finally {
    busy = false;
  }
}

log(`watching ${ROOT}`);
log(`commit if dirty every ${COMMIT_EVERY_MS / 1000}s; push every ${PUSH_EVERY_MS / 1000}s`);

await withLock(async () => {
  commitIfDirty();
  pushIfAhead();
});

setInterval(() => {
  withLock(async () => {
    commitIfDirty();
  });
}, COMMIT_EVERY_MS);

setInterval(() => {
  withLock(async () => {
    commitIfDirty();
    pushIfAhead();
  });
}, PUSH_EVERY_MS);
