/**
 * Exact-match required-prefix blacklist.
 *
 * A candidate next prefix (last N letters of the accepted word) is rejected
 * only when it equals one of these strings, OR when it is any double
 * consonant (mm, tt, ff, …) via isBlockedPrefix. Longer prefixes that merely
 * contain / start with / end with a listed string are fine
 * (e.g. "ofa" blocked ≠ "ofat" blocked).
 *
 * resolvePrefixLen skips blocked lengths and shortens (ck → k, mm → m).
 */

/** Never allowed as the required next prefix at any chain length. */
const ALL = [
  // A
  'ah',
  'aj',
  'ak',
  'anda',
  'ande',
  'ay',
  'ady',

  // B
  'bb',
  'bd',
  'baa',
  'bk',
  'bs',
  'bsf',
  'bsh',
  'bskt',

  // C
  'cc',
  'cd',
  'ct',
  'cz',
  'caa',
  'ck',
  'cp',
  'cs',
  'cg',
  'cf',

  // D
  'dd',
  'dl',
  'db',
  'daa',
  'dy',
  'dc',
  'ddt',
  'dib',
  'dj',
  'dk',
  'dn',
  'ds',
  'dsr',
  'dt',

  // E
  'ery',
  'erl',
  'ek',
  'edh',
  'edhs',

  // F
  'ff',
  'fy',
  'faa',
  'fj',
  'ft',
  'fth',
  'fthm',
  'ftn',
  'ftnc',
  'ftne',

  // G
  'gg',
  'gn',
  'gt',
  'gs',
  'gp',
  'gv',

  // H
  'hh',
  'hn',
  'haa',
  'hox',
  'hp',
  'ht',
  'hs',
  'hr',

  // I
  'ia',
  'iap',
  'ie',
  'if',
  'ih',
  'ij',
  'ini',
  'io',
  'iq',
  'ip',
  'irg',
  'irk',
  'iu',
  'ive',
  'iw',
  'ix',
  'iz',
  'ish',

  // J
  'jb',
  'jf',
  'jg',
  'jj',
  'jl',
  'jm',
  'jn',
  'jq',
  'jt',
  'jx',
  'jr',
  'js',

  // K
  'kk',
  'kr',
  'ks',
  'kv',
  'kaa',
  'kh',

  // L
  'ls',
  'lf',
  'laa',
  'lb',
  'lc',
  'ld',
  'lm',
  'ln',
  'lp',
  'lr',
  'lt',

  // M
  'mm',
  'mn',
  'mk',
  'ml',
  'mp',
  'mr',
  'ms',
  'mt',

  // N
  'nd',
  'ng',
  'nn',
  'nr',
  'ns',
  'nt',
  'naa',
  'naw',
  'nb',
  'nc',
  'nj',
  'np',
  'nv',

  // O
  'oj',
  'oy',
  'ofa',
  'ofe',
  'onk',

  // P
  'pp',
  'pt',
  'paa',
  'pyv',
  'pyx',
  'pq',
  'pss',
  'poz',
  'pox',
  'psw',
  'ptg',
  'pty',
  'pst',
  'psis',
  'ptp',
  'pts',
  'ptt',
  'pto',

  // Q
  'qn',
  'qq',
  'qy',
  'qh',

  // R
  'raa',
  'rc',
  'rb',
  'reaa',
  'rh',
  'rj',
  'rq',
  'rr',
  'rv',
  'rw',
  'rx',
  'rz',
  'ry',
  'rl',
  'rm',
  'rs',
  'rn',
  'rp',
  'rt',
  'rd',

  // S
  'sr',
  'ss',
  'saa',
  'sth',

  // T
  'tc',
  'tm',
  'tst',
  'tt',
  'tz',
  'taa',
  'td',
  'ty',
  'tl',
  'tn',

  // U
  'ubc',
  'ua',
  'uay',
  'uak',
  'uan',
  'ubb',
  'uc',
  'ud',
  'ue',
  'ueu',
  'ueue',

  // V
  'vv',
  'vaa',
  'vox',
  'vr',
  'vug',
  'vs',

  // W
  'ww',
  'wn',
  'ws',
  'waa',
  'wee',
  'wf',
  'wg',
  'wj',
  'wk',
  'wm',
  'wl',
  'wpm',
  'wc',

  // X
  'xx',
  'xr',
  'xc',
  'xd',

  // Y
  'yy',
  'ym',
  'yt',
  'yl',
  'yn',
  'yr',
  'ys',
  'yq',
  'yd',
  'yx',

  // Z
  'zz',
];

/**
 * Also never allowed as a multi-letter required prefix (same exact-match rule).
 * Kept separate from ALL for clarity / future tuning.
 */
const LAST_LETTER_ONLY = [
  'eue',
  'eueu',
  'gles',
  'lly',
  'llyn',
  'ead',
  'pt',
  'ps',
  'pn',
  'ts',
  'ec',
  'wy',
  'wu',
  'ght',
  'nda',
  'nde',
  'rf',
  'rk',
  'ught',
  'ugh',
  'gp',
  'oe',
  'eb',
  'if',
  'gu',
  'll',
  'ig',
  'ini',
  'ny',
  'shes',
  'oos',
  'ish',
  'ishe',
  'ishes',
  'oot',
  'oots',
  'urs',
  'tion',
  'taum',
];

/** @type {Set<string>} */
const BLOCKED = new Set([...ALL, ...LAST_LETTER_ONLY]);

/** Consonants — doubled as a required prefix (mm, tt, …) is always banned. */
const CONSONANT = /[bcdfghjklmnpqrstvwxyz]/;

/**
 * True when `prefix` is exactly a blacklisted required-prefix string,
 * or a double-consonant digraph (mm, ff, gg, …).
 * @param {string} prefix
 */
export function isBlockedPrefix(prefix) {
  if (!prefix) return false;
  const p = String(prefix).toLowerCase();
  if (BLOCKED.has(p)) return true;
  // Rule: double consonants end lots of words but barely start any.
  if (p.length === 2 && p[0] === p[1] && CONSONANT.test(p[0])) return true;
  return false;
}

export const PREFIX_BLACKLIST = {
  all: ALL,
  lastLetterOnly: LAST_LETTER_ONLY,
  blocked: BLOCKED,
};
