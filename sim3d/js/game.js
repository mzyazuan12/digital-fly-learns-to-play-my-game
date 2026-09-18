/** Last scraped shiritori.lol HUD state. Empty until the live browser answers. */

export function emptySnap() {
  return {
    phase: "boot",
    prefix: "",
    lastWord: "",
    buffer: "",
    score: 0,
    lives: null,
    botLives: null,
    tries: null,
    lastFly: "",
    lastBot: "",
    message: "Opening shiritori.lol…",
    winner: null,
    flash: "",
  };
}
