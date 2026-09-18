# Connecting a trained fly to shiritori.lol

`python -m sim3d.serve` opens a real Chromium tab of
[shiritori.lol](https://shiritori.lol/) and paints it on the desk monitor.

Credentials come from `.env` (`SHIRITORI_EMAIL`, `SHIRITORI_PASSWORD`). The fly
can start Casual / Pro / Featherine / Lambdadelta. It types on the live page.
It never clicks Skip or forfeit. It only gives up when the local dictionary
has no remaining legal word for the live prefix — that is a model failure.

MaleCNS can still emit keys through `/api/brain`. Training itself still uses
`shiritori/environment.py`, not the website.
