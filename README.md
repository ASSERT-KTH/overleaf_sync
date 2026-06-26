# overleaf-sync

Bidirectional live sync between Overleaf and an AI coding agent — without going through Git.

## The problem

Overleaf's Git integration is painful when two beings are editing simultaneously (human-human, human-agent): every push risks a conflict, and pushing while someone is typing in the browser fails outright. The Git bridge was not designed for high-frequency, automated writes.

## The solution

This project uses the Overleaf real-time collaboration protocol and speaks it directly. Changes flow both ways in real time — the same way a second browser tab would — so there are no commits, no pushes, and no conflicts.

## Status

Early MVP. The core sync loop is fully working:

- reads the current document state from Overleaf
- applies agent edits via `applyOtUpdate` (insert / delete ops)
- polls for remote changes and merges them locally
- triggers LaTeX compile on demand

Not yet covered: folder operations.

## Usage

```bash
pip install requests websocket-client

python3 overleaf_sync.py <project_id>
```

Get `overleaf_session2` from your browser cookies while logged in to overleaf.com. Get `project_id` from the URL (`/project/<id>`).
