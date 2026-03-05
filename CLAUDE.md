# Recipe Pipeline

## Conventions

- Use `data_nosync/` (not `data/`) for bulk data directories to prevent Dropbox from syncing thousands of files upstream. Any directory suffix `_nosync` is excluded from Dropbox sync.
- Prompts live in `prompts/` (git-tracked, Dropbox-synced) with explicit version naming (v1.txt, v2.txt, etc.) so previous versions are always accessible without git archaeology.
