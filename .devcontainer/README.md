# Development container

This repository uses the open Dev Container specification and can be opened in
both Zed and Visual Studio Code.

## Zed

Requirements:

- Zed 0.218 or newer
- Docker or Podman available in `PATH`

Open the repository in Zed. When prompted, choose **Open in Container**. If the
prompt was dismissed, run **Project: Open Remote** from the command palette and
choose **Connect Dev Container**.

Zed does not currently rebuild a running container automatically after
`.devcontainer/devcontainer.json` changes. Stop or remove the existing
container, then reopen the project in its Dev Container.

Python support, basedpyright, Ruff, and debugpy are built into Zed. The
`python-requirements` extension is requested by `devcontainer.json` to add
syntax highlighting for `requirements.txt`.

Useful project tasks are available through the task picker:

- `Tests: pytest`
- `Environment: verify audio stack`

## Visual Studio Code

Run **Dev Containers: Reopen in Container**. The existing VS Code extension and
Python testing configuration remains available under `customizations.vscode`.
