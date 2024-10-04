# automated-ai-scientist
An automated ai scientist developed during the Gradient Descent Hackathon, Stockholm, 27-28 sep 2024

## Managing the Project with uv

This project uses `uv` for dependency management and virtual environment handling. Here's how to use it:

**Install uv:**
If you haven't installed `uv` yet, you can do so using:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Install dependencies and create virtual environment:**
```
uv sync
```

**Alternatively, let uv install the dependencies when you run the project:**
```
uv run <script>.py
```

**Adding dependencies:**
```
uv add <package_name>
```