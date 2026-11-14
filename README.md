# llmtest — multi-model playground

> Built with Python · JavaScript · Shell (Python, Flask).

A side-by-side playground: one prompt, many models, answers streaming into parallel panes with latency and token statistics next to each. Useful for seeing how two models actually differ on the same question.

## Installation

```bash
git clone https://github.com/i-william-hr/llmtest.git
cd llmtest
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

## Usage

```bash
python3 llmtest.py
```

The llmtest.py entry point runs standalone.

## Tests

```bash
python3 -m unittest discover
```

## License

MIT License — see [LICENSE](LICENSE).
