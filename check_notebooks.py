from pathlib import Path
import ast, json

for path in Path('.').glob('*.ipynb'):
    nb = json.loads(path.read_text())
    for i, cell in enumerate(nb.get('cells', [])):
        if cell.get('cell_type') == 'code':
            src = ''.join(cell.get('source', []))
            try:
                ast.parse(src)
            except SyntaxError as exc:
                raise SystemExit(f'{path}: code cell {i}: {exc}')
    print(f'OK: {path}')
