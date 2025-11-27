# \"\"\"Generates a simple API README from aimath __all__ and docstrings\"\"\"

import inspect
import stmath

def make_md(path='API.md'):
    lines = ['# stMATH API\n']
    for name in sorted(getattr(stmath, '__all__', [])):
        if hasattr(stmath, name):
            obj = getattr(stmath, name)
            doc = inspect.getdoc(obj) or ''
            lines.append(f'## {name}\\n')
            lines.append('`\n' + (doc.splitlines()[0] if doc else '') + '\n`\n')
    open(path, 'w', encoding='utf-8').write('\\n'.join(lines))
    print('Wrote', path)

if __name__=='__main__':
    make_md()
