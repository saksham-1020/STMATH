# \"\"\"Generates a simple API README from aimath __all__ and docstrings\"\"\"

import inspect
import stmath_old

def make_md(path='API.md'):
    lines = ['# stMATH API\n']
    for name in sorted(getattr(stmath_old, '__all__', [])):
        if hasattr(stmath_old, name):
            obj = getattr(stmath_old, name)
            doc = inspect.getdoc(obj) or ''
            lines.append(f'## {name}\\n')
            lines.append('`\n' + (doc.splitlines()[0] if doc else '') + '\n`\n')
    open(path, 'w', encoding='utf-8').write('\\n'.join(lines))
    print('Wrote', path)

if __name__=='__main__':
    make_md()